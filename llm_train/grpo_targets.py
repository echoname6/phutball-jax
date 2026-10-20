#!/usr/bin/env python3
"""Target the next gated-GRPO phase at what the model knows but does not finish (grpo_gated_colab.ipynb).

The gap between forced accuracy (an answer forced at the token cap) and finished accuracy (an answer the model gave by
itself), per puzzle bucket on the current best's validation run, is knowledge the model has but does not commit within
the budget - what GRPO with a cut-off cost can turn into finished wins. After gated GRPO p9 (finished 0.64, plateaued):
1-2 jump wins finished ~90%+ (forced 95-100%), while 4-jump wins, 4-jump backward traps and forced / prevent placements
had 15-23-point gaps, and uniform sampling spent most steps on the mastered buckets.

  bucket share  ~ max(forced - finished, FLOOR) + headroom * 4 q (1 - q)      (q = forced accuracy; whatever the
                  bucket's size)   from <best-eval>.summary / -forced
  puzzle weight = 0.15 if finished correct in >= 90% of its past GRPO samples (mastered), else 1 + its cut-off share
--headroom (default 0.5): the gap alone starves buckets the model cannot solve even when forced (gated GRPO p12:
prevent 27% finished / 30% forced, 3.8% of the draw); 4 q (1 - q) adds the learnable headroom (largest at q = 0.5,
zero at q = 0 or 1, where every group would be all wrong or all right and GRPO gets no signal).
w_prior per drawn puzzle = (its bucket's share of validation) / (its share of this draw), clipped to [0.25, 4]:
grpo_pilot --prior-correct multiplies the rollout rewards by it (Dr. GRPO without reward scaling: the advantage scales
by w), so targeting chooses where the steps go but the policy still learns the validation base rates. Without it, a
draw of 29% no-win puzzles (12.5% of validation) taught gated GRPO p14 to say NO WIN sooner: near-miss negatives
finished 68% -> 81%, but forced false NO WIN on real wins 48 -> 65 (4-jump backward traps forced 69% -> 62%).
Writes --n puzzles drawn with replacement by bucket weight x puzzle weight (grpo_pilot shuffles and reads the first
prompts x max-steps of them), and prints the bucket shares.

  python -m llm_train.grpo_targets --puzzles PUZ --best-eval EVAL/v8 --rollouts 'OUT/p*/rollouts.jsonl' --out OUT/p10/puzzles.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter, defaultdict

KEY_TO_BUCKET = {"win: positive, 1-jump": "win1", "win: positive, 2-jump": "win2", "win: positive, 3-jump": "win3",
                 "win: positive, 4-jump": "win4", "win: backward-trap, 2-jump": "win2b",
                 "win: backward-trap, 3-jump": "win3b", "win: backward-trap, 4-jump": "win4b",
                 "win: near-miss negative": "nowin", "forced placement": "forced", "block placement": "block",
                 "prevent placement": "prevent"}
FLOOR = 0.03


def bucket_weights(prefix: str, headroom: float = 0.0) -> tuple[dict, dict]:
    """(bucket weight, bucket share of the validation set)."""
    fin = json.load(open(f"{prefix}.summary.json")); frc = json.load(open(f"{prefix}-forced.summary.json"))
    keys = [(k, b) for k, b in KEY_TO_BUCKET.items() if k in fin and k in frc]
    w = {}
    for k, b in keys:
        q = frc[k]["accuracy"]; w[b] = max(q - fin[k]["accuracy"], FLOOR) + headroom * 4 * q * (1 - q)
    n = {b: fin[k].get("n", 0) for k, b in keys}; tot = sum(n.values()) or 1
    return w, {b: v / tot for b, v in n.items()}


def history(pattern: str) -> dict:
    """puzzle id -> (finished-correct share, cut-off share) over every logged GRPO sample."""
    h = defaultdict(lambda: [0, 0, 0])
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line); x = h[r.get("id")]
            x[0] += 1; x[1] += bool(r.get("correct")) and not r.get("trunc"); x[2] += bool(r.get("trunc"))
    return {k: (v[1] / v[0], v[2] / v[0]) for k, v in h.items() if v[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True); ap.add_argument("--best-eval", required=True)
    ap.add_argument("--rollouts", default=""); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--headroom", type=float, default=0.5, help="weight of 4 q (1 - q) on top of the gap (0: gap only)")
    a = ap.parse_args(); rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.puzzles)]
    bw, vshare = bucket_weights(a.best_eval, a.headroom); hist = history(a.rollouts) if a.rollouts else {}
    count = Counter(r.get("bucket") for r in rows); w = []
    for r in rows:                                                   # bucket share ~ its gap, whatever its size
        b = r.get("bucket"); p = hist.get(r["id"])
        pw = 1.0 if p is None else (0.15 if p[0] >= 0.9 else 1.0 + p[1])
        w.append(bw.get(b, FLOOR) / count[b] * pw)
    pick = rng.choices(rows, weights=w, k=a.n)
    share = Counter(r.get("bucket") for r in pick); have = Counter(r.get("bucket") for r in rows)
    wp = {b: min(4.0, max(0.25, vshare.get(b, 0) / (share[b] / a.n))) if share[b] and vshare.get(b) else 1.0 for b in share}
    with open(a.out, "w") as f:
        for r in pick: f.write(json.dumps({**r, "w_prior": round(wp[r.get("bucket")], 4)}) + "\n")
    label = "forced - finished" + (f" + {a.headroom:g} x 4q(1-q)" if a.headroom else "")
    print(f"bucket weight ({label}) | puzzles available | share of this phase's draw | validation share -> w_prior")
    for b in sorted(bw, key=lambda b: -bw[b]):
        print(f"  {b:8s} {bw[b]:5.2f} | {have[b]:4d} | {share[b] / a.n:5.1%} | {vshare.get(b, 0):5.1%} -> {wp.get(b, 1.0):4.2f}")
    seen = sum(r["id"] in hist for r in rows); mast = sum(1 for r in rows if hist.get(r["id"], (0, 0))[0] >= 0.9)
    print(f"{len(rows)} puzzles, {seen} with GRPO history, {mast} mastered (down-weighted); {a.n} drawn -> {a.out}")


if __name__ == "__main__":
    main()
