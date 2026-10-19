#!/usr/bin/env python3
"""Target the next gated-GRPO phase at what the model knows but does not finish (grpo_gated_colab.ipynb).

The gap between forced accuracy (an answer forced at the token cap) and finished accuracy (an answer the model gave by
itself), per puzzle bucket on the current best's validation run, is knowledge the model has but does not commit within
the budget - what GRPO with a cut-off cost can turn into finished wins. After gated GRPO p9 (finished 0.64, plateaued):
1-2 jump wins finished ~90%+ (forced 95-100%), while 4-jump wins, 4-jump backward traps and forced / prevent placements
had 15-23-point gaps, and uniform sampling spent most steps on the mastered buckets.

  bucket share  ~ max(forced - finished, FLOOR)  (whatever the bucket's size)   from <best-eval>.summary / -forced
  puzzle weight = 0.15 if finished correct in >= 90% of its past GRPO samples (mastered), else 1 + its cut-off share
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


def bucket_weights(prefix: str) -> dict:
    fin = json.load(open(f"{prefix}.summary.json")); frc = json.load(open(f"{prefix}-forced.summary.json"))
    return {b: max(frc[k]["accuracy"] - fin[k]["accuracy"], FLOOR) for k, b in KEY_TO_BUCKET.items() if k in fin and k in frc}


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
    a = ap.parse_args(); rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(a.puzzles)]
    bw = bucket_weights(a.best_eval); hist = history(a.rollouts) if a.rollouts else {}
    count = Counter(r.get("bucket") for r in rows); w = []
    for r in rows:                                                   # bucket share ~ its gap, whatever its size
        b = r.get("bucket"); p = hist.get(r["id"])
        pw = 1.0 if p is None else (0.15 if p[0] >= 0.9 else 1.0 + p[1])
        w.append(bw.get(b, FLOOR) / count[b] * pw)
    pick = rng.choices(rows, weights=w, k=a.n)
    with open(a.out, "w") as f:
        for r in pick: f.write(json.dumps(r) + "\n")
    share = Counter(r.get("bucket") for r in pick); have = Counter(r.get("bucket") for r in rows)
    print("bucket weight (forced - finished) | puzzles available | share of this phase's draw")
    for b in sorted(bw, key=lambda b: -bw[b]):
        print(f"  {b:8s} {bw[b]:5.2f} | {have[b]:4d} | {share[b] / a.n:5.1%}")
    seen = sum(r["id"] in hist for r in rows); mast = sum(1 for r in rows if hist.get(r["id"], (0, 0))[0] >= 0.9)
    print(f"{len(rows)} puzzles, {seen} with GRPO history, {mast} mastered (down-weighted); {a.n} drawn -> {a.out}")


if __name__ == "__main__":
    main()
