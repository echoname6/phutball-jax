#!/usr/bin/env python3
"""Automatic curriculum for expert iteration: which puzzles each round samples, which are paroled, and which model
generates the next round. All rules are fixed up front so rounds run unattended (no hand-tuning between rounds).

Buckets ("similarity groups"): win1..win4 (with a "b" suffix for the backward-trap family), nowin, block, forced,
prevent.

Each round:
  allocate   N puzzles across buckets in proportion to the expert-iteration priority of the bucket's solve rate p
             (the share of samples that produced a verified solution last time it was sampled):
                 priority(p) = (1 - (1-p)^k) * (1 - p)        chance of >= 1 success in k samples x room to improve
             (floor FLOOR so no bucket starves; buckets not yet sampled get the mean priority). For GRPO the
             matching priority would be p(1-p), the variance of the group reward. Within a bucket, puzzles never
             attempted come first, then the least recently attempted. SPOT of N goes to paroled puzzles at random.
  parole     a puzzle with no verified solution in its last PAROLE_AFTER attempts leaves the pool; the bucket's rate
             at that moment is recorded.
  release    a paroled puzzle re-enters when its bucket's rate has risen by RELEASE_UPLIFT over the recorded rate,
             or when a spot check solves it.
  gate       the new model becomes the generator for the next round only if its mean finished pass@1 over GATE_KEYS
             beats the current best and no GATE_KEYS group's budget-forced accuracy drops more than MAX_DROP below the
             current best's (an AlphaGo-Zero-style gate). Otherwise the next round samples from the previous best.
State: one JSON file (resumable; each round is applied once).

  python -m llm_train.curriculum allocate --state S --pool P --n 600 --round 1 --out R1/ids.json
  python -m llm_train.curriculum update   --state S --round 1 --outcomes R1/outcomes.jsonl
  python -m llm_train.curriculum gate     --state S --round 1 --new EVAL/r1-forced... (see main)
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

K = 4
FLOOR = 0.02
SPOT = 0.05
PAROLE_AFTER = 2
RELEASE_UPLIFT = 0.15
MAX_DROP = 0.10
MIN_SAMPLES = 16          # a bucket's rate is updated only from at least this many regular (non-spot-check) samples
GATE_KEYS = ("win: positive, 1-jump", "win: positive, 2-jump", "win: positive, 3-jump", "win: positive, 4-jump",
             "win: near-miss negative", "block placement", "forced placement")   # prevent: one-rule shortcut, reported only


def bucket(row: dict) -> str:
    sub = row["subtask"]
    if sub == "win_pos":
        m = row["item"]["meta"]
        return f"win{m.get('J', 1)}" + ("b" if m.get("family") == "back" else "")
    return {"win_neg": "nowin"}.get(sub, sub)


def load_pool(path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f]


def priority(p: float, k: int = K) -> float:
    return max(FLOOR, (1 - (1 - p) ** k) * (1 - p))


class State:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else \
            {"puzzles": {}, "rates": {}, "applied": [], "best": None, "best_eval": None, "log": []}

    def save(self):
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(self.d)); tmp.replace(self.path)

    def init_pool(self, pool):
        for r in pool:
            self.d["puzzles"].setdefault(r["id"], {"bucket": bucket(r), "status": "active", "attempts": []})

    def rate(self, b: str):
        """Latest measured solve rate of bucket b (None if never sampled)."""
        for rnd in sorted(self.d["rates"], key=int, reverse=True):
            if b in self.d["rates"][rnd]: return self.d["rates"][rnd][b]
        return None

    def allocate(self, n: int, rnd: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        P = self.d["puzzles"]
        active = defaultdict(list); paroled = []
        for pid, s in P.items():
            (active[s["bucket"]] if s["status"] == "active" else paroled).append(pid)
        n_spot = min(len(paroled), int(round(SPOT * n)))
        known = [priority(self.rate(b)) for b in active if self.rate(b) is not None]
        default = sum(known) / len(known) if known else 1.0
        w = {b: (priority(self.rate(b)) if self.rate(b) is not None else default) for b in active if active[b]}
        total = sum(w.values()); budget = n - n_spot
        alloc = {b: min(len(active[b]), int(round(budget * w[b] / total))) for b in w}
        short = budget - sum(alloc.values())                  # hand leftovers to buckets with room, by priority
        for b in sorted(w, key=lambda b: -w[b]):
            if short <= 0: break
            add = min(short, len(active[b]) - alloc[b]); alloc[b] += add; short -= add
        ids = []
        for b, m in alloc.items():
            pool = active[b][:]; rng.shuffle(pool)
            pool.sort(key=lambda pid: (len(P[pid]["attempts"]), P[pid]["attempts"][-1][0] if P[pid]["attempts"] else -1))
            ids += pool[:m]
        rng.shuffle(paroled); ids += paroled[:n_spot]
        self.d["log"].append({"round": rnd, "event": "allocate", "alloc": alloc, "spot_checks": n_spot,
                              "weights": {b: round(v, 3) for b, v in w.items()}})
        return ids

    def update(self, rnd: int, outcomes: list[dict]):
        """outcomes: [{id, samples, verified, ...}] from expert_iter. Applied once per round."""
        if rnd in self.d["applied"]: return
        P = self.d["puzzles"]; per = defaultdict(lambda: [0, 0])
        for o in outcomes:
            s = P[o["id"]]; s["attempts"].append([rnd, o["verified"], o["samples"]])
            if s["status"] == "active":                       # spot checks of paroled puzzles do not move bucket rates
                per[s["bucket"]][0] += min(o["verified"], o["samples"]); per[s["bucket"]][1] += o["samples"]
        self.d["rates"][str(rnd)] = {b: v / n for b, (v, n) in per.items() if n >= MIN_SAMPLES}
        released = paroled_now = 0
        for o in outcomes:                                    # spot checks that succeed are released at once
            s = P[o["id"]]
            if s["status"] == "paroled" and o["verified"] > 0:
                s["status"] = "active"; s["released_round"] = rnd; released += 1
        for pid, s in P.items():
            recent = [x for x in s["attempts"] if x[0] > s.get("released_round", -1)]   # only since the last release
            if s["status"] == "active" and len(recent) >= PAROLE_AFTER and \
                    all(x[1] == 0 for x in recent[-PAROLE_AFTER:]):
                s["status"] = "paroled"; s["paroled_round"] = rnd; s["paroled_rate"] = self.rate(s["bucket"]) or 0.0
                paroled_now += 1
            elif s["status"] == "paroled" and self.rate(s["bucket"]) is not None and \
                    self.rate(s["bucket"]) >= s.get("paroled_rate", 0.0) + RELEASE_UPLIFT:
                s["status"] = "active"; s["released_round"] = rnd; released += 1
        self.d["applied"].append(rnd)
        st = Counter(s["status"] for s in P.values())
        self.d["log"].append({"round": rnd, "event": "update", "rates": {b: round(v, 3) for b, v in self.d["rates"][str(rnd)].items()},
                              "paroled_now": paroled_now, "released": released, "status": dict(st)})


FAMILIES = {"wins": ("win: positive, 1-jump", "win: positive, 2-jump", "win: positive, 3-jump", "win: positive, 4-jump"),
            "no-win": ("win: near-miss negative",), "placements": ("block placement", "forced placement")}


def pooled(S: dict, keys) -> float | None:
    """n-weighted accuracy over several summary groups (pooling keeps the drop check out of small-sample noise)."""
    num = sum(S[k]["accuracy"] * S[k]["n"] for k in keys if k in S); den = sum(S[k]["n"] for k in keys if k in S)
    return num / den if den else None


def gate(best: dict | None, new: dict, best_forced: dict | None, new_forced: dict) -> tuple[bool, str]:
    """best/new: finished summaries; *_forced: budget-forced summaries (llm_bench.run), same eval protocol.
    Promote if the mean finished pass@1 over GATE_KEYS improves and no family's pooled forced accuracy
    (wins, no-win, placements) drops more than MAX_DROP. Returns (promote, reason)."""
    score = lambda S: sum(S[k]["accuracy"] for k in GATE_KEYS if k in S) / max(1, sum(k in S for k in GATE_KEYS))
    if best is None: return True, "no previous model"
    drops = {}
    for fam, keys in FAMILIES.items():
        b, n = pooled(best_forced, keys), pooled(new_forced, keys)
        if b is not None and n is not None: drops[fam] = round(b - n, 3)
    worst = max(drops.items(), key=lambda kv: kv[1]) if drops else (None, 0)
    if worst[1] > MAX_DROP: return False, f"forced accuracy on {worst[0]} dropped {worst[1]:.0%} (> {MAX_DROP:.0%})"
    if score(new) <= score(best): return False, f"mean finished pass@1 {score(new):.3f} <= best {score(best):.3f}"
    return True, f"mean finished pass@1 {score(best):.3f} -> {score(new):.3f}; largest forced drop {worst[1]:.0%}"


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("allocate"); a1.add_argument("--state", required=True); a1.add_argument("--pool", required=True)
    a1.add_argument("--n", type=int, default=600); a1.add_argument("--round", type=int, required=True); a1.add_argument("--out", required=True)
    a2 = sub.add_parser("update"); a2.add_argument("--state", required=True); a2.add_argument("--round", type=int, required=True)
    a2.add_argument("--outcomes", required=True)
    a3 = sub.add_parser("gate"); a3.add_argument("--state", required=True); a3.add_argument("--round", type=int, required=True)
    a3.add_argument("--new", required=True, help="eval prefix of the new model (…/name -> name.summary.json, name-forced.summary.json)")
    a3.add_argument("--adapter", required=True); a3.add_argument("--init-best", default=None, help="eval prefix of the starting model")
    a3.add_argument("--init-adapter", default=None)
    a4 = sub.add_parser("show"); a4.add_argument("--state", required=True)
    a = ap.parse_args(); st = State(a.state)
    if a.cmd == "allocate":
        st.init_pool(load_pool(a.pool))
        ids = st.allocate(a.n, a.round, seed=1000 + a.round)
        Path(a.out).write_text(json.dumps(ids)); st.save()
        print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "update":
        st.update(a.round, [json.loads(l) for l in open(a.outcomes)]); st.save()
        print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "gate":
        if st.d["best"] is None and a.init_best:
            st.d["best"], st.d["best_eval"] = a.init_adapter, a.init_best
        ld = lambda pfx, sfx: json.load(open(f"{pfx}{sfx}"))
        best = ld(st.d["best_eval"], ".summary.json") if st.d["best_eval"] else None
        bestf = ld(st.d["best_eval"], "-forced.summary.json") if st.d["best_eval"] else None
        ok, why = gate(best, ld(a.new, ".summary.json"), bestf, ld(a.new, "-forced.summary.json"))
        if ok: st.d["best"], st.d["best_eval"] = a.adapter, a.new
        st.d["log"].append({"round": a.round, "event": "gate", "promoted": ok, "reason": why, "best": st.d["best"]})
        st.save(); print(("PROMOTED: " if ok else "kept previous best: ") + why); print("best:", st.d["best"])
    else:
        print(json.dumps({k: v for k, v in st.d.items() if k != "puzzles"}, indent=1)[:6000])
        print(dict(Counter(s["status"] for s in st.d["puzzles"].values())))


if __name__ == "__main__":
    main()
