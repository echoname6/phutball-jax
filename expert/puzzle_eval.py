"""Score a policy on the frozen held-out puzzle set (expert_data/puzzle_eval.npz), by TRUE difficulty.

  first-move accuracy  the policy's first action is one of the winning first jumps
  chain success        the policy plays the whole turn itself (jumps, halts) and the side to move wins this turn

A policy is any object with .action(state) -> micro-action (expert/arena.py conventions; NetPolicy for networks).

  python -m expert.puzzle_eval expert
  python -m expert.puzzle_eval random
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import State, legal_actions, step  # noqa: E402


def load_set(path=ROOT / "expert_data" / "puzzle_eval.npz"):
    return list(np.load(path, allow_pickle=True)["recs"])


def evaluate(policy, recs, rows=21, cols=15, max_steps=40, limit_per_diff=None):
    n = rows * cols; first = defaultdict(lambda: [0, 0]); chain = defaultdict(lambda: [0, 0]); seen = defaultdict(int)
    for r in recs:
        d = int(r["difficulty"]); d = min(d, 5)
        if limit_per_diff and seen[d] >= limit_per_diff: continue
        seen[d] += 1
        s = State(rows, cols, [int(v) for v in r["board"]], int(r["ball"]), int(r["player"])); me = s.player
        a = policy.action(s)
        first[d][0] += int(n <= a < 2 * n and (a - n) in set(int(x) for x in r["winning_first"])); first[d][1] += 1
        won = False
        for _ in range(max_steps):
            if a not in legal_actions(s): break
            s = step(s, a)
            if s.winner: won = s.winner == me; break
            if s.player != me or not s.jumping: break       # the turn ended without a win
            a = policy.action(s)
        chain[d][0] += int(won); chain[d][1] += 1
    return {d: {"first": first[d][0] / first[d][1], "chain": chain[d][0] / chain[d][1], "n": first[d][1]} for d in sorted(first)}


def report(res, name=""):
    tot = sum(v["n"] for v in res.values())
    f = sum(v["first"] * v["n"] for v in res.values()) / tot; c = sum(v["chain"] * v["n"] for v in res.values()) / tot
    rows = " | ".join(f"{d}{'+' if d == 5 else ''}j: {v['first']:.0%}/{v['chain']:.0%}" for d, v in res.items())
    print(f"{name:12s} overall first-move {f:.0%}, chain {c:.0%}  ||  by difficulty (first/chain): {rows}", flush=True)
    return f, c


if __name__ == "__main__":
    from expert.arena import RandomPolicy
    from expert.oracle import Expert
    which = sys.argv[1] if len(sys.argv) > 1 else "expert"
    pol = {"expert": Expert(), "random": RandomPolicy(0)}[which]
    report(evaluate(pol, load_set(), limit_per_diff=150), which)
