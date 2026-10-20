#!/usr/bin/env python3
"""Synthetic checks of curriculum_pp.gate (python -m llm_train.test_gate). Each case builds validation rows for a
"best" and a "new" model with chosen rates and states the decision the gate should make. The cases that change how
often the model finishes are the ones that caught gate mistakes: v3.0 counted NO WIN answers and v3.3 counted
finished errors, and both counts move with the finishing rate alone (gated GRPO p4: finished 0.454 -> 0.604 rejected).
"""
from __future__ import annotations

import random

from llm_train.curriculum_pp import gate


def rows(seed, cut, fnw, bad_chain=0.01, neg_claim=0.03, n_win=700, n_neg=100, forced_ok=0.75):
    """cut: share of wins cut off (forced right with forced_ok); fnw: share of wins answered NO WIN (finished);
    bad_chain: share of wins answered with a chain that does not win; neg_claim: share of no-win puzzles claimed."""
    rng = random.Random(seed); out = {}
    for i in range(n_win + n_neg):
        win = i < n_win; u = rng.random(); r = {"id": i, "task": "win", "answers": {"win": win}}
        if win:
            if u < cut:
                ok = rng.random() < forced_ok
                r.update(correct=False, outcome="unparsed", forced_correct=ok,
                         forced_outcome="found a win" if ok else "missed win (said NO WIN)")
            elif u < cut + fnw: r.update(correct=False, outcome="missed win (said NO WIN)")
            elif u < cut + fnw + bad_chain: r.update(correct=False, outcome="the sequence ends without reaching the goal")
            else: r.update(correct=True, outcome="found a win")
        else:
            if u < neg_claim: r.update(correct=False, outcome="claimed a win: the sequence ends without reaching the goal")
            else: r.update(correct=True, outcome="correct NO WIN")
        r.setdefault("forced_correct", r["correct"]); r.setdefault("forced_outcome", r["outcome"])
        out[i] = r
    return out


CASES = [   # (description, best kwargs, new kwargs, should promote)
    ("same model, new seed", dict(cut=0.3, fnw=0.06), dict(cut=0.3, fnw=0.06), True),
    ("finishes much MORE, same error rate among finished (gated GRPO p3 -> p4)",
     dict(cut=0.47, fnw=0.043), dict(cut=0.22, fnw=0.063), True),
    ("finishes much LESS, same error rate among finished", dict(cut=0.22, fnw=0.063), dict(cut=0.40, fnw=0.048), False),
    ("false NO WIN rate doubles at the same finishing rate", dict(cut=0.3, fnw=0.06), dict(cut=0.3, fnw=0.16), False),
    ("false NO WINs turned into broken chains, accuracy flat (cheaper error: utility up)", dict(cut=0.3, fnw=0.12, bad_chain=0.01),
     dict(cut=0.3, fnw=0.02, bad_chain=0.12), True),
    ("false NO WINs turned into MORE broken chains, accuracy down a little", dict(cut=0.3, fnw=0.06, bad_chain=0.01),
     dict(cut=0.3, fnw=0.01, bad_chain=0.20), False),
    ("finishes more hard puzzles: accuracy up a lot, more false claims (gated GRPO p9 -> p10)",
     dict(cut=0.21, fnw=0.05, bad_chain=0.03), dict(cut=0.15, fnw=0.04, bad_chain=0.06), True),
    ("false win claims on no-win puzzles surge (v4 r15-like)", dict(cut=0.3, fnw=0.06, neg_claim=0.03),
     dict(cut=0.3, fnw=0.02, neg_claim=0.35), False),
]


def main():
    fails = 0
    for k, (desc, b, n, want) in enumerate(CASES):
        ok, why = gate(rows(100 + k, **b), rows(200 + k, **n))
        tag = "ok  " if ok == want else "FAIL"; fails += ok != want
        print(f"{tag} {'promote' if want else 'reject ':7s} | {desc}\n       {why[:200]}")
    print(f"{len(CASES) - fails}/{len(CASES)} cases as expected"); raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
