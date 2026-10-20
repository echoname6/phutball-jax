#!/usr/bin/env python3
"""Synthetic checks of curriculum_pp.gate (python -m llm_train.test_gate). Each case builds validation rows for a
"best" and a "new" model with chosen rates and states the decision the gate should make. The cases that change how
often the model finishes are the ones that caught gate mistakes: v3.0 counted NO WIN answers and v3.3 counted
finished errors, and both counts move with the finishing rate alone (gated GRPO p4: finished 0.454 -> 0.604 rejected).
"""
from __future__ import annotations

import math
import random

from llm_train.curriculum_pp import gate, gate_curve


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


CUTS = (2048, 4096, 8192, 16384)


def curve(seed, speed=1.0, p_ok=0.88, peak=None, nowin_err=0.3, n_win=700, n_neg=100):
    """load_curve()-shaped rows: each item finishes at a lognormal token count (median 5k / speed; ~25% never within
    16k at speed 1). One latent u per item: its finished answer is right if u < p_ok (errors: false NO WIN with share
    nowin_err of the wrong win answers, else a broken chain); if it never finishes, forced at 16k it is right if
    u < peak (default p_ok: the same answers whether finished or forced), and forced at a smaller budget right with a
    share that grows with the budget."""
    peak = p_ok if peak is None else peak
    rng = random.Random(seed); C = {B: {} for B in CUTS}
    for i in range(n_win + n_neg):
        win = i < n_win; t = 5000 / speed * math.exp(rng.gauss(0, 0.9))
        u = rng.random(); ok = u < p_ok; err = "missed win (said NO WIN)" if (win and rng.random() < nowin_err) else \
            ("the sequence ends without reaching the goal" if win else "claimed a win: the sequence ends without reaching the goal")
        u_force = u
        for B in CUTS:
            r = {"id": i, "task": "win", "answers": {"win": win}}
            if t <= B:
                r.update(correct=ok, outcome=("found a win" if win else "correct NO WIN") if ok else err)
                r.update(forced_correct=r["correct"], forced_outcome=r["outcome"])
            else:
                fq = peak * (0.55 + 0.45 * math.log2(B / 2048) / 3)               # forced accuracy grows with budget
                r.update(correct=False, outcome="unparsed", forced_correct=u_force < fq,
                         forced_outcome="found a win" if u_force < fq else "missed win (said NO WIN)")
            C[B][i] = r
    return C


CURVE_CASES = [   # (description, best kwargs, new kwargs, should promote)
    ("curve: same model, new seed", dict(), dict(), True),
    ("curve: same answers, ~2x faster (same peak)", dict(), dict(speed=2.0), True),
    ("curve: same answers, 2x slower", dict(), dict(speed=0.5), False),
    ("curve: faster but a lower ceiling (right answers 0.88 -> 0.76)", dict(), dict(speed=1.6, p_ok=0.76), False),
    ("curve: same speed, higher ceiling (right answers 0.88 -> 0.94)", dict(), dict(p_ok=0.94), True),
    ("curve: faster; the ~15% still cut off at 16k forced worse (0.88 -> 0.70): ceiling -2 points, within noise",
     dict(), dict(speed=1.3, peak=0.70), True),
    ("curve: same speed, the ~25% cut off at 16k forced much worse (0.88 -> 0.25): a hidden ceiling loss", dict(),
     dict(peak=0.25), False),
    ("curve: faster by committing to wrong answers (accuracy 0.85 -> 0.65)", dict(), dict(speed=2.0, p_ok=0.65), False),
]


def main():
    fails = 0
    for k, (desc, b, n, want) in enumerate(CASES):
        ok, why = gate(rows(100 + k, **b), rows(200 + k, **n))
        tag = "ok  " if ok == want else "FAIL"; fails += ok != want
        print(f"{tag} {'promote' if want else 'reject ':7s} | {desc}\n       {why[:200]}")
    for k, (desc, b, n, want) in enumerate(CURVE_CASES):
        ok, why = gate_curve(curve(300 + k, **b), curve(400 + k, **n))
        tag = "ok  " if ok == want else "FAIL"; fails += ok != want
        print(f"{tag} {'promote' if want else 'reject ':7s} | {desc}\n       {why[:300]}")
    n = len(CASES) + len(CURVE_CASES)
    print(f"{n - fails}/{n} cases as expected"); raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
