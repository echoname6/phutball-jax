#!/usr/bin/env python3
"""Training curves from a GRPO run's rollouts.jsonl: accuracy, truncation, length and drift of the thinking per
generation batch, by subtask; plus the thinking of one correct sample early and late (to read the drift).

  python -m llm_train.analyze_rollouts RUN/rollouts.jsonl [--plot RUN/curves.png] [--window 5]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--plot", default=None); ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--examples", type=int, default=1)
    ap.add_argument("--puzzles", default=None, help="grpo_train.jsonl: adds the compression ratio vs the engine trace")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path)]
    engine = {}
    if a.puzzles:                                    # item id -> tokens of the engine's verified trace for that puzzle
        for l in open(a.puzzles):
            p = json.loads(l); engine[p["item"]["id"]] = p.get("engine_think_tokens")
    calls = defaultdict(list)
    for r in rows: calls[r["call"]].append(r)
    ks = sorted(calls); W = a.window
    print(f"{len(rows)} samples in {len(ks)} generation batches")
    print("finished-ok: finished on its own and correct (share of all samples); forced-ok: share of CUT-OFF samples whose")
    print("forced answer at the cap verifies; anytime: their mean share over the forced prefixes. Watch for forced-ok")
    print("rising while finished-ok stays flat: better forced guesses, not better search.")
    if engine: print("x-engine: median (model tokens / engine trace tokens) over finished-correct samples; 1 = as short as explicit search")
    print(f"{'batches':>9s} {'fin-ok':>7s} {'trunc':>6s} {'forced-ok':>9s} {'anytime':>8s} {'len':>6s} {'len(ok)':>8s} {'wordlike':>9s} {'nonascii':>9s} {'x-engine':>9s}  finished-ok by subtask")
    curve = []
    for i in range(0, len(ks), W):
        g = [r for k in ks[i:i + W] for r in calls[k]]; n = len(g); ok = [r for r in g if r["correct"]]
        sub = defaultdict(list)
        for r in g: sub[r["subtask"]].append(r["correct"])
        c = {"from": ks[i], "correct": sum(r["correct"] for r in g) / n, "trunc": sum(r["trunc"] for r in g) / n,
             "len": sum(r["len"] for r in g) / n, "len_ok": sum(r["len"] for r in ok) / max(len(ok), 1),
             "wordlike": sum(r["wordlike"] for r in g) / n, "nonascii": sum(r["nonascii"] for r in g) / n}
        ratios = sorted(r["len"] / engine[r["id"]] for r in ok if engine.get(r.get("id")))
        c["compress"] = ratios[len(ratios) // 2] if ratios else float("nan")
        cut = [r for r in g if r["trunc"]]
        c["forced_ok"] = sum(bool(r.get("forced_correct")) for r in cut) / len(cut) if cut else float("nan")
        c["anytime"] = sum(r.get("forced_score", float(bool(r.get("forced_correct")))) for r in cut) / len(cut) if cut else float("nan")
        curve.append(c)
        print(f"{ks[i]:>4d}-{ks[min(i + W, len(ks)) - 1]:<4d} {c['correct']:7.1%} {c['trunc']:6.1%} {c['forced_ok']:9.1%} {c['anytime']:8.2f} {c['len']:6.0f} {c['len_ok']:8.0f} "
              f"{c['wordlike']:9.2f} {c['nonascii']:9.3f} {c['compress']:9.1f}  " + " ".join(f"{k} {sum(v) / len(v):.0%}" for k, v in sorted(sub.items())))
    print("\nhow each subtask ends (all batches; top outcomes):")
    by = defaultdict(lambda: defaultdict(int))
    for r in rows: by[r["subtask"]][r["outcome"] if not r["trunc"] else "cut off"] += 1
    for sub, oc in sorted(by.items()):
        n = sum(oc.values())
        print(f"  {sub:8s} n={n:5d}  " + "; ".join(f"{k} {v / n:.0%}" for k, v in sorted(oc.items(), key=lambda kv: -kv[1])[:5]))
    for label, pool in (("EARLY", ks[:W]), ("LATE", ks[-W:])):
        ok = sorted((r for k in pool for r in calls[k] if r["correct"]), key=lambda r: r["len"])
        for r in ok[len(ok) // 2: len(ok) // 2 + a.examples]:
            print(f"\n===== {label} correct sample: {r['subtask']} {r['len']} tokens, wordlike {r['wordlike']:.2f} =====")
            t = r["text"]; print(t if len(t) < 3000 else t[:1500] + "\n[...]\n" + t[-1500:])
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4)); x = [c["from"] for c in curve]
        ax[0].plot(x, [c["correct"] for c in curve], label="finished & correct"); ax[0].plot(x, [c["trunc"] for c in curve], label="truncated")
        ax[0].plot(x, [c["forced_ok"] for c in curve], "--", label="forced-correct (of cut-off)")
        ax[0].legend(); ax[0].set_xlabel("generation batch")
        ax[1].plot(x, [c["len"] for c in curve], label="all"); ax[1].plot(x, [c["len_ok"] for c in curve], label="correct")
        ax[1].set_ylabel("completion tokens"); ax[1].legend()
        ax[2].plot(x, [c["wordlike"] for c in curve], label="word-like share"); ax[2].plot(x, [c["nonascii"] for c in curve], label="non-ASCII share")
        ax[2].legend(); fig.tight_layout(); fig.savefig(a.plot, dpi=110); print("plot ->", a.plot)


if __name__ == "__main__":
    main()
