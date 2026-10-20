#!/usr/bin/env python3
"""Accuracy-by-budget curves from llm_transfer.sweep (results/<name>_sweep.jsonl), per family and dial, model by model.

For every budget B: finished@B (answered on its own within B tokens) and forced@B (best answer within B), as percent.
Then each model against the first (paired over the same items) at the largest budget: +wins / -losses and z, for
finished and forced; that is the plateau comparison (skill), the curves' horizontal offset is the efficiency.

  python -m llm_transfer.sweep_report grpo r14 base warm [--by all|dial]
  python -m llm_transfer.sweep_report grpo r14 --ladder     (the same puzzles on the board / in numbered coordinates /
      with named lanes: <m>_ladder_sweep.jsonl + <m>_sweep.jsonl; where along the ladder each model's accuracy drops)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_transfer.run import dial_key  # noqa: E402


def load(name):
    rows = {}
    for l in open(ROOT / f"llm_transfer/results/{name}_sweep.jsonl"):
        r = json.loads(l); r["at"] = {int(k): v for k, v in r["at"].items()}; rows[r["id"]] = r
    return rows


def pz(ref, new, ids, cut, key):
    w = sum(new[i]["at"][cut][key] and not ref[i]["at"][cut][key] for i in ids)
    l = sum(ref[i]["at"][cut][key] and not new[i]["at"][cut][key] for i in ids)
    return w, l, (w - l) / math.sqrt(w + l) if w + l else 0.0


def curves(names, group_by="dial"):
    R = {n: load(n) for n in names}; ref = names[0]
    ids = sorted(set.intersection(*(set(r) for r in R.values())))
    bad = {i for i in ids if len({R[n][i]["prompt_sha"] for n in names}) > 1}
    if bad: print(f"WARNING: {len(bad)} items differ in prompt across models; dropped"); ids = [i for i in ids if i not in bad]
    cuts = sorted(set.intersection(*(set(R[n][ids[0]]["at"]) for n in names))) if ids else []
    groups = defaultdict(list)
    for i in ids:
        r = R[ref][i]; groups[(r["family"], "all")].append(i)
        if group_by == "dial": groups[(r["family"], dial_key(r["dial"]))].append(i)
    head = " ".join(f"{c // 1024:>4d}k" for c in cuts)
    print(f"accuracy by thinking budget: finished (answered by itself within B) / forced (best answer within B), %")
    for (fam, key), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1])):
        print(f"\n{fam} | {key}   (n={len(g)})            B = {head}")
        for n in names:
            fin = " ".join(f"{100 * sum(R[n][i]['at'][c]['finished'] for i in g) / len(g):5.0f}" for c in cuts)
            frc = " ".join(f"{100 * sum(R[n][i]['at'][c]['forced'] for i in g) / len(g):5.0f}" for c in cuts)
            print(f"  {n:14s} finished {fin}\n  {'':14s} forced   {frc}")
        if len(names) > 1:
            top = cuts[-1]
            for n in names[1:]:
                wf, lf, zf = pz(R[ref], R[n], g, top, "finished"); wo, lo, zo = pz(R[ref], R[n], g, top, "forced")
                print(f"  {n} vs {ref} at {top // 1024}k: finished +{wf}/-{lf} z{zf:+.1f}, forced +{wo}/-{lo} z{zo:+.1f}")


FMTS = (("board", "board"), ("bridge_xy", "xy"), ("bridge", "named"))


def ladder(models):
    """Per model and bucket: forced (and finished) curves on each surface of the same puzzles, then the plateau
    (largest budget) per surface and each model's change vs the first model on each surface (paired over puzzles)."""
    D = {}                                                     # (model, fmt) -> {source id: row}
    for m in models:
        rows = {}
        for f in (f"{m}_ladder", m):
            path = ROOT / f"llm_transfer/results/{f}_sweep.jsonl"
            if path.exists(): rows.update(load(f))
        for fam, fmt in FMTS:
            D[(m, fmt)] = {i.split("-", 1)[1]: r for i, r in rows.items() if r["family"] == fam}
    have = [(m, fmt) for m in models for _, fmt in FMTS if D[(m, fmt)]]
    src = sorted(set.intersection(*(set(D[k]) for k in have)))
    cuts = sorted(set.intersection(*(set(next(iter(D[k].values()))["at"]) for k in have)))
    fmts = [fmt for _, fmt in FMTS if all(D[(m, fmt)] for m in models)]
    buckets = defaultdict(list)
    for i in src:
        r = D[have[0]][i]; buckets["all"].append(i); buckets[r["dial"]["bucket"]].append(i)
    head = " ".join(f"{c // 1024:>4d}k" for c in cuts); top = cuts[-1]
    print(f"forced accuracy by budget on each surface of the same {len(src)} puzzles (finished in brackets at {top // 1024}k)")
    for b in ["all"] + sorted(k for k in buckets if k != "all"):
        g = buckets[b]; print(f"\n{b}   (n={len(g)})          B = {head}")
        for m in models:
            for fmt in fmts:
                d = D[(m, fmt)]
                frc = " ".join(f"{100 * sum(d[i]['at'][c]['forced'] for i in g) / len(g):5.0f}" for c in cuts)
                fin = 100 * sum(d[i]["at"][top]["finished"] for i in g) / len(g)
                print(f"  {m:8s} {fmt:6s} {frc}   ({fin:.0f})")
        for m in models[1:]:
            parts = []
            for fmt in fmts:
                ref, new = D[(models[0], fmt)], D[(m, fmt)]
                w = sum(new[i]["at"][top]["forced"] and not ref[i]["at"][top]["forced"] for i in g)
                l = sum(ref[i]["at"][top]["forced"] and not new[i]["at"][top]["forced"] for i in g)
                parts.append(f"{fmt} +{w}/-{l} z{(w - l) / math.sqrt(w + l) if w + l else 0:+.1f}")
            print(f"  {m} vs {models[0]} at {top // 1024}k (forced): " + ", ".join(parts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+")
    ap.add_argument("--by", choices=["all", "dial"], default="dial")
    ap.add_argument("--ladder", action="store_true", help="names are models: compare the surfaces of the same puzzles")
    a = ap.parse_args()
    if a.ladder: ladder(a.names)
    else: curves(a.names, a.by)


if __name__ == "__main__":
    main()
