#!/usr/bin/env python3
"""Compare transfer-suite results across models (llm_transfer/results/<name>.jsonl), item by item.

Per family and per dial: finished / forced accuracy for every model, and each model's change against the first one
(the reference, normally the base model) with a paired z = (wins - losses) / sqrt(wins + losses) over the same items;
false absent (said none when an answer exists) and false present (answered when the answer is none) per model.
The bridge is also compared with the original-format results of the same puzzles when --phutball maps a model name to
its validation run (EVAL/v<R>.jsonl): does the skill survive the change of surface?

  python -m llm_transfer.report base warm r14 grpo_p3 [--phutball r14=/path/eval/v14 grpo_p3=/path/eval/v3]
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
    return {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / f"llm_transfer/results/{name}.jsonl")}


def pz(ref, new, ids, key):
    w = sum(new[i][key] and not ref[i][key] for i in ids); l = sum(ref[i][key] and not new[i][key] for i in ids)
    return w, l, (w - l) / math.sqrt(w + l) if w + l else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="+")
    ap.add_argument("--key", choices=["correct", "forced_correct"], default="forced_correct")
    ap.add_argument("--phutball", nargs="*", default=[], help="name=EVAL/vR (original-format validation results)")
    a = ap.parse_args()
    R = {n: load(n) for n in a.names}; ref = a.names[0]
    ids = sorted(set.intersection(*(set(r) for r in R.values())))
    shas = {i: {R[n][i]["prompt_sha"] for n in a.names if "prompt_sha" in R[n][i]} for i in ids}
    bad = [i for i in ids if len(shas[i]) > 1]                 # same id, different prompt: a rebuilt suite
    if bad:
        print(f"WARNING: {len(bad)} items have different prompts across models (suite rebuilt between runs?); dropped")
        ids = [i for i in ids if i not in set(bad)]
    groups = defaultdict(list)
    for i in ids:
        r = R[ref][i]; groups[(r["family"], "all")].append(i); groups[(r["family"], dial_key(r["dial"]))].append(i)
    print(f"{'forced' if a.key == 'forced_correct' else 'finished'} accuracy (change vs {ref}: +wins/-losses, paired z)")
    print(f"{'':46s}" + "".join(f"{n:>22s}" for n in a.names))
    for (fam, key), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1])):
        line = f"{fam + ' | ' + key:46s}"
        for n in a.names:
            acc = sum(R[n][i][a.key] for i in g) / len(g)
            if n == ref: line += f"{acc:>22.1%}"
            else:
                w, l, z = pz(R[ref], R[n], g, a.key); line += f"{acc:>8.1%} +{w:<3d}-{l:<3d}z{z:+5.1f}"
        print(line + f"   (n={len(g)})")
    cat = "forced_category" if a.key == "forced_correct" else "category"
    print("\nfalse absent (among items with an answer) / false present (among items whose answer is none)")
    for fam in sorted({R[ref][i]["family"] for i in ids}):
        pres = [i for i in ids if R[ref][i]["family"] == fam and not R[ref][i]["absent"]]
        abse = [i for i in ids if R[ref][i]["family"] == fam and R[ref][i]["absent"]]
        line = f"  {fam:12s}"
        for n in a.names:
            fa = sum(R[n][i][cat] == "false_absent" for i in pres) / max(len(pres), 1)
            fp = sum(R[n][i][cat] == "false_present" for i in abse) / max(len(abse), 1) if abse else float("nan")
            line += f"  {n}: {fa:5.1%} / {fp:5.1%}"
        print(line)
    for spec in a.phutball:                                    # bridge vs the original board format, same puzzles
        n, path = spec.split("=", 1)
        if n not in R: continue
        orig = {}
        for l in open(f"{path}.jsonl"):
            r = json.loads(l); orig.setdefault(r["id"], r)
        pairs = [(R[n][i], orig[R[n][i]["id"][len("bridge-"):]]) for i in ids if i.startswith("bridge-")
                 and R[n][i]["id"][len("bridge-"):] in orig]
        if not pairs: continue
        fo = sum(o.get("forced_correct", o["correct"]) for _, o in pairs) / len(pairs)
        fb = sum(b["forced_correct"] for b, _ in pairs) / len(pairs)
        print(f"\nbridge vs board format, {n}: forced {fo:.1%} (board) -> {fb:.1%} (text), same {len(pairs)} puzzles")


if __name__ == "__main__":
    main()
