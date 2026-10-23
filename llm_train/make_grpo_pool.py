#!/usr/bin/env python3
"""The full GRPO training pool: pool_v4 minus every held-out puzzle (validation val_v3 by id and position, the test
benchmark bench_v2 and the prevent test set by position). pool_v4 contains 1,000 validation puzzles, so it must never be
used for training as is.

Why: gated GRPO run 2 trained on the 1,059 puzzles v4 r14 found learnable (67 block, 41 forced, 41 prevent). Phase 2
drew 31.5% forced placements from those 41 puzzles; training rates rose (block 14/16, forced 12/16) while held-out
validation fell (forced placements forced-at-16k 60 -> 53, wins 93.4 -> 87.6, 4-jump backward traps 81 -> 61) and new
illegal-jump errors appeared: overfitting to a handful of positions. The clean pool has ~1,800+ block and forced puzzles.

  python -m llm_train.make_grpo_pool --out /content/drive/MyDrive/phutball/grpo_pool_full.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def key(it):
    return (tuple(it["board"]), it["ball"], it["player"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=ROOT / "llm_train/puzzles/pool_v4.jsonl.gz")
    a = ap.parse_args()
    held, held_ids = set(), set(json.load(open(ROOT / "llm_train/puzzles/val_v3_ids.json")))
    for f in ("llm_train/puzzles/val_v3.jsonl", "llm_bench/data/bench_v2.jsonl", "llm_bench/data/bench.jsonl",
              "llm_bench/data/prevent_v2_test.jsonl"):
        p = ROOT / f
        if p.exists():
            for l in open(p):
                r = json.loads(l); held.add(key(r.get("item", r)))
    rows, dropped = [], 0
    for l in gzip.open(a.pool, "rt"):
        r = json.loads(l)
        if r["id"] in held_ids or key(r["item"]) in held: dropped += 1; continue
        rows.append(r)
    assert not any(key(r["item"]) in held or r["id"] in held_ids for r in rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} puzzles -> {a.out} ({dropped} held-out puzzles excluded)")
    print(dict(sorted(Counter(r["bucket"] for r in rows).items())))


if __name__ == "__main__":
    main()
