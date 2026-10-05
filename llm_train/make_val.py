#!/usr/bin/env python3
"""Validation set for the curriculum gate: N puzzles per pool bucket (the 11 of llm_train.curriculum.bucket: win1-4,
the backward-trap wins win2b-4b, nowin, block, forced, prevent), carved out of the pool so the scheduler never trains
on them. The benchmark (llm_bench/data/bench.jsonl, 1,000 items) stays the test set: reported, never used to choose a
model. Pool positions were already de-duplicated against the benchmark by build_traces.

Writes llm_bench's item format (prompt in the ascii format; llm_bench.run --prompt-format rebuilds it), with the win
meta carrying difficulty (= J) and family, so summaries report backward-trap wins as their own groups.

  python -m llm_train.make_val --pool llm_train/puzzles/pool_v2.jsonl.gz --per-bucket 100 \
      --out llm_train/puzzles/val_v3.jsonl --ids-out llm_train/puzzles/val_v3_ids.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_train.curriculum import bucket, load_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True); ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--ids-out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from expert.engine import State
    from llm_bench.text import prompt
    by_b = defaultdict(list)
    for r in load_pool(a.pool): by_b[bucket(r)].append(r)
    rng = random.Random(a.seed); picked = []
    for b in sorted(by_b):
        rows = sorted(by_b[b], key=lambda r: r["id"]); rng.shuffle(rows); picked += rows[: a.per_bucket]
    with open(a.out, "w") as f:
        for r in picked:
            it = r["item"]; meta = dict(it.get("meta", {})); meta["bucket"] = bucket(r)
            if it["task"] == "win" and it["answers"]["win"]: meta.setdefault("difficulty", meta.get("J", 1))
            s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
            f.write(json.dumps({"id": it["id"], "task": it["task"], "rows": it["rows"], "cols": it["cols"],
                                "board": it["board"], "ball": it["ball"], "player": it["player"],
                                "answers": it["answers"], "meta": meta, "prompt": prompt(it["task"], s, "ascii")}) + "\n")
    a.ids_out.write_text(json.dumps(sorted(r["id"] for r in picked)))
    print(len(picked), "validation items:", dict(Counter(bucket(r) for r in picked)))


if __name__ == "__main__":
    main()
