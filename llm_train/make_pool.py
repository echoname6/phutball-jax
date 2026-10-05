#!/usr/bin/env python3
"""Package a build_traces output into a curriculum pool: one row per puzzle {id, task, subtask, bucket, item,
engine_think_tokens}, gzipped. Positions in the benchmark or puzzle_eval were already excluded by build_traces;
duplicates (same board, ball, player) are dropped here.

  python -m llm_train.make_pool OUT.jsonl.gz SRC1[=prefix] SRC2[=prefix] ...
ids are "<prefix><id>" (a prefix keeps ids from separate build_traces runs from colliding; the scheduler keys on ids).
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter

from llm_train.curriculum import bucket


def main():
    dst, srcs = sys.argv[1], sys.argv[2:]
    seen, rows = set(), []
    for spec in srcs:
        src, _, prefix = spec.partition("=")
        for l in open(src):
            r = json.loads(l); it = r["item"]
            key = (tuple(it["board"]), it["ball"], it["player"])
            if key in seen: continue
            seen.add(key)
            row = {"id": prefix + r["id"], "task": r["task"], "subtask": r["subtask"], "item": it,
                   "engine_think_tokens": r.get("think_tokens")}
            row["item"]["id"] = row["id"]; row["bucket"] = bucket(row); rows.append(row)
    with gzip.open(dst, "wt") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} puzzles -> {dst}"); print(dict(sorted(Counter(r["bucket"] for r in rows).items())))


if __name__ == "__main__":
    main()
