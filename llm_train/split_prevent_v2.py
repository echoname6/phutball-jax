#!/usr/bin/env python3
"""Finish the regenerated prevent puzzles (llm_train/make_prevent_v2.py): measure what simple rules still solve, then
split them into test / validation / training.

Shortcut report (the goal: no simple rule above ~33%):
  best fixed square     the single square at a fixed offset from the ball (in the defender's frame, up to 3 away) that
                        saves the most puzzles; a model that always plays there gets this score
  any single jump       the first legal jump of the ball (stop after one jump); jump toward own goal likewise
  random placement      expected score of a uniformly random legal placement

Split (seeded): --test to llm_bench/data/prevent_v2_test.jsonl (llm_bench format; report with
llm_bench.run --bench), --val appended to the curriculum validation set (+ its id list), the rest added to the pool
(pool_v3 + them -> pool_v4).

  python -m llm_train.split_prevent_v2 --src llm_train/puzzles/prevent_v2.jsonl.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(src: Path) -> list[dict]:
    """The merged file, or (run still going) its per-worker part files."""
    from llm_train.build_traces import key_of
    if src.exists(): rows = [json.loads(l) for l in gzip.open(src, "rt")]
    else: rows = [json.loads(l) for p in sorted(glob.glob(f"{src}.part*.jsonl")) for l in open(p)]
    seen, out = set(), []
    for r in rows:
        k = key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"])
        if k not in seen: seen.add(k); out.append(r)
    return out


def shortcuts(items: list[dict]) -> dict:
    from expert.engine import BALL, MAN, State
    from llm_bench.run import jump_landings, score
    from llm_bench.text import sq
    n = len(items); rel = Counter(); res = Counter(); rand = 0.0
    for it in items:
        s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"]); C = s.cols
        br, bc = divmod(s.ball, C); fw = -1 if s.player == 1 else 1
        saves = {int(x[1:]) * C + ord(x[0]) - 97 for x in it["answers"]["placements"]}
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = br + dr * fw, bc + dc * fw                            # defender's frame (mirrored for player 2)
                if 1 <= r <= s.rows - 2 and 0 <= c < C and r * C + c in saves: rel[(dr, dc)] += 1
        js = jump_landings(s.board, s.ball, s.rows, s.cols)
        if js:
            res["any single jump"] += score(it, f"ANSWER: JUMP {sq(js[0][0], C)}")["correct"]
            toward = [l for l, _ in js if (l // C - br) * fw > 0]
            res["single jump toward own goal"] += bool(toward) and score(it, f"ANSWER: JUMP {sq(toward[0], C)}")["correct"]
        empty = [i for i in range(C, (s.rows - 1) * C) if s.board[i] not in (BALL, MAN)]
        rand += len(saves & set(empty)) / len(empty)
    (off, k), = rel.most_common(1) or [((None, None), 0)]
    return {"puzzles": n, "best fixed square (rows toward own goal, cols)": [list(off), round(k / n, 3)],
            "any single jump": round(res["any single jump"] / n, 3),
            "single jump toward own goal": round(res["single jump toward own goal"] / n, 3),
            "random placement": round(rand / n, 3),
            "top fixed squares": [[list(o), round(v / n, 3)] for o, v in rel.most_common(5)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "llm_train/puzzles/prevent_v2.jsonl.gz")
    ap.add_argument("--test", type=int, default=200); ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--report-only", action="store_true"); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rows = load(a.src)
    print(json.dumps(shortcuts([r["item"] for r in rows]), indent=1))
    if a.report_only: return
    from expert.engine import State
    from llm_bench.text import prompt
    rng = random.Random(a.seed); rows = sorted(rows, key=lambda r: r["id"]); rng.shuffle(rows)
    test, val, train = rows[: a.test], rows[a.test: a.test + a.val], rows[a.test + a.val:]
    bench_row = lambda it: {**{k: it[k] for k in ("id", "task", "rows", "cols", "board", "ball", "player", "answers")},
                            "meta": {**it["meta"], "bucket": "prevent"},
                            "prompt": prompt("prevent", State(it["rows"], it["cols"], it["board"], it["ball"], it["player"]), "ascii")}
    with open(ROOT / "llm_bench/data/prevent_v2_test.jsonl", "w") as f:
        for r in test: f.write(json.dumps(bench_row(r["item"])) + "\n")
    vpath, vids = ROOT / "llm_train/puzzles/val_v3.jsonl", ROOT / "llm_train/puzzles/val_v3_ids.json"
    have = {json.loads(l)["id"] for l in open(vpath)}
    with open(vpath, "a") as f:
        for r in val:
            if r["id"] not in have: f.write(json.dumps(bench_row(r["item"])) + "\n")
    vids.write_text(json.dumps(sorted(set(json.loads(vids.read_text())) | {r["id"] for r in val})))
    with gzip.open(ROOT / "llm_train/puzzles/pool_v3.jsonl.gz", "rt") as f, \
            gzip.open(ROOT / "llm_train/puzzles/pool_v4.jsonl.gz", "wt") as g:
        for l in f: g.write(l)
        for r in train: g.write(json.dumps(r) + "\n")
    print(f"test {len(test)} -> llm_bench/data/prevent_v2_test.jsonl | val {len(val)} -> val_v3 | "
          f"train {len(train)} -> pool_v4 (= pool_v3 + them)")


if __name__ == "__main__":
    main()
