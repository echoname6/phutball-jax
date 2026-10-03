#!/usr/bin/env python3
"""Build the zero-shot phutball benchmark for language models (the same puzzle families the network trained on).

  win     "Can the side to move win THIS turn?" Positives: the frozen held-out winning-jump set
          (expert_data/puzzle_eval.npz, both families, stratified by true shortest-win length 1-4). Negatives: near-miss
          threat positions (expert/puzzle_place.make_threat: one stone of a winning chain removed, the engine proves no
          win remains). Answer: a winning jump sequence (checked by replay) or NO WIN.
  forced  Unstoppable threat (expert/puzzle_place.make_forced): every placement after which the side to move wins next
          turn whatever the opponent does. The answer set is complete (all placements are checked).

Fresh puzzles use seed --seed (training pools used 0-3). Output: llm_bench/data/bench.jsonl, one item per line.

  ~/Projects/phutball/venv/bin/python -m llm_bench.build --per-difficulty 25 --negatives 100 --forced 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import State  # noqa: E402
from expert.puzzle_data import CELLS, Generator  # noqa: E402
from expert.puzzle_place import CAP_DROPS, CapHit, make_forced, make_threat  # noqa: E402
from llm_bench.text import prompt, sq  # noqa: E402


def item(task, s: State, answers, meta, idx):
    return {"id": f"{task}-{idx:04d}", "task": task, "rows": s.rows, "cols": s.cols, "board": list(map(int, s.board)),
            "ball": int(s.ball), "player": int(s.player), "answers": answers, "meta": meta, "prompt": prompt(task, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-difficulty", type=int, default=25, help="win positives per shortest-win length 1..4")
    ap.add_argument("--negatives", type=int, default=100); ap.add_argument("--forced", type=int, default=100)
    ap.add_argument("--seed", type=int, default=9001); ap.add_argument("--out", type=Path, default=ROOT / "llm_bench/data/bench.jsonl")
    a = ap.parse_args(); t0 = time.time(); rng = random.Random(a.seed); items = []

    recs = list(np.load(ROOT / "expert_data/puzzle_eval.npz", allow_pickle=True)["recs"])
    rng.shuffle(recs)
    for d in (1, 2, 3, 4):
        pick = [r for r in recs if int(r["difficulty"]) == d][:a.per_difficulty]
        for r in pick:
            s = State(21, 15, [int(v) for v in r["board"]], int(r["ball"]), int(r["player"]))
            items.append(item("win", s, {"win": True, "winning_first": [sq(int(x), 15) for x in r["winning_first"]]},
                              {"difficulty": d, "family": str(r["family"]), "source": "puzzle_eval"}, len(items)))
    n_pos = len(items)

    gen = Generator(seed=a.seed); cells = [c for c in CELLS if c[1] <= 3]; tries = 0; neg = forced = 0
    while neg < a.negatives or forced < a.forced:
        base = gen.puzzle(cells[tries % len(cells)]); tries += 1
        if base is None: continue
        if neg < a.negatives:
            try:
                r = make_threat(base, rng)
            except CapHit:
                r = None
            if r:
                s, _tw, meta = r
                items.append(item("win", s, {"win": False}, {"source": "near-miss threat", **{k: int(v) for k, v in meta.items()}}, len(items)))
                neg += 1
                continue
        if forced < a.forced:
            r = make_forced(base, rng)
            if r:
                s, tw, meta = r
                items.append(item("forced", s, {"placements": sorted(sq(p, s.cols) for p in tw)},
                                  {"source": "make_forced", **{k: int(v) for k, v in meta.items()}}, len(items)))
                forced += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    c = Counter((it["task"], it["answers"].get("win", "forced")) for it in items)
    print(f"{len(items)} items ({n_pos} win positives, {neg} near-miss negatives, {forced} forced) from {tries} bases in "
          f"{time.time() - t0:.0f}s; {dict(c)}; dropped for a search cap: {CAP_DROPS[0]} -> {a.out}")
    print("median forced answer-set size:", float(np.median([len(it["answers"]["placements"]) for it in items if it["task"] == "forced"] or [0])))


if __name__ == "__main__":
    main()
