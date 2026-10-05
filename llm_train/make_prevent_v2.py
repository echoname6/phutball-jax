#!/usr/bin/env python3
"""Prevent puzzles without the escape-hatch shortcut.

Why: in the v1 prevent puzzles (pool_v2 and the benchmark) "place a man next to the ball on your own side" saves 98-99%
of the time: the man gives the defender an escape jump next turn, whatever the attacker builds. It is sound phutball,
but on the generator's sparse boards it is always available, so the task tested one rule (a no-model baseline that
places there scores 99% on the pool, 98% on the benchmark). In 946 freshly generated sparse-board puzzles, no square on
the defender's side of the ball ever failed to save.

Here: bases are positions from the scripted expert's games (expert_data/games, median ~22 men), either side to move,
through the same engine-verified expert.puzzle_place.make_prevent, kept only if NONE of the three squares on the
defender's side of the ball (toward the defender's own goal, the escape-hatch squares) saves. Yield ~3% of prevent
puzzles, ~1.4 per worker-minute. Positions in the benchmark or puzzle_eval are skipped.

Measured on 33 kept puzzles: the best single fixed square (beside the ball: a sideways escape) still saves 45%, any
single legal jump 24%, a random placement 2.5% (v1: 98-99% for the one rule). --strict also drops puzzles where ANY of
the 8 squares next to the ball saves (about a third of the yield: ~0.45 per worker-minute).

Writes pool rows (id prefix prevent2-) to --out (.jsonl.gz).

  python -m llm_train.make_prevent_v2 --n 600 --workers 3 --minutes 480 --strict --out llm_train/puzzles/prevent_v2.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def escape_squares(s) -> list[int]:
    """The three squares next to the ball on the defender's side (s.player = defender, to move)."""
    br, bc = divmod(s.ball, s.cols); fw = -1 if s.player == 1 else 1
    return [(br + fw) * s.cols + bc + dc for dc in (-1, 0, 1) if 0 <= bc + dc < s.cols and 1 <= br + fw <= s.rows - 2]


def neighbours(s) -> list[int]:
    br, bc = divmod(s.ball, s.cols)
    return [(br + dr) * s.cols + bc + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= bc + dc < s.cols and 1 <= br + dr <= s.rows - 2]


def worker(args):
    wid, seed, quota, deadline, strict, part = args
    import numpy as np
    from expert.engine import BALL, EMPTY, MAN, State
    from expert.puzzle_place import CapHit, make_prevent
    from llm_bench.text import sq
    from llm_train.build_traces import bench_keys, key_of
    rng = random.Random(seed); avoid = bench_keys(); stats = Counter(); out = []
    chunks = sorted((ROOT / "expert_data/games").glob("chunk_*.npz"))
    states = np.load(chunks[wid % len(chunks)])["states"]
    while len(out) < quota and time.time() < deadline:
        o = states[rng.randrange(len(states))]
        R, C = o.shape[1], o.shape[2]
        b = [EMPTY] * (R * C)
        for i in np.flatnonzero(o[1].reshape(-1) > 0.5): b[int(i)] = MAN
        ball = int(np.flatnonzero(o[0].reshape(-1) > 0.5)[0]); b[ball] = BALL
        base = State(R, C, b, ball, rng.choice((1, 2)))           # any side to move is a legal position
        stats["bases"] += 1
        try: r = make_prevent(base, rng)
        except CapHit: stats["search cap"] += 1; continue
        if not r: continue
        s, pl, jl, meta = r; stats["prevent puzzles"] += 1
        if set(pl) & set(escape_squares(s)): stats["dropped: an escape-hatch square saves"] += 1; continue
        if strict and set(pl) & set(neighbours(s)): stats["dropped (strict): a square next to the ball saves"] += 1; continue
        if key_of(s.board, s.ball, s.player) in avoid: stats["dropped: in the benchmark"] += 1; continue
        pid = f"prevent2-{wid}-{len(out)}"
        item = {"id": pid, "task": "prevent", "rows": R, "cols": C, "board": list(map(int, s.board)), "ball": int(s.ball),
                "player": int(s.player), "answers": {"placements": sorted(sq(p, C) for p in pl),
                                                     "saving_first_jumps": sorted(sq(l, C) for l in jl)},
                "meta": {**{k: int(v) for k, v in meta.items()}, "source": "expert games, no escape hatch" + (" (strict)" if strict else "")}}
        row = {"id": pid, "task": "prevent", "subtask": "prevent", "item": item, "bucket": "prevent"}
        out.append(row); stats["kept"] += 1
        with open(part, "a") as f: f.write(json.dumps(row) + "\n")         # saved as found: a crash loses nothing
        if stats["kept"] % 10 == 0: print(f"worker {wid}: {stats['kept']} kept, {dict(stats)}", flush=True)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=180); ap.add_argument("--seed", type=int, default=31000)
    ap.add_argument("--strict", action="store_true", help="also drop puzzles where any square next to the ball saves")
    ap.add_argument("--out", type=Path, default=ROOT / "llm_train/puzzles/prevent_v2.jsonl.gz")
    a = ap.parse_args(); t0 = time.time()
    quota = -(-a.n // a.workers); deadline = t0 + a.minutes * 60
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(worker, [(w, a.seed + w, quota, deadline, a.strict, f"{a.out}.part{w}.jsonl")
                                for w in range(a.workers)])
    from llm_train.build_traces import key_of
    seen, rows = set(), []
    for out, _ in res:
        for r in out:
            k = key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"])
            if k not in seen: seen.add(k); rows.append(r)
    with gzip.open(a.out, "wt") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} puzzles in {(time.time() - t0) / 60:.0f} min -> {a.out}")
    print(dict(sum((st for _, st in res), Counter())))


if __name__ == "__main__":
    main()
