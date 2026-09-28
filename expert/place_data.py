"""Placement-puzzle pools for training (expert/puzzle_place.py), engine-verified, in the curriculum pool format plus
per-example value weights.

  forced   unstoppable threat: target the unstoppable placements; value +1, weight 1
  block    pure-placement block of a winning chain: target the blocking placements; value weight 0
  prevent  denial: stop an unstoppable threat one turn early, by placement or by a jump chain (labelled by its first
           jump); value weight 0

  python -m expert.place_data --kind forced --n 2000 --seed 1 --out expert_data/place_pools/forced_00.npz
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expert.puzzle_data import CELLS, Generator, encode  # noqa: E402
from expert.puzzle_place import CAP_DROPS, make_block, make_forced, make_prevent  # noqa: E402


def sample(kind: str, n: int, seed: int, max_jumps: int = 3):
    gen = Generator(seed=seed); rng = random.Random(seed); cells = [c for c in CELLS if c[1] <= max_jumps]
    ex, V, W, meta = [], [], [], Counter(); tries = 0
    while len(ex) < n:
        base = gen.puzzle(cells[tries % len(cells)]); tries += 1
        if base is None: continue
        N = base.rows * base.cols
        if kind == "forced":
            r = make_forced(base, rng)
            if not r: continue
            s, tw, m = r; ex.append((s, dict(tw))); V.append(1.0); W.append(1.0); meta[f"{len(tw)} answers"] += 1
        elif kind == "block":
            r = make_block(base)
            if not r: continue
            s, tw, m = r; ex.append((s, dict(tw))); V.append(0.0); W.append(0.0); meta[f"{len(tw)} answers"] += 1
        else:
            r = make_prevent(base, rng)
            if not r: continue
            s, pl, jl, m = r; ex.append((s, {**pl, **{N + l: w for l, w in jl.items()}})); V.append(0.0); W.append(0.0)
            meta["saved by a jump too" if jl else "placements only"] += 1
    return ex, np.array(V, np.float32), np.array(W, np.float32), tries, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("forced", "block", "prevent"), required=True)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-jumps", type=int, default=3); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); t0 = time.time()
    ex, V, W, tries, meta = sample(a.kind, a.n, a.seed, a.max_jumps)
    S, P, _ = encode(ex, actions=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, states=S.astype(np.int8), policy_targets=P, value_targets=V, value_weights=W)
    print(f"{a.kind}: {len(ex)} puzzles from {tries} bases in {time.time() - t0:.0f}s; {dict(meta)}; dropped for a search cap: {CAP_DROPS[0]}")
