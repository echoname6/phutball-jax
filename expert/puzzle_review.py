"""Review the winning-jump puzzle families (curriculum_puzzles.py) with the independent fast engine.

For each setting (jumps, stones per jump, noise stones) generate puzzles and check:
  valid      the canonical solution is legal and wins for the side to move
  ambiguous  more than one winning FIRST jump exists (a one-hot label on the canonical one would be wrong)
  shorter    a win with FEWER jumps than intended exists (the puzzle is easier than its label)
Also prints an ASCII board per family for eyeballing.

  python -m expert.puzzle_review --n 60
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from expert.engine import State, step  # noqa: E402
from expert.oracle import chains  # noqa: E402


def to_engine(js) -> State:
    b = np.asarray(js.board); rows, cols = b.shape; br, bc = (int(x) for x in np.asarray(js.ball_pos))
    return State(rows, cols, [int(v) for v in b.reshape(-1)], br * cols + bc, int(js.current_player))


def winning_chains(s: State):
    """All complete jump chains that win for the side to move, as (first landing, n jumps)."""
    out = []
    for path, nb, nbl, w in chains(s.board, s.ball, s.rows, s.cols, 20000):
        if w == s.player: out.append((path[0], len(path)))
    return out


def render(s: State, mark=()) -> str:
    sym = {0: ".", -1: "O", 1: "#", 2: "-", -2: "-"}; lines = []
    for r in range(s.rows):
        row = "".join("*" if (r * s.cols + c) in mark else sym.get(s.board[r * s.cols + c], "?") for c in range(s.cols))
        lines.append(f"{r:2d} {row}")
    return "\n".join(lines)


def play_actions(s: State, actions) -> State:
    n = s.rows * s.cols; s = s.copy()
    for a in actions:
        s = step(s, int(a))
        if s.winner: break
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--rows", type=int, default=21)
    ap.add_argument("--cols", type=int, default=15); ap.add_argument("--show", action="store_true")
    x = ap.parse_args()
    import jax
    import curriculum_puzzles as C
    from phutball_env_jax import EnvConfig
    cfg = EnvConfig(rows=x.rows, cols=x.cols); key = jax.random.PRNGKey(0)
    settings = [("one-move", dict(min_jump_len=L, max_jump_len=L, add_noise_men=nz > 0, max_noise_men=max(nz, 1)), 1, L, nz)
                for L in (1, 3, 6) for nz in (0, 10)]
    settings += [("n-move", dict(num_jumps=J, min_jump_len=L, max_jump_len=L, add_noise_men=nz > 0, max_noise_men=max(nz, 1)), J, L, nz)
                 for J in (2, 3, 4) for L in (1, 3) for nz in (0, 8)]
    print(f"{'family':9s} {'jumps':>5s} {'stones/jump':>11s} {'noise<=':>7s} | {'valid':>6s} {'ambiguous':>9s} {'shorter':>7s} "
          f"{'win-1st-moves':>13s} | gen ms")
    shown = set()
    for fam, kw, J, L, nz in settings:
        valid = amb = shorter = 0; nfirst = Counter(); t0 = time.time()
        for i in range(x.n):
            key, k = jax.random.split(key)
            if fam == "one-move":
                js, a = C.generate_one_move_win_state(k, cfg, player=1 + (i % 2), **kw); acts = [a]
            else:
                js, acts = C.generate_n_move_win_state(k, cfg, player=1 + (i % 2), **kw)
            s = to_engine(js)
            end = play_actions(s, acts); ok = end.winner == s.player; valid += ok
            wins = winning_chains(s); firsts = {f for f, _ in wins}; nfirst[len(firsts)] += 1
            amb += len(firsts) > 1; shorter += any(nj < J for _, nj in wins)
            if x.show and fam + str(J) not in shown and nz:
                shown.add(fam + str(J))
                n = s.rows * s.cols; land = [int(a) - n for a in acts if n <= int(a) < 2 * n]
                print(f"\n{fam}, {J} jump(s), {L} stones/jump, noise: player {s.player} to move (O ball, # stones, * canonical landings)")
                print(render(s, set(land)))
        ms = (time.time() - t0) / x.n * 1000
        dist = " ".join(f"{k}:{v}" for k, v in sorted(nfirst.items()))
        print(f"{fam:9s} {J:5d} {L:11d} {nz:7d} | {valid/x.n:6.0%} {amb/x.n:9.0%} {shorter/x.n:7.0%} {dist:>13s} | {ms:6.0f}", flush=True)


if __name__ == "__main__":
    main()
