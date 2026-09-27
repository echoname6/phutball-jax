"""Expert games -> training examples in the exact format TransformerTrainer's replay buffer uses.

  states:          state_to_network_input(PhutballState)  (rotated 180 deg for P2, as in self-play)
  policy_targets:  (2*R*C + 1,) distribution in VISUAL coords: for P2 the placement block and the
                   jump block are each reversed (as trajectory_to_training_examples does)
  value_targets:   final result from the side-to-move's point of view (+1 win, -1 loss, draw_value)

Generation is pure Python (expert/engine.py + expert/oracle.py) and parallelises over processes;
encoding uses the repo's own JAX encoder so there is no representation drift.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from expert.engine import legal_actions, new_game, step  # noqa: E402
from expert.oracle import Expert  # noqa: E402


def play_record(seed: int, rows: int, cols: int, eps: float, max_turns: int, opp: str = "expert"):
    """One game. Returns [(State, expert_action)] for every decision of an expert-controlled side,
    and the winner. With prob eps a random legal micro-action is PLAYED (the label stays the expert's)."""
    rng = random.Random(seed); ex = Expert(); s = new_game(rows, cols); rec = []
    expert_side = {1, 2} if opp == "expert" else {1 + seed % 2}
    while not s.winner and not s.draw and s.turns < max_turns:
        if s.player in expert_side:
            a = ex.action(s); rec.append((s, a))
            played = a if rng.random() > eps else rng.choice(legal_actions(s))
        else:
            acts = legal_actions(s); n = rows * cols
            jumps = [x for x in acts if x >= n]
            played = rng.choice(jumps) if (s.jumping and jumps) else rng.choice(acts)
        s = step(s, played)
    return rec, s.winner


def _gen(args):
    seed, rows, cols, eps, max_turns, opp = args
    return play_record(seed, rows, cols, eps, max_turns, opp)


def generate(n_games: int, rows=21, cols=15, eps=0.1, max_turns=400, workers=None, seed=0, opp_mix=(0.8, 0.2)):
    """opp_mix: share of games expert-vs-expert, expert-vs-random."""
    rng = random.Random(seed)
    jobs = [(seed * 1_000_003 + i, rows, cols, eps, max_turns, "expert" if rng.random() < opp_mix[0] else "random")
            for i in range(n_games)]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_gen, jobs, chunksize=4)


def to_examples(games, rows=21, cols=15, draw_value=0.0, smoothing=0.0):
    """Encode with the repo's JAX encoder. Returns (states, policies, values) numpy arrays."""
    import jax
    import jax.numpy as jnp
    import phutball_env_jax as J
    cfg = J.EnvConfig(rows=rows, cols=cols)
    n = rows * cols; A = 2 * n + 1
    enc = jax.jit(jax.vmap(lambda st: J.state_to_network_input(st, cfg)))
    S, P, V = [], [], []
    boards, balls, players, jumping, turns, seqs, seqlen = [], [], [], [], [], [], []
    for rec, winner in games:
        for s, a in rec:
            boards.append(np.array(s.board, np.int32).reshape(rows, cols)); balls.append(divmod(s.ball, cols))
            players.append(s.player); jumping.append(s.jumping); turns.append(s.turns)
            seq = np.full((J.MAX_JUMP_SEQUENCE_LENGTH, 2), -1, np.int32)
            for k, pos in enumerate(s.seq[: J.MAX_JUMP_SEQUENCE_LENGTH]): seq[k] = divmod(pos, cols)
            seqs.append(seq); seqlen.append(len(s.seq) if s.jumping else 0)
            pol = np.full(A, smoothing / A, np.float32); pol[a] += 1.0 - smoothing
            if s.player == 2:                                   # physical -> visual (180 deg), as in self-play
                pol[:n] = pol[:n][::-1].copy(); pol[n:2 * n] = pol[n:2 * n][::-1].copy()
            P.append(pol)
            V.append(draw_value if winner == 0 else (1.0 if winner == s.player else -1.0))
    for i in range(0, len(boards), 4096):
        sl = slice(i, i + 4096)
        st = J.PhutballState(board=jnp.array(np.stack(boards[sl])), ball_pos=jnp.array(np.array(balls[sl], np.int32)),
                             current_player=jnp.array(players[sl], jnp.int32), is_jumping=jnp.array(jumping[sl]),
                             terminated=jnp.zeros(len(boards[sl]), bool), winner=jnp.zeros(len(boards[sl]), jnp.int32),
                             num_turns=jnp.array(turns[sl], jnp.int32), jump_sequence=jnp.array(np.stack(seqs[sl])),
                             jump_sequence_length=jnp.array(seqlen[sl], jnp.int32))
        S.append(np.array(enc(st), np.float32))
    return np.concatenate(S), np.stack(P), np.array(V, np.float32)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--games", type=int, default=200); ap.add_argument("--out", default="expert_data.npz")
    ap.add_argument("--rows", type=int, default=21); ap.add_argument("--cols", type=int, default=15)
    ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--workers", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    games = generate(a.games, a.rows, a.cols, a.eps, workers=a.workers, seed=a.seed)
    t1 = time.time(); S, P, V = to_examples(games, a.rows, a.cols)
    np.savez_compressed(a.out, states=S, policy_targets=P, value_targets=V)
    print(f"{a.games} games -> {len(V)} examples in {t1-t0:.0f}s generate + {time.time()-t1:.0f}s encode; "
          f"value mean {V.mean():+.2f}; saved {a.out}")
