"""Curriculum data from the winning-jump puzzle families, relabelled by the fast engine.

Every puzzle (curriculum_puzzles.py one-/n-move wins, optionally mirrored and cluttered) is relabelled:
  * difficulty = the SHORTEST winning chain the engine finds (the generator's intended jump count can be wrong:
    stones placed for later jumps often allow shortcuts);
  * EVERY state along a shortest winning chain becomes an example (the start, then after each jump), so the network
    learns to continue a chain, not just to start it;
  * the policy target at each state covers EVERY winning next jump: each move's best completion is scored
        score(move) = -(jumps still needed to win) + material_w * (stones removed on that completion)
    and the target is softmax(score / temp) over winning moves (0 on everything else). material_w defaults to 0:
    a winning turn ends the game, so removed stones change nothing; a tiny value only breaks ties between equally
    short wins. Value target: +1 (the side to move wins this turn).

Curriculum cells: (J intended jumps, L stones per jump, noise stones, clutter) with clutter in {none, real}; "real"
adds stones taken from positions of expert games (expert/dataset.py) around the puzzle, then relabels (clutter can
create or block wins, the relabel keeps the labels exact; puzzles with no remaining win are dropped).

  python -m expert.puzzle_data --build-clutter 120          # pool of realistic stone patterns from expert games
  python -m expert.puzzle_data --eval-set --per-cell 40     # frozen held-out set (fixed seeds)
  python -m expert.puzzle_data --sample 2000 --out d.npz    # a training batch from the curriculum
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import BALL, EMPTY, MAN, State, jump_landings, step, winner_at  # noqa: E402

DATA = ROOT / "expert_data"
CELLS = [(J, L, nz, cl) for J in (1, 2, 3, 4) for L in (1, 2, 3) for nz in (0, 8) for cl in ("none", "real")]


# --------------------------------------------------------------------------------------------------------------
# relabelling
def best_completions(s: State, cap_nodes: int = 200_000) -> dict:
    """For the side to move (jumping or not): {next jump landing: (fewest jumps to win incl. this one,
    most stones removed among those shortest wins)} over every winning continuation this turn.
    Exhaustive search over distinct (ball, removed stones) states, depth-first with memoisation."""
    me, rows, cols = s.player, s.rows, s.cols
    memo = {}; nodes = [0]

    def solve(board, ball, removed):       # -> (fewest jumps, stones removed) to win from here, or None
        key = (ball, removed)
        if key in memo: return memo[key]
        memo[key] = None                     # guards cycles through the same state
        nodes[0] += 1
        if nodes[0] > cap_nodes: return None
        best = None
        for land, jumped in jump_landings(board, ball, rows, cols):
            w = winner_at(rows, land // cols)
            if w and w != me: continue       # into my own goal: a loss
            if w == me:
                cand = (1, len(jumped))
            else:
                nb = board[:]; nb[ball] = EMPTY
                for j in jumped: nb[j] = EMPTY
                nb[land] = BALL
                sub = solve(nb, land, removed | frozenset(jumped))
                if sub is None: continue
                cand = (sub[0] + 1, sub[1] + len(jumped))
            if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]): best = cand
        memo[key] = best
        return best

    out = {}
    for land, jumped in jump_landings(s.board, s.ball, rows, cols):
        w = winner_at(rows, land // cols)
        if w and w != me: continue
        if w == me:
            out[land] = (1, len(jumped)); continue
        nb = s.board[:]; nb[s.ball] = EMPTY
        for j in jumped: nb[j] = EMPTY
        nb[land] = BALL
        sub = solve(nb, land, frozenset(jumped))
        if sub is not None: out[land] = (sub[0] + 1, sub[1] + len(jumped))
    return out


def target_weights(comp: dict, temp: float, material_w: float) -> dict:
    if not comp: return {}
    sc = {m: -d + material_w * k for m, (d, k) in comp.items()}; mx = max(sc.values())
    w = {m: math.exp((v - mx) / max(temp, 1e-6)) for m, v in sc.items()}; z = sum(w.values())
    return {m: v / z for m, v in w.items()}


def chain_examples(s: State, temp: float, material_w: float):
    """(state, {jump landing: weight}) for the start and every state along one shortest winning chain."""
    out = []; cur = s
    for _ in range(64):
        comp = best_completions(cur)
        if not comp: break
        out.append((cur.copy(), target_weights(comp, temp, material_w)))
        nxt = min(comp, key=lambda m: (comp[m][0], -comp[m][1], m))           # follow a shortest (deterministic)
        cur = step(cur, cur.rows * cur.cols + nxt)
        if cur.winner: break
    return out


# --------------------------------------------------------------------------------------------------------------
# puzzle generation
def to_engine(js) -> State:
    b = np.asarray(js.board); rows, cols = b.shape; br, bc = (int(x) for x in np.asarray(js.ball_pos))
    return State(rows, cols, [int(v) for v in b.reshape(-1)], br * cols + bc, int(js.current_player))


def mirror(s: State) -> State:
    rows, cols = s.rows, s.cols; b = [0] * (rows * cols)
    for r in range(rows):
        for c in range(cols): b[r * cols + c] = s.board[r * cols + (cols - 1 - c)]
    br, bc = divmod(s.ball, cols)
    return State(rows, cols, b, br * cols + (cols - 1 - bc), s.player)


def add_clutter(s: State, pattern: np.ndarray, rng: random.Random, max_add: int) -> State:
    """Add up to max_add stones from a real game's stone pattern (placement rows only, empty squares only)."""
    rows, cols = s.rows, s.cols; s = s.copy()
    cand = [i for i in np.flatnonzero(pattern) if cols <= i < rows * cols - cols and s.board[i] == EMPTY]
    rng.shuffle(cand)
    for i in cand[:max_add]: s.board[int(i)] = MAN
    return s


class Generator:
    def __init__(self, rows=21, cols=15, seed=0, clutter_pool: Path | None = DATA / "clutter_pool.npz"):
        import jax
        import curriculum_puzzles as C
        from phutball_env_jax import EnvConfig
        self.jax, self.C, self.cfg = jax, C, EnvConfig(rows=rows, cols=cols)
        self.key = jax.random.PRNGKey(seed); self.rng = random.Random(seed)
        self.pool = np.load(clutter_pool)["patterns"] if clutter_pool and Path(clutter_pool).exists() else None

    def puzzle(self, cell) -> State | None:
        J, L, nz, cl = cell; self.key, k = self.jax.random.split(self.key); player = self.rng.choice((1, 2))
        kw = dict(min_jump_len=L, max_jump_len=L, add_noise_men=nz > 0, max_noise_men=max(nz, 1))
        js = (self.C.generate_one_move_win_state(k, self.cfg, player=player, **kw)[0] if J == 1 else
              self.C.generate_n_move_win_state(k, self.cfg, num_jumps=J, player=player, **kw)[0])
        s = to_engine(js)
        if self.rng.random() < 0.5: s = mirror(s)
        if cl == "real" and self.pool is not None:
            s = add_clutter(s, self.pool[self.rng.randrange(len(self.pool))], self.rng, max_add=self.rng.randint(4, 24))
        return s


# --------------------------------------------------------------------------------------------------------------
# encoding (same format as expert/dataset.py: visual coords for P2)
def encode(examples, rows=21, cols=15):
    import jax
    import jax.numpy as jnp
    import phutball_env_jax as J
    cfg = J.EnvConfig(rows=rows, cols=cols); n = rows * cols; A = 2 * n + 1
    enc = jax.jit(jax.vmap(lambda st: J.state_to_network_input(st, cfg)))
    P, boards, balls, players, jumping, turns, seqs, seqlen = [], [], [], [], [], [], [], []
    for s, tw in examples:
        pol = np.zeros(A, np.float32)
        for m, w in tw.items(): pol[n + m] = w
        if s.player == 2: pol[:n] = pol[:n][::-1].copy(); pol[n:2 * n] = pol[n:2 * n][::-1].copy()
        P.append(pol); boards.append(np.array(s.board, np.int32).reshape(rows, cols)); balls.append(divmod(s.ball, cols))
        players.append(s.player); jumping.append(s.jumping); turns.append(s.turns)
        seq = np.full((J.MAX_JUMP_SEQUENCE_LENGTH, 2), -1, np.int32)
        for k, pos in enumerate(s.seq[: J.MAX_JUMP_SEQUENCE_LENGTH]): seq[k] = divmod(pos, cols)
        seqs.append(seq); seqlen.append(len(s.seq) if s.jumping else 0)
    S = []
    for i in range(0, len(boards), 4096):
        sl = slice(i, i + 4096); m_ = len(boards[sl])
        st = J.PhutballState(board=jnp.array(np.stack(boards[sl])), ball_pos=jnp.array(np.array(balls[sl], np.int32)),
                             current_player=jnp.array(players[sl], jnp.int32), is_jumping=jnp.array(jumping[sl]),
                             terminated=jnp.zeros(m_, bool), winner=jnp.zeros(m_, jnp.int32), num_turns=jnp.array(turns[sl], jnp.int32),
                             jump_sequence=jnp.array(np.stack(seqs[sl])), jump_sequence_length=jnp.array(seqlen[sl], jnp.int32))
        S.append(np.array(enc(st), np.float32))
    return np.concatenate(S), np.stack(P), np.ones(len(P), np.float32)


# --------------------------------------------------------------------------------------------------------------
def build_clutter(n_games: int, out: Path):
    from expert.dataset import play_record
    pats = []
    for g in range(n_games):
        rec, _ = play_record(10_000 + g, 21, 15, 0.1, 300, "expert")
        for s, _a in rec[5::6]:                                    # every 6th decision of each game
            pats.append(np.array([1 if v == MAN else 0 for v in s.board], np.int8))
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, patterns=np.stack(pats))
    print(f"clutter pool: {len(pats)} stone patterns from {n_games} expert games -> {out}")


def eval_set(per_cell: int, out: Path, seed: int = 12345):
    """Frozen held-out puzzles: engine boards + the winning first moves; bucketed by TRUE difficulty."""
    gen = Generator(seed=seed); recs = []
    for cell in CELLS:
        got = 0; tries = 0
        while got < per_cell and tries < per_cell * 4:
            tries += 1; s = gen.puzzle(cell); comp = best_completions(s)
            if not comp: continue
            d = min(v[0] for v in comp.values())
            recs.append(dict(board=np.array(s.board, np.int8), ball=s.ball, player=s.player, cell=str(cell), difficulty=d,
                             winning_first=np.array(sorted(comp), np.int32)))
            got += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, recs=np.array(recs, dtype=object))
    from collections import Counter
    print(f"eval set: {len(recs)} puzzles -> {out}; true difficulty: {dict(sorted(Counter(r['difficulty'] for r in recs).items()))}")


def sample(n_puzzles: int, cells, temp: float, material_w: float, seed: int):
    gen = Generator(seed=seed); ex = []; diffs = []
    for i in range(n_puzzles):
        s = gen.puzzle(cells[i % len(cells)]); ch = chain_examples(s, temp, material_w)
        if ch: ex += ch; diffs.append(len(ch))
    return ex, diffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-clutter", type=int, default=0); ap.add_argument("--eval-set", action="store_true")
    ap.add_argument("--per-cell", type=int, default=40); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.5); ap.add_argument("--material-w", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=DATA / "puzzles.npz")
    x = ap.parse_args(); t0 = time.time()
    if x.build_clutter: build_clutter(x.build_clutter, DATA / "clutter_pool.npz")
    if x.eval_set: eval_set(x.per_cell, DATA / "puzzle_eval.npz")
    if x.sample:
        ex, diffs = sample(x.sample, CELLS, x.temp, x.material_w, x.seed); S, P, V = encode(ex)
        np.savez_compressed(x.out, states=S, policy_targets=P, value_targets=V)
        from collections import Counter
        print(f"{x.sample} puzzles -> {len(V)} chain-state examples in {time.time()-t0:.0f}s; chain lengths "
              f"{dict(sorted(Counter(diffs).items()))}; mean winning moves per state {np.mean((P > 0).sum(1)):.2f}")
