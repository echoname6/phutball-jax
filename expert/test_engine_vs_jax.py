"""Play random games in both engines with identical actions; compare every state.
Run with a Python that has jax installed (e.g. the phutball app venv)."""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
import phutball_env_jax as J  # noqa: E402
from expert.engine import legal_actions, new_game, step  # noqa: E402


def compare(games=30, rows=21, cols=15, seed=0, max_moves=400):
    cfg = J.EnvConfig(rows=rows, cols=cols, max_turns=7200)
    rng = random.Random(seed); checked = 0
    step_j = __import__("jax").jit(J.step, static_argnums=2)
    legal_j = __import__("jax").jit(J.get_legal_actions, static_argnums=1)
    for g in range(games):
        js = J.reset(cfg); ps = new_game(rows, cols)
        for m in range(max_moves):
            lj = set(np.nonzero(np.array(legal_j(js, cfg)))[0].tolist()); lp = set(legal_actions(ps))
            assert lj == lp, f"game {g} move {m}: legal sets differ: jax-only {sorted(lj - lp)[:10]} py-only {sorted(lp - lj)[:10]}"
            # bias toward jumps so chains and wins actually happen
            acts = sorted(lp); n = rows * cols
            jumps = [a for a in acts if a >= n]
            a = rng.choice(jumps) if jumps and rng.random() < 0.7 else rng.choice(acts)
            js = step_j(js, a, cfg); ps = step(ps, a)
            assert np.array_equal(np.array(js.board).ravel(), np.array(ps.board)), f"board differs g{g} m{m}"
            assert int(js.ball_pos[0]) * cols + int(js.ball_pos[1]) == ps.ball
            assert int(js.current_player) == ps.player and bool(js.is_jumping) == ps.jumping
            assert int(js.winner) == ps.winner, f"winner differs g{g} m{m}: jax {int(js.winner)} py {ps.winner}"
            checked += 1
            if ps.winner or bool(js.terminated):
                break
    print(f"PASS: {games} random games, {checked} moves, legal sets + boards + winners identical")


if __name__ == "__main__":
    compare()
