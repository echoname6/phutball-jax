"""Play a trained network (raw policy, no MCTS) in the fast Python engine, for evaluation."""
from __future__ import annotations

import numpy as np

from expert.engine import legal_actions


class NetPolicy:
    def __init__(self, network, params, rows=21, cols=15, temperature=0.0, seed=0):
        import jax
        import phutball_env_jax as J
        self.J, self.rows, self.cols, self.temp = J, rows, cols, temperature
        self.cfg = J.EnvConfig(rows=rows, cols=cols)
        self.apply = jax.jit(lambda x: network.apply({"params": params}, x, train=False))
        self.enc = jax.jit(lambda st: J.state_to_network_input(st, self.cfg))
        self.rng = np.random.default_rng(seed)

    def _jax_state(self, s):
        import jax.numpy as jnp
        J, rows, cols = self.J, self.rows, self.cols
        seq = np.full((J.MAX_JUMP_SEQUENCE_LENGTH, 2), -1, np.int32)
        for k, pos in enumerate(s.seq[: J.MAX_JUMP_SEQUENCE_LENGTH]): seq[k] = divmod(pos, cols)
        return J.PhutballState(board=jnp.array(np.array(s.board, np.int32).reshape(rows, cols)),
                               ball_pos=jnp.array(divmod(s.ball, cols), jnp.int32), current_player=jnp.int32(s.player),
                               is_jumping=jnp.bool_(s.jumping), terminated=jnp.bool_(False), winner=jnp.int32(0),
                               num_turns=jnp.int32(s.turns), jump_sequence=jnp.array(seq),
                               jump_sequence_length=jnp.int32(len(s.seq) if s.jumping else 0))

    def action(self, s):
        n = self.rows * self.cols
        x = self.enc(self._jax_state(s))[None]
        logits = np.array(self.apply(x)[0][0], np.float64)
        if s.player == 2:                                   # visual -> physical
            logits[:n] = logits[:n][::-1].copy(); logits[n:2 * n] = logits[n:2 * n][::-1].copy()
        legal = np.array(legal_actions(s))
        z = logits[legal]
        if self.temp <= 0:
            return int(legal[int(np.argmax(z))])
        p = np.exp((z - z.max()) / self.temp); p /= p.sum()
        return int(self.rng.choice(legal, p=p))
