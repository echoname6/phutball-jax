"""Looped (weight-shared, recurrent-depth) transformer for phutball: no language, no search.

Same inputs as network.PhutballTransformer: one token per square with 11 features (4 board planes, 5 jump-sequence
planes, goal-distance row and centred column). Plus `n_scratch` learned scratch tokens with no square of their own.

  e      = [Dense(board tokens), scratch embeddings]                     (the problem, re-read every loop)
  h_0    = e
  h_t+1  = Blocks(Adapter([h_t ; e]))       Blocks = `blocks_per_loop` TransformerBlocks, SHARED across loops
  heads after EVERY loop (shared): per-square placement / jump logits, halt-move logit and value from the mean of the
  square tokens, and a stop logit from scratch token 0 ("my current answer is final").

Compute per loop = `blocks_per_loop` blocks; T loops of 2 blocks ~ a 2T-layer transformer, with 1/3 of the block
parameters of the 6-layer model when blocks_per_loop = 2.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from network import TransformerBlock


class LoopedPhutball(nn.Module):
    rows: int = 21
    cols: int = 15
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    blocks_per_loop: int = 2
    n_scratch: int = 4

    def setup(self):
        self.embed = nn.Dense(self.d_model)
        self.scratch = self.param("scratch", nn.initializers.normal(0.02), (self.n_scratch, self.d_model))
        self.adapter = nn.Dense(self.d_model)
        self.blocks = [TransformerBlock(d_model=self.d_model, n_heads=self.n_heads, ffn_dim=self.ffn_dim)
                       for _ in range(self.blocks_per_loop)]
        self.final_ln = nn.LayerNorm()
        self.place_head = nn.Dense(1); self.jump_head = nn.Dense(1); self.halt_head = nn.Dense(1)
        self.value_1 = nn.Dense(64); self.value_2 = nn.Dense(1); self.stop_head = nn.Dense(1)

    def tokens(self, x):
        B = x.shape[0]; R, C = self.rows, self.cols
        x = jnp.transpose(x, (0, 2, 3, 1))                                   # NCHW -> NHWC
        row = jnp.broadcast_to((jnp.arange(R) - 1)[None, :, None, None], (B, R, C, 1)).astype(x.dtype)
        col = jnp.broadcast_to((jnp.arange(C) - (C - 1) / 2)[None, None, :, None], (B, R, C, 1)).astype(x.dtype)
        t = jnp.concatenate([x[..., 0:4], x[..., 4:9], row, col], axis=-1).reshape(B, R * C, 11)
        board = self.embed(t)
        scratch = jnp.broadcast_to(self.scratch[None], (B, self.n_scratch, self.d_model))
        return jnp.concatenate([board, scratch], axis=1)

    def heads(self, h):
        N = self.rows * self.cols
        z = self.final_ln(h); sq = z[:, :N]
        pooled = jnp.mean(sq, axis=1)
        policy = jnp.concatenate([self.place_head(sq)[..., 0], self.jump_head(sq)[..., 0], self.halt_head(pooled)], axis=-1)
        value = jnp.tanh(self.value_2(nn.gelu(self.value_1(pooled))))[..., 0]
        stop = self.stop_head(z[:, N])[..., 0]
        return policy, value, stop

    def __call__(self, x, loops: int, train: bool = False):
        """Returns per-loop outputs: policy (T, B, 2N+1), value (T, B), stop logit (T, B)."""
        e = self.tokens(x); h = e; P, V, S = [], [], []
        for _ in range(loops):
            h = self.adapter(jnp.concatenate([h, e], axis=-1))
            for blk in self.blocks:
                h = blk(h, train=train)
            p, v, s = self.heads(h); P.append(p); V.append(v); S.append(s)
        return jnp.stack(P), jnp.stack(V), jnp.stack(S)


class LoopedPolicy:
    """expert/arena-style policy (.action(state)) for a looped network at a fixed number of loops (or adaptive:
    stop at the first loop whose stop probability exceeds `stop_at`, up to `loops`)."""

    def __init__(self, net, params, loops: int, stop_at: float | None = None, rows=21, cols=15):
        import jax
        import numpy as np
        import phutball_env_jax as J
        from expert.net_policy import NetPolicy
        self.np, self.loops, self.stop_at, self.rows, self.cols = np, loops, stop_at, rows, cols
        self.apply = jax.jit(lambda x: net.apply({"params": params}, x, loops, False))
        self._helper = NetPolicy.__new__(NetPolicy); self._helper.J, self._helper.rows, self._helper.cols = J, rows, cols
        self.enc = jax.jit(lambda st: J.state_to_network_input(st, J.EnvConfig(rows=rows, cols=cols)))
        self.used = []

    def action(self, s):
        from expert.engine import legal_actions
        np = self.np; n = self.rows * self.cols
        x = self.enc(self._helper._jax_state(s))[None]
        P, _, S = self.apply(x); P = np.array(P[:, 0], np.float64); S = np.array(S[:, 0])
        t = self.loops - 1
        if self.stop_at is not None:
            hit = np.flatnonzero(1 / (1 + np.exp(-S)) > self.stop_at); t = int(hit[0]) if len(hit) else self.loops - 1
        self.used.append(t + 1)
        logits = P[t]
        if s.player == 2:
            logits[:n] = logits[:n][::-1].copy(); logits[n:2 * n] = logits[n:2 * n][::-1].copy()
        legal = np.array(legal_actions(s))
        return int(legal[int(np.argmax(logits[legal]))])
