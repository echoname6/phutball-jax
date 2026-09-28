"""Continuous batched self-play: every slot always plays. When a game ends (win, or the turn cap = draw), its slot is
reset to a new game in the same step, so no search is spent on finished games waiting for the batch's longest game.

Games are played in fixed chunks of K steps (one jitted lax.scan per chunk). Each step records, per slot:
  obs      int8 (B, 9, R, C)   network input (0/1 planes, already rotated for P2)
  policy   float16 (B, A)      Gumbel MCTS visit weights, in VISUAL coords (P2 blocks reversed, as in training)
  player / game id / action, and for slots whose game ended at this step: that game's id and winner.
Games still running at the end of a chunk carry over; the host Collector keeps their positions pending until the game
ends, then emits (states, policies, values) with values from each mover's point of view.

The search is the repo's (transformer_mcts_policy, Gumbel MuZero); its sampled action is ignored and each slot samples
from the visit weights with its own temperature (temperature until move `temp_moves` of its game, then temp_final).
"""
from __future__ import annotations

from functools import partial

import numpy as np


def make_selfplay(net, cfg, slots: int, chunk: int, sims: int, considered: int, temp: float, temp_final: float,
                  temp_moves: int, max_turns: int):
    import jax
    import jax.numpy as jnp
    from phutball_env_jax import reset, state_to_network_input, step
    from self_play_batched import make_transformer_recurrent_fn, transformer_mcts_policy

    R, C = cfg.rows, cfg.cols; N = R * C
    recurrent_fn = make_transformer_recurrent_fn(net, cfg)
    fresh = reset(cfg)

    def init_carry():
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (slots,) + x.shape), fresh)
        return {"states": states, "move": jnp.zeros(slots, jnp.int32), "gid": jnp.arange(slots, dtype=jnp.int32),
                "next_gid": jnp.int32(slots)}

    def to_visual(pol, p2):
        flipped = jnp.concatenate([pol[:, :N][:, ::-1], pol[:, N:2 * N][:, ::-1], pol[:, 2 * N:]], axis=1)
        return jnp.where(p2[:, None], flipped, pol)

    def one_step(params, carry, rng):
        st = carry["states"]
        r_mcts, r_samp = jax.random.split(rng)
        obs = jax.vmap(lambda s: state_to_network_input(s, cfg))(st)
        _, weights, _ = transformer_mcts_policy({"network_params": params}, st, r_mcts, net, cfg, num_simulations=sims,
                                                temperature=1.0, max_num_considered_actions=considered,
                                                recurrent_fn=recurrent_fn)
        t = jnp.where(carry["move"] < temp_moves, temp, temp_final)[:, None]
        logits = jnp.log(weights + 1e-8) / jnp.maximum(t, 1e-3)
        sampled = jax.random.categorical(r_samp, logits, axis=-1)
        actions = jnp.where(t[:, 0] < 0.01, jnp.argmax(weights, -1), sampled).astype(jnp.int32)
        new = jax.vmap(lambda s, a: step(s, a, cfg))(st, actions)
        ended = new.terminated | (new.num_turns >= max_turns)
        winner = jnp.where(new.terminated, new.winner, 0)
        rec = {"obs": obs.astype(jnp.int8), "policy": to_visual(weights, st.current_player == 2).astype(jnp.float16),
               "player": st.current_player.astype(jnp.int8), "gid": carry["gid"], "action": actions,
               "ended_gid": jnp.where(ended, carry["gid"], -1), "winner": winner.astype(jnp.int8)}
        # reset ended slots to a fresh game with a new id
        def pick(f, n):
            e = ended.reshape((slots,) + (1,) * (n.ndim - 1))
            return jnp.where(e, jnp.broadcast_to(f, n.shape), n)
        states = jax.tree_util.tree_map(pick, fresh, new)
        new_ids = carry["next_gid"] + jnp.cumsum(ended.astype(jnp.int32)) - 1
        carry = {"states": states, "move": jnp.where(ended, 0, carry["move"] + 1),
                 "gid": jnp.where(ended, new_ids, carry["gid"]), "next_gid": carry["next_gid"] + ended.sum()}
        return carry, rec

    @jax.jit
    def run_chunk(params, carry, rng):
        def body(c, r): return one_step(params, c, r)
        return jax.lax.scan(body, carry, jax.random.split(rng, chunk))

    return init_carry, run_chunk


class Collector:
    """Host side: holds positions of unfinished games; emits finished games as training examples."""

    def __init__(self):
        self.pend = None

    def add(self, rec):
        r = {k: np.asarray(v) for k, v in rec.items()}
        K, B = r["gid"].shape
        flat = {"obs": r["obs"].reshape((K * B,) + r["obs"].shape[2:]), "policy": r["policy"].reshape(K * B, -1),
                "player": r["player"].reshape(-1), "gid": r["gid"].reshape(-1), "action": r["action"].reshape(-1)}
        eg = r["ended_gid"].reshape(-1); ew = r["winner"].reshape(-1); m = eg >= 0
        ids, wins = eg[m].astype(np.int64), ew[m]                 # games that ended in this chunk (each id once)
        if self.pend is not None: flat = {k: np.concatenate([self.pend[k], flat[k]]) for k in flat}
        fin = np.isin(flat["gid"], ids)                           # every position of those games, pending or new
        self.pend = {k: v[~fin] for k, v in flat.items()}
        out = {k: v[fin] for k, v in flat.items()}
        order = np.argsort(ids); w = wins[order][np.searchsorted(ids[order], out["gid"])] if fin.any() else np.zeros(0, np.int8)
        values = np.where(w == 0, 0.0, np.where(w == out["player"], 1.0, -1.0)).astype(np.float32)
        lengths = np.bincount(np.searchsorted(ids[order], out["gid"]), minlength=len(ids)) if fin.any() else np.zeros(0)
        games = {"n": len(ids), "p1": int((wins == 1).sum()), "draws": int((wins == 0).sum()),
                 "mean_moves": float(lengths.mean()) if len(lengths) else float("nan"), "pending": len(self.pend["gid"])}
        return out, values, games
