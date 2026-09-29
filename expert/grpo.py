"""GRPO for phutball on the GPU: grouped rollouts from restored in-progress positions, with a light Gumbel search per
move and per-rollout sampling (so the rollouts in a group differ), RLOO advantages, and a policy-gradient update with
value, puzzle and KL anchors. Used by expert/grpo_pbt.py.

Rollouts (make_rollout): B start states, each repeated G times (S = B*G slots). Each move: the repo's Gumbel search
(sims > 0) or the raw policy (sims = 0) gives a distribution; the move is sampled at temperature tau. A rollout ends at a
win or after H micro-actions; an unfinished rollout is scored by the value head (bootstrapped), from the root player's
side. Recorded per move: network input (int8), action in VISUAL coordinates, the mover's side relative to the root
player (+1 / -1), and whether the slot was still playing.

Update (make_update): per group, RLOO advantage A_i = r_i - mean_{j != i} r_j. Every move of rollout i gets weight
A_i * side (zero-sum: the opponent's moves get the negated advantage), so both sides' moves are trained. Loss =
policy gradient (moves sampled from informative groups only) + value MSE on the rollout results + kl_final *
KL(anchor || model) on rollout states + puz_w * (puzzle CE + kl_puz * KL(puzzle model || model)).
"""
from __future__ import annotations

import numpy as np


def visual_action(a, p2, N):
    """Physical action -> visual (180-degree rotated) action for player 2."""
    import jax.numpy as jnp
    flipped = jnp.where(a < N, N - 1 - a, jnp.where(a < 2 * N, 3 * N - 1 - a, a))
    return jnp.where(p2, flipped, a)


def make_rollout(net, cfg, B: int, G: int, H: int, sims: int, considered: int):
    import jax
    import jax.numpy as jnp
    from phutball_env_jax import get_legal_actions, state_to_network_input, step
    from self_play_batched import make_transformer_recurrent_fn, transform_policy_for_p2, transformer_mcts_policy

    R, C = cfg.rows, cfg.cols; N = R * C; S = B * G
    rf = make_transformer_recurrent_fn(net, cfg)

    def policy_weights(params, st, rng):
        if sims > 0:
            _, w, _ = transformer_mcts_policy({"network_params": params}, st, rng, net, cfg, num_simulations=sims,
                                              temperature=1.0, max_num_considered_actions=considered, recurrent_fn=rf)
            return w
        obs = jax.vmap(lambda s: state_to_network_input(s, cfg))(st)
        logits, _ = net.apply({"params": params}, obs, train=False)
        logits = jnp.where((st.current_player == 2)[:, None], transform_policy_for_p2(logits, R, C), logits)
        legal = jax.vmap(lambda s: get_legal_actions(s, cfg))(st)
        return jax.nn.softmax(jnp.where(legal == 1, logits, -1e9), -1)

    @jax.jit
    def rollout(params, starts, rng, tau):
        st = jax.tree_util.tree_map(lambda x: jnp.repeat(x, G, axis=0), starts)
        root = st.current_player

        def body(carry, r):
            st, done = carry
            r1, r2 = jax.random.split(r)
            obs = jax.vmap(lambda s: state_to_network_input(s, cfg))(st)
            w = policy_weights(params, st, r1)
            a = jax.random.categorical(r2, jnp.log(w + 1e-8) / jnp.maximum(tau, 1e-3), axis=-1).astype(jnp.int32)
            new = jax.vmap(lambda s, x: step(s, x, cfg))(st, a)
            keep = lambda o, n: jnp.where(done.reshape((S,) + (1,) * (n.ndim - 1)), o, n)
            rec = {"obs": obs.astype(jnp.int8), "act": visual_action(a, st.current_player == 2, N),
                   "side": jnp.where(st.current_player == root, 1.0, -1.0), "active": ~done}
            return (jax.tree_util.tree_map(keep, st, new), done | new.terminated), rec

        (fin, done), recs = jax.lax.scan(body, (st, jnp.zeros(S, bool)), jax.random.split(rng, H))
        obs = jax.vmap(lambda s: state_to_network_input(s, cfg))(fin)
        _, v = net.apply({"params": params}, obs, train=False)
        v_root = jnp.where(fin.current_player == root, v, -v)
        won = jnp.where(fin.winner == root, 1.0, jnp.where(fin.winner == 0, 0.0, -1.0))
        reward = jnp.where(done, won, v_root)
        return recs, reward, done

    return rollout


def rloo(reward, G):
    """(S,) rewards in groups of G -> leave-one-out advantages."""
    import jax.numpy as jnp
    r = reward.reshape(-1, G); tot = r.sum(1, keepdims=True)
    return (r - (tot - r) / (G - 1)).reshape(-1)


def make_update(net, optimizer, K: int, M: int):
    """Returns build(G) -> jitted update: K optimizer steps (one lax.scan) on one rollout batch with groups of G,
    each step M policy-gradient moves (from informative groups) + M value moves + one puzzle minibatch."""
    import jax
    import jax.numpy as jnp
    import optax

    def kl(anchor_logits, logp):
        alp = jax.nn.log_softmax(anchor_logits); return jnp.mean(jnp.sum(jnp.exp(alp) * (alp - logp), -1))

    def build(G):
        @jax.jit
        def update(p, o, anchor, pref, recs, reward, rng, puz, kl_final, kl_puz, puz_w):
            H, S = recs["act"].shape
            adv = rloo(reward, G)
            obs = recs["obs"].reshape((H * S,) + recs["obs"].shape[2:])
            act = recs["act"].reshape(-1); side = recs["side"].reshape(-1); active = recs["active"].reshape(-1)
            w = (jnp.tile(adv, H) * side)                              # per-move advantage, mover's side
            vt = (jnp.tile(reward, H) * side)                          # value target, mover's side
            pg_ok = active & (jnp.abs(w) > 1e-6)
            p_pg = jnp.where(pg_ok.sum() > 0, pg_ok, active).astype(jnp.float32); p_pg = p_pg / p_pg.sum()
            p_v = active.astype(jnp.float32) / jnp.maximum(active.sum(), 1)

            def loss_fn(p, ipg, iv, pb):
                x = obs[ipg].astype(jnp.float32)
                logits, _ = net.apply({"params": p}, x, train=True); logp = jax.nn.log_softmax(logits)
                lp_a = jnp.take_along_axis(logp, act[ipg][:, None], 1)[:, 0]
                pg = -jnp.mean(w[ipg] * lp_a)
                al = jax.lax.stop_gradient(net.apply({"params": anchor}, x, train=False)[0]); klf = kl(al, logp)
                _, v = net.apply({"params": p}, obs[iv].astype(jnp.float32), train=True)
                val = jnp.mean(jnp.square(v - vt[iv]))
                pl, pv = net.apply({"params": p}, pb["states"].astype(jnp.float32), train=True)
                plp = jax.nn.log_softmax(pl)
                pce = -jnp.mean(jnp.sum(pb["policy_targets"].astype(jnp.float32) * plp, -1))
                pw = pb["value_weights"]; pval = jnp.sum(pw * jnp.square(pv - pb["value_targets"])) / jnp.maximum(pw.sum(), 1.0)
                prl = jax.lax.stop_gradient(net.apply({"params": pref}, pb["states"].astype(jnp.float32), train=False)[0])
                klp = kl(prl, plp)
                ent = -jnp.mean(jnp.sum(jnp.exp(logp) * logp, -1))
                tot = pg + val + kl_final * klf + puz_w * (pce + pval + kl_puz * klp)
                return tot, {"pg": pg, "value": val, "kl_final": klf, "puzzle_ce": pce, "puzzle_kl": klp, "entropy": ent}

            def one(carry, xs):
                p, o = carry; r, pb = xs
                r1, r2 = jax.random.split(r)
                ipg = jax.random.choice(r1, H * S, (M,), p=p_pg); iv = jax.random.choice(r2, H * S, (M,), p=p_v)
                (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, ipg, iv, pb)
                u, o = optimizer.update(g, o, p); return (optax.apply_updates(p, u), o), m

            (p, o), ms = jax.lax.scan(one, (p, o), (jax.random.split(rng, K), puz))
            stats = jax.tree_util.tree_map(jnp.mean, ms)
            grp = reward.reshape(-1, G)
            stats["informative_groups"] = jnp.mean(grp.max(1) - grp.min(1) > 0.05)
            return p, o, stats
        return update

    return build
