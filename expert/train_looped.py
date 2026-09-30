"""Train the looped transformer (expert/looped.py) or a matched fixed-depth baseline on the same data and steps.

Data per batch (shares): jump-chain puzzles J1-J4 (exact targets, every chain state), placement puzzles (unstoppable
threat / block / prevent), and game positions from the 21x15 expert games labelled by a TEACHER network (the self-play
final: its policy as soft targets, its value as the value target). Puzzle depth never exceeds 4 jumps in training;
the deep set (expert_data/puzzle_eval_deep.npz, true depth 5-10) measures extrapolation.

Looped model: each step runs T loops (T drawn from --loops-train), loss on EVERY loop (weights rising with t), and a
stop head trained to predict "this loop's top move equals the last loop's" (BCE). Fixed model: the repo transformer.

Eval (every --eval-every steps): chain success by TRUE depth on the standard and deep sets, at each T in --eval-loops
(looped) or once (fixed), and top-1 agreement with the teacher on held-out games.

  python -m expert.train_looped --model looped --teacher <selfplay final params .pkl> --run-dir expert_data/looped/run1
  python -m expert.train_looped --model fixed  --teacher <same>                        --run-dir expert_data/looped/fixed1
"""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(root, pattern, stride=1):
    files = sorted(glob.glob(str(root / pattern)))
    if not files: raise FileNotFoundError(root / pattern)
    S, P, V, W = [], [], [], []
    for f in files:
        z = np.load(f); sl = slice(None, None, stride)
        S.append(z["states"][sl].astype(np.int8)); P.append(z["policy_targets"][sl].astype(np.float16))
        V.append(z["value_targets"][sl].astype(np.float32))
        W.append(z["value_weights"][sl].astype(np.float32) if "value_weights" in z else np.ones(len(V[-1]), np.float32))
    return [np.concatenate(x) for x in (S, P, V, W)]


def chain_eval(policy, recs, per_depth, rows=21, cols=15):
    from expert.engine import State, legal_actions, step
    N = rows * cols; res = defaultdict(lambda: [0, 0]); seen = defaultdict(int)
    for r in recs:
        d = int(r["difficulty"])
        if seen[d] >= per_depth: continue
        seen[d] += 1
        s = State(rows, cols, [int(v) for v in r["board"]], int(r["ball"]), int(r["player"])); me = s.player
        a = policy.action(s); won = False
        for _ in range(60):
            if a not in legal_actions(s): break
            s = step(s, a)
            if s.winner: won = s.winner == me; break
            if s.player != me or not s.jumping: break
            a = policy.action(s)
        res[d][0] += int(won); res[d][1] += 1
    return {d: v[0] / v[1] for d, v in sorted(res.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("looped", "fixed"), required=True)
    ap.add_argument("--teacher", type=Path, required=True); ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=2, help="looped: blocks per loop"); ap.add_argument("--layers", type=int, default=6, help="fixed: layers")
    ap.add_argument("--loops-train", default="2,4,6,8"); ap.add_argument("--eval-loops", default="1,2,4,8,16,32")
    ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--eval-per-depth", type=int, default=25)
    ap.add_argument("--share", default="0.4,0.15,0.45", help="jump puzzles, placement puzzles, teacher-labelled games")
    ap.add_argument("--stop-w", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-thresholds", default="0.5,0.8,0.95,0.99",
                    help="adaptive thinking: stop at the first loop whose stop probability exceeds each threshold; one "
                         "accuracy vs average-compute point per threshold (the looped curve to set against search)")
    ap.add_argument("--replay", type=Path, default=None, help="self-play replay buffer (buffers_21x15.npz): its MCTS visit "
                    "distributions and game results replace the teacher-labelled expert games (distil the SEARCH, not the net)")
    ap.add_argument("--search-sims", default="0,8,32,128", help="baseline: the teacher network with this many Gumbel sims")
    a = ap.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(a.run_dir / "train.log", "a")

    def say(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import jax
    if not hasattr(jax.core, "get_opaque_trace_state"):
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    import jax.numpy as jnp
    import optax
    from network import create_transformer_network
    from expert.looped import LoopedPhutball, LoopedPolicy
    from expert.net_policy import NetPolicy
    from expert.puzzle_eval import load_set

    R, C = 21, 15
    teacher_net = create_transformer_network(rows=R, cols=C, d_model=128, n_layers=6, n_heads=4, ffn_dim=256, pos_encoding="goal_distance")
    tp = pickle.load(open(a.teacher, "rb")); tp = tp.get("params", tp)
    teach = jax.jit(lambda x: teacher_net.apply({"params": tp}, x, train=False))
    if a.model == "looped":
        net = LoopedPhutball(rows=R, cols=C, d_model=a.width, n_heads=4, ffn_dim=2 * a.width, blocks_per_loop=a.blocks)
        params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, 9, R, C)), 1)["params"]
    else:
        net = create_transformer_network(rows=R, cols=C, d_model=a.width, n_layers=a.layers, n_heads=4, ffn_dim=2 * a.width, pos_encoding="goal_distance")
        params = net.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, 9, R, C)), train=False)["params"]
    say(f"{a.model}: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters | devices {jax.devices()}")

    say("loading data ...")
    jumps = load(a.data_root, "expert_data/pools/J*_*.npz")
    place = [load(a.data_root, f"expert_data/place_pools/{k}_*.npz") for k in ("forced", "block", "prevent")]
    place = [np.concatenate(x) for x in zip(*place)]
    if a.replay:                                       # search targets: MCTS visit distributions + game results
        z = np.load(a.replay); n_rep = len(z["V"]); cut = int(n_rep * 0.97)
        rep_S, rep_P, rep_V = z["S"].astype(np.int8), z["P"].astype(np.float16), z["V"].astype(np.float32)
        games = rep_S[:cut]; held = rep_S[cut:][::4]; held_target = rep_P[cut:][::4].astype(np.float32).argmax(1)
        say(f"replay: {n_rep:,} positions with 128-sim search targets ({len(games):,} train, {len(held):,} held out)")
    else:
        games = load(a.data_root, "expert_data/games/chunk_[0-8].npz", stride=2)[0]
        held = load(a.data_root, "expert_data/games/chunk_9.npz", stride=8)[0]; held_target = None
    recs_std = load_set(a.data_root / "expert_data" / "puzzle_eval.npz")
    recs_deep = list(np.load(a.data_root / "expert_data" / "puzzle_eval_deep.npz", allow_pickle=True)["recs"])
    shares = [float(x) for x in a.share.split(",")]; nb = [int(round(a.batch * s)) for s in shares]; nb[2] = a.batch - nb[0] - nb[1]
    say(f"jump states {len(jumps[0]):,}, placement puzzles {len(place[0]):,}, game positions {len(games):,} ({'search targets' if a.replay else 'teacher-labelled'}) | batch split {nb}")

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=1e-4))
    ck = a.run_dir / "latest.pkl"
    if ck.exists():
        d = pickle.load(open(ck, "rb")); params, opt_state, step0 = d["params"], d["opt_state"], d["step"]; say(f"resumed at step {step0}")
    else:
        opt_state = opt.init(params); step0 = 0

    def losses(logits, v, P, V, W):
        lp = jax.nn.log_softmax(logits)
        pol = -jnp.mean(jnp.sum(P * lp, -1)); val = jnp.sum(W * jnp.square(v - V)) / jnp.maximum(W.sum(), 1.0)
        return pol, val

    steps_cache = {}
    def step_fn(T):
        if T in steps_cache: return steps_cache[T]
        def loss_fn(p, x, P, V, W):
            if a.model == "fixed":
                logits, v = net.apply({"params": p}, x, train=True); pol, val = losses(logits, v, P, V, W)
                return pol + val, {"policy": pol, "value": val, "stop": 0.0}
            Ls, Vs, Ss = net.apply({"params": p}, x, T, True)
            wt = jnp.arange(1, T + 1, dtype=jnp.float32); wt = wt / wt.sum()
            pol_t, val_t = jax.vmap(lambda l, v: losses(l, v, P, V, W))(Ls, Vs)
            final = jax.lax.stop_gradient(jnp.argmax(Ls[-1], -1))
            same = (jax.lax.stop_gradient(jnp.argmax(Ls, -1)) == final[None]).astype(jnp.float32)
            stop = jnp.mean(optax.sigmoid_binary_cross_entropy(Ss, same))
            tot = jnp.sum(wt * (pol_t + val_t)) + a.stop_w * stop
            return tot, {"policy": pol_t[-1], "value": val_t[-1], "stop": stop}

        @jax.jit
        def f(p, o, x, P, V, W):
            (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, P, V, W)
            u, o = opt.update(g, o, p); return optax.apply_updates(p, u), o, m
        steps_cache[T] = f; return f

    rng = np.random.default_rng(a.seed + step0); loops_train = [int(x) for x in a.loops_train.split(",")]

    def batch():
        i = rng.integers(0, len(jumps[0]), nb[0]); j = rng.integers(0, len(place[0]), nb[1]); k = rng.integers(0, len(games), nb[2])
        if a.replay:
            tP = rep_P[k].astype(np.float32); tv = rep_V[k]
        else:
            gx = games[k].astype(np.float32); tl, tv = teach(gx)
            tP = np.array(jax.nn.softmax(tl), np.float32)
        x = np.concatenate([jumps[0][i], place[0][j], games[k]]).astype(np.float32)
        P = np.concatenate([jumps[1][i].astype(np.float32), place[1][j].astype(np.float32), tP])
        V = np.concatenate([jumps[2][i], place[2][j], np.array(tv, np.float32)])
        W = np.concatenate([jumps[3][i], place[3][j], np.ones(nb[2], np.float32)])
        return x, P, V, W

    def evaluate(p, step):
        out = {}
        tl = held_target if held_target is not None else np.array(teach(held.astype(np.float32))[0]).argmax(1)
        if a.model == "looped":
            fwd = {}
            for T in [int(x) for x in a.eval_loops.split(",")]:
                if T not in fwd: fwd[T] = jax.jit(lambda q, x, T=T: net.apply({"params": q}, x, T, False)[0][-1])
                agree = float((np.array(fwd[T](p, held.astype(np.float32))).argmax(1) == tl).mean())
                pol = LoopedPolicy(net, p, T)
                std = chain_eval(pol, recs_std, a.eval_per_depth); deep = chain_eval(pol, recs_deep, a.eval_per_depth)
                out[T] = (agree, std, deep)
            deep_adaptive = []
            for th in [float(x) for x in a.stop_thresholds.split(",")]:
                pol = LoopedPolicy(net, p, max(int(x) for x in a.eval_loops.split(",")), stop_at=th)
                res = chain_eval(pol, recs_deep, a.eval_per_depth)
                deep_adaptive.append((th, res, float(np.mean(pol.used)) if pol.used else float("nan")))
            used = None
        else:
            agree = float((np.array(jax.jit(lambda q, x: net.apply({"params": q}, x, train=False)[0])(p, held.astype(np.float32))).argmax(1) == tl).mean())
            pol = NetPolicy(net, p); out[0] = (agree, chain_eval(pol, recs_std, a.eval_per_depth), chain_eval(pol, recs_deep, a.eval_per_depth))
            deep_adaptive, used = None, None
        for T, (agree, std, deep) in out.items():
            blk = f"~{2 * T} blocks/move" if a.model == "looped" else f"~{a.layers} blocks/move"
            say(f"  [eval step {step}] {'T=' + str(T) if a.model == 'looped' else 'fixed'} ({blk}) | "
                f"{'search-target' if a.replay else 'teacher'} top-1 {agree:.1%} | standard "
                + " ".join(f"{d}j {v:.0%}" for d, v in std.items()) + " | deep " + " ".join(f"{d}j {v:.0%}" for d, v in deep.items()))
        if deep_adaptive is not None:
            for th, res, used in deep_adaptive:
                say(f"  [eval step {step}] adaptive stop > {th} (up to T={max(out)}) | mean loops {used:.1f} (~{2 * used:.0f} blocks/move) | deep "
                    + " ".join(f"{d}j {v:.0%}" for d, v in res.items()))
                with open(a.run_dir / "evals.csv", "a", newline="") as f:
                    csv.writer(f).writerow([step, f"adaptive{th}", round(used, 2)] + [f"deep{d}:{v:.3f}" for d, v in res.items()])
        with open(a.run_dir / "evals.csv", "a", newline="") as f:
            w = csv.writer(f)
            for T, (agree, std, deep) in out.items():
                w.writerow([step, T, round(agree, 4)] + [f"std{d}:{v:.3f}" for d, v in std.items()] + [f"deep{d}:{v:.3f}" for d, v in deep.items()])

    def search_baseline():
        """The teacher network with N Gumbel simulations (16 root candidates), greedy on the visit distribution."""
        from phutball_env_jax import EnvConfig
        from self_play_batched import make_transformer_recurrent_fn, transformer_mcts_policy
        cfg = EnvConfig(rows=R, cols=C); rf = make_transformer_recurrent_fn(teacher_net, cfg); helper = NetPolicy(teacher_net, tp)
        f = a.run_dir / "search_baseline.csv"
        if f.exists(): say("search baseline: see search_baseline.csv"); return
        rows_out = []
        for n_sims in [int(x) for x in a.search_sims.split(",")]:
            if n_sims == 0: pol = helper
            else:
                run = jax.jit(lambda st, r, n=n_sims: transformer_mcts_policy({"network_params": tp}, st, r, teacher_net, cfg,
                              num_simulations=n, temperature=1.0, max_num_considered_actions=16, recurrent_fn=rf)[1])
                class SearchPolicy:
                    def __init__(self): self.k = jax.random.PRNGKey(0)
                    def action(self, s):
                        st = jax.tree_util.tree_map(lambda x: x[None], helper._jax_state(s)); self.k, r = jax.random.split(self.k)
                        return int(np.array(run(st, r))[0].argmax())
                pol = SearchPolicy()
            std = chain_eval(pol, recs_std, a.eval_per_depth); deep = chain_eval(pol, recs_deep, a.eval_per_depth)
            blocks = 6 * (n_sims + 1)
            say(f"  [search baseline] teacher + {n_sims} sims (~{blocks} blocks/move) | standard "
                + " ".join(f"{d}j {v:.0%}" for d, v in std.items()) + " | deep " + " ".join(f"{d}j {v:.0%}" for d, v in deep.items()))
            rows_out.append([n_sims, blocks] + [f"std{d}:{v:.3f}" for d, v in std.items()] + [f"deep{d}:{v:.3f}" for d, v in deep.items()])
        with open(f, "w", newline="") as fh: csv.writer(fh).writerows(rows_out)

    if a.model == "looped" and a.search_sims.strip(): search_baseline()      # --search-sims "" skips it (measured once)
    t0 = time.time(); acc = []; step = step0
    while step < a.steps:
        T = int(rng.choice(loops_train)) if a.model == "looped" else 0
        x, P, V, W = batch()
        params, opt_state, m = step_fn(T)(params, opt_state, x, P, V, W); step += 1
        acc.append({k: float(v) for k, v in m.items()})
        if step % 50 == 0:
            mm = {k: np.mean([q[k] for q in acc]) for k in acc[0]}; acc = []
            say(f"step {step}/{a.steps} | policy {mm['policy']:.3f} value {mm['value']:.3f} stop {mm['stop']:.3f} | {(time.time() - t0) / 60:.1f} min")
        if step % a.eval_every == 0 or step == a.steps:
            pickle.dump({"params": params, "opt_state": opt_state, "step": step, "config": vars(a)}, open(str(ck) + ".tmp", "wb"))
            Path(str(ck) + ".tmp").replace(ck)
            evaluate(params, step)
    say("done")


if __name__ == "__main__":
    main()
