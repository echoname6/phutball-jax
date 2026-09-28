"""Self-play board-size ladder (GPU / Colab): Gumbel-MCTS self-play at 13x9 -> 15x11 -> 17x13 -> 21x15, one
size-agnostic parameter set, starting from the expert-game ladder checkpoint.

Each iteration, at the current size:
  1. continuous self-play (expert/selfplay_fast.py): --slots games in parallel, a finished slot restarts at once, until
     --games games have finished (games in progress carry over to the next iteration);
  2. add them to that size's replay ring;
  3. train --reuse x (new examples) / (current-size share of the batch) steps, --scan steps per jitted lax.scan
     (int8 / float16 host batches, converted on the device). Every step sums fixed-count sub-batches:
       current size   self-play policy (MCTS visit targets) + value (game results)
       earlier sizes  their frozen replay rings, plus kl_prev(t) * KL(promoted snapshot || model), halving every
                      --kl-prev-half iterations after a promotion
       puzzles        21x15 proven targets (value only where known) + --kl-puzzle * KL(puzzle model || model)
Every --eval-every iterations: a match against the snapshot from the previous evaluation (--match-games per colour
pair, main network side recomputed from the same key the game loop uses), 21x15 puzzle scores, self-play statistics.
Promotion when the match score stays below --promote-below for --patience evaluations (after --min-iters), or at
--max-iters. The last size never promotes; it keeps milestone checkpoints.

Search: SEARCH gives (simulations, root candidates) phases per size: 32x16 on the small boards; 21x15 starts at 64x16
and moves to 128x16 when 64 plateaus.
Live control: <run-dir>/control.json is re-read every iteration. Keys (all optional): sims, considered (override SEARCH), games, reuse, lr,
kl_puzzle, kl_prev, kl_prev_half, share_old, share_puzzle, eval_every, match_games, promote_below, patience,
min_iters, max_iters, temperature, pause (sleep until cleared), promote_now, stop (checkpoint and exit 0),
reload (checkpoint and exit 3: the notebook pulls the branch and relaunches).
Outputs in <run-dir>: train.log, log.csv, status.json, latest.pkl (+ buffers_*.npz), milestone checkpoints.

  python -m expert.selfplay_ladder --init expert_data/ladder/rung3_21x15.pkl --puzzle-ref expert_data/place_run/latest.pkl \\
      --run-dir /content/drive/MyDrive/phutball/selfplay --data-root /content/data
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIZES = [(13, 9), (15, 11), (17, 13), (21, 15)]
# search schedule per size: (simulations, Gumbel root candidates) phases; a plateau moves to the next phase before the
# size promotes (21x15: 64 sims, then 128 once 64 plateaus)
SEARCH = {(13, 9): [(32, 16)], (15, 11): [(32, 16)], (17, 13): [(32, 16)], (21, 15): [(64, 16), (128, 16)]}
MAX_TURNS = {(13, 9): 200, (15, 11): 250, (17, 13): 300, (21, 15): 400}
PUZZLES = [("jumps", "expert_data/pools/J*_*.npz"), ("forced", "expert_data/place_pools/forced_*.npz"),
           ("block", "expert_data/place_pools/block_*.npz"), ("prevent", "expert_data/place_pools/prevent_*.npz")]
PUZZLE_EVAL = [("forced", "expert_data/place_pools/heldout/forced.npz"), ("block", "expert_data/place_pools/heldout/block.npz"),
               ("prevent", "expert_data/place_pools/heldout/prevent.npz")]
HOT = ("sims", "considered", "games", "slots", "reuse", "lr", "kl_puzzle", "kl_prev", "kl_prev_half", "share_old", "share_puzzle", "eval_every",
       "match_games", "promote_below", "patience", "min_iters", "max_iters", "temperature")


def load(root: Path, pattern: str):
    files = sorted(glob.glob(str(root / pattern)))
    if not files: raise FileNotFoundError(root / pattern)
    S, P, V, W = [], [], [], []
    for f in files:
        z = np.load(f)
        S.append(z["states"].astype(np.int8)); P.append(z["policy_targets"].astype(np.float16))
        V.append(z["value_targets"].astype(np.float32))
        W.append(z["value_weights"].astype(np.float32) if "value_weights" in z else np.ones(len(V[-1]), np.float32))
    return [np.concatenate(x) for x in (S, P, V, W)]


class Ring:
    """Replay ring for one board size: int8 states, float16 policies."""
    def __init__(self, cap, rc):
        n = rc[0] * rc[1]; self.cap, self.n, self.i = cap, 0, 0
        self.S = np.zeros((cap, 9, rc[0], rc[1]), np.int8); self.P = np.zeros((cap, 2 * n + 1), np.float16)
        self.V = np.zeros(cap, np.float32)

    def add(self, S, P, V):
        k = len(V)
        if k > self.cap: S, P, V, k = S[-self.cap:], P[-self.cap:], V[-self.cap:], self.cap
        idx = (self.i + np.arange(k)) % self.cap
        self.S[idx], self.P[idx], self.V[idx] = S, P, V
        self.i = int((self.i + k) % self.cap); self.n = min(self.n + k, self.cap)

    def sample(self, rng, shape):
        idx = rng.integers(0, self.n, shape)
        return {"states": self.S[idx], "policy_targets": self.P[idx], "value_targets": self.V[idx],
                "value_weights": np.ones(idx.shape, np.float32)}

    def save(self, path):
        np.savez(path, S=self.S[:self.n], P=self.P[:self.n], V=self.V[:self.n], i=self.i)

    def load(self, path):
        z = np.load(path); k = len(z["V"])
        self.S[:k], self.P[:k], self.V[:k] = z["S"], z["P"], z["V"]; self.n, self.i = k, int(z["i"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True); ap.add_argument("--puzzle-ref", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--sizes", default="0,1,2,3")
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--games", type=int, default=512, help="finished self-play games per iteration")
    ap.add_argument("--slots", type=int, default=1024, help="games played in parallel (continuous: a finished slot restarts)")
    ap.add_argument("--chunk", type=int, default=32, help="steps per jitted self-play chunk")
    ap.add_argument("--scan", type=int, default=25, help="training steps per jitted lax.scan call")
    ap.add_argument("--sims", type=int, default=0, help="override the SEARCH schedule (0 = schedule)")
    ap.add_argument("--considered", type=int, default=0, help="override the schedule's root candidates (0 = schedule)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reuse", type=float, default=4.0, help="times each new current-size example is trained on")
    ap.add_argument("--ring", type=int, default=300_000)
    ap.add_argument("--share-old", type=float, default=0.15); ap.add_argument("--share-puzzle", type=float, default=0.15)
    ap.add_argument("--kl-puzzle", type=float, default=0.5)
    ap.add_argument("--kl-prev", type=float, default=1.0); ap.add_argument("--kl-prev-half", type=float, default=10)
    ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--match-games", type=int, default=64)
    ap.add_argument("--promote-below", type=float, default=0.55); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-iters", type=int, default=10); ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--milestone-every", type=int, default=20); ap.add_argument("--eval-per-diff", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True)
    sizes = [SIZES[int(x)] for x in a.sizes.split(",")]
    logf = open(a.run_dir / "train.log", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import jax
    if not hasattr(jax.core, "get_opaque_trace_state"):   # JAX >= 0.11 moved it; older Flax still calls jax.core's
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    import flax
    import jax.numpy as jnp
    import optax
    from network import create_transformer_network
    from phutball_env_jax import EnvConfig
    from self_play_batched import make_transformer_recurrent_fn, play_games_batched, transformer_mcts_policy
    from expert.selfplay_fast import Collector, make_selfplay
    from expert.net_policy import NetPolicy
    from expert.puzzle_eval import evaluate, load_set
    say(f"devices: {jax.devices()} | jax {jax.__version__}, flax {flax.__version__}")

    nets, fns = {}, {}
    def net_for(rc):
        if rc not in nets:
            nets[rc] = create_transformer_network(rows=rc[0], cols=rc[1], d_model=a.width, n_layers=a.layers, n_heads=4,
                                                  ffn_dim=a.width * 2, pos_encoding="goal_distance")
        return nets[rc]

    def games_fn(rc, n_games, search, temp, vs_opponent):
        """Jitted batched games at one size (compiled once per setting). search = (simulations, root candidates)."""
        sims, considered = search; key = (rc, n_games, sims, considered, temp, vs_opponent)
        if key not in fns:
            cfg = EnvConfig(rows=rc[0], cols=rc[1]); T = MAX_TURNS[rc]
            kw = dict(network=net_for(rc), env_config=cfg, batch_size=n_games, max_turns=T, max_moves=2 * T,
                      temperature=temp, temp_threshold=rc[0], temp_final=0.1 if not vs_opponent else temp,
                      num_simulations=sims, max_num_considered_actions=considered, random_opponent_ratio=0.0, mcts_policy_fn=transformer_mcts_policy,
                      recurrent_fn=make_transformer_recurrent_fn(net_for(rc), cfg))
            if vs_opponent:
                fns[key] = jax.jit(lambda p, r, o: play_games_batched({"network_params": p}, r, opponent_params={"network_params": o},
                                                                      opponent_ratio=1.0, **kw))
            else:
                fns[key] = jax.jit(lambda p, r: play_games_batched({"network_params": p}, r, **kw))
        return fns[key]

    sp_cache, sp = {}, {}
    def selfplay(rc, params, rng):
        """Play until --games games have finished; returns int8 states, float16 visual policies, values, stats."""
        sims, cons = search(rc); key = (rc, sims, cons, a.slots, a.chunk, a.temperature)
        if key not in sp_cache:
            sp_cache[key] = make_selfplay(net_for(rc), EnvConfig(rows=rc[0], cols=rc[1]), a.slots, a.chunk, sims, cons,
                                          a.temperature, 0.1, rc[0], MAX_TURNS[rc])
        init, run = sp_cache[key]
        if sp.get("key") != key: sp.update(key=key, carry=init(), col=Collector())      # new size or search setting
        S, P, V = [], [], []; n = p1 = dr = 0; mv = 0.0; chunks = 0
        while n < a.games:
            rng, r = jax.random.split(rng); sp["carry"], rec = run(params, sp["carry"], r); chunks += 1
            out, vals, g = sp["col"].add(rec)
            if g["n"]:
                S.append(out["obs"]); P.append(out["policy"]); V.append(vals)
                n += g["n"]; p1 += g["p1"]; dr += g["draws"]; mv += g["mean_moves"] * g["n"]
        return (np.concatenate(S), np.concatenate(P), np.concatenate(V),
                {"games": n, "p1_win": p1 / n, "draws": dr / n, "mean_moves": mv / n, "chunks": chunks,
                 "pending": sp["col"].pend["gid"].shape[0]})

    # ---------------------------------------------------------------- data and state
    say("loading puzzles ...")
    puzzles = [(n, load(a.data_root, p)) for n, p in PUZZLES]
    puzzle_eval = [(n, load(a.data_root, p)) for n, p in PUZZLE_EVAL]
    recs = load_set(a.data_root / "expert_data" / "puzzle_eval.npz")
    puzzle_ref = pickle.load(open(a.puzzle_ref, "rb"))["params"]
    rings = {rc: Ring(a.ring, rc) for rc in sizes}

    def make_opt(lr):
        return optax.inject_hyperparams(lambda learning_rate: optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=1e-4)))(learning_rate=lr)
    optimizer = make_opt(a.lr)
    ck = a.run_dir / "latest.pkl"
    if ck.exists():
        st = pickle.load(open(ck, "rb")); params, opt_state = st["params"], st["opt_state"]
        for rc in sizes:
            f = a.run_dir / f"buffers_{rc[0]}x{rc[1]}.npz"
            if f.exists(): rings[rc].load(f)
        say(f"resumed: size {st['pos']}, iteration {st['it']} (buffers: " +
            ", ".join(f"{rc[0]}x{rc[1]} {rings[rc].n:,}" for rc in sizes) + ")")
    else:
        params = pickle.load(open(a.init, "rb"))["params"]; opt_state = optimizer.init(params)
        st = {"pos": 0, "it": 0, "total_it": 0, "promoted": None, "snap": params, "stale": 0, "hist": [], "phase": 0}
    st.setdefault("phase", 0)

    def search(rc):
        s_, c_ = SEARCH[rc][min(st["phase"], len(SEARCH[rc]) - 1)]
        return (a.sims or s_, a.considered or c_)
    ctl_seen = {}

    def control():
        f = a.run_dir / "control.json"
        try: c = json.loads(f.read_text()) if f.exists() else {}
        except Exception as e: say(f"control.json unreadable ({e}); ignored"); return {}
        for k in HOT:
            if k in c and getattr(a, k) != c[k]:
                say(f"control: {k} {getattr(a, k)} -> {c[k]}"); setattr(a, k, type(getattr(a, k))(c[k]))
        return c

    def clear_flag(flag):
        f = a.run_dir / "control.json"
        try:
            c = json.loads(f.read_text()); c.pop(flag, None); f.write_text(json.dumps(c, indent=1))
        except Exception: pass

    # ---------------------------------------------------------------- training step
    def sub_loss(p, net, b, anchor):
        logits, v = net.apply({"params": p}, b["states"], train=True)
        logp = jax.nn.log_softmax(logits)
        pol = -jnp.mean(jnp.sum(b["policy_targets"] * logp, -1))
        w = b["value_weights"]; val = jnp.sum(w * jnp.square(v - b["value_targets"])) / jnp.maximum(jnp.sum(w), 1.0)
        kl = jnp.float32(0.0)
        if anchor is not None:
            rl = jax.lax.stop_gradient(net.apply({"params": anchor}, b["states"], train=False)[0])
            rlp = jax.nn.log_softmax(rl); kl = jnp.mean(jnp.sum(jnp.exp(rlp) * (rlp - logp), -1))
        return pol, val, kl

    steps_cache = {}
    def step_fn(layout):
        if layout in steps_cache: return steps_cache[layout]
        def loss_fn(p, subs, kl_prev_c, kl_puz_c, prev, pref):
            tot = 0.0; m = {}
            for (role, rc, cnt), b in zip(layout, subs):
                anchor = None if role == "cur" else (prev if role == "old" else pref)
                pol, val, kl = sub_loss(p, net_for(rc), b, anchor)
                kc = 0.0 if role == "cur" else (kl_prev_c if role == "old" else kl_puz_c)
                frac = cnt / a.batch; tot = tot + frac * (pol + val + kc * kl)
                for k, x in (("policy", pol), ("value", val), ("kl", kl)):
                    m[f"{role}_{k}"] = m.get(f"{role}_{k}", 0.0) + x * frac
            return tot, m

        def one(carry, subs, kl_prev_c, kl_puz_c, prev, pref):
            p, o = carry
            subs = [{"states": b["states"].astype(jnp.float32), "policy_targets": b["policy_targets"].astype(jnp.float32),
                     "value_targets": b["value_targets"], "value_weights": b["value_weights"]} for b in subs]
            (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, subs, kl_prev_c, kl_puz_c, prev, pref)
            u, o = optimizer.update(g, o, p); return (optax.apply_updates(p, u), o), m

        @jax.jit
        def f(p, o, subs_stack, kl_prev_c, kl_puz_c, prev, pref):
            (p, o), ms = jax.lax.scan(lambda c, b: one(c, b, kl_prev_c, kl_puz_c, prev, pref), (p, o), subs_stack)
            return p, o, jax.tree_util.tree_map(jnp.mean, ms)
        steps_cache[layout] = f; return f

    fwd_cache = {}
    def top1(p, rc, d):
        if rc not in fwd_cache:
            net = net_for(rc); fwd_cache[rc] = jax.jit(lambda q, x: net.apply({"params": q}, x, train=False))
        hit = 0
        for i in range(0, len(d[0]), 1024):
            lg = np.array(fwd_cache[rc](p, d[0][i:i + 1024].astype(np.float32))[0])
            hit += int((d[1][i:i + 1024][np.arange(len(lg)), lg.argmax(1)] > 0).sum())
        return hit / len(d[0])

    def match(p, opp, rc, rng):
        """Score of p vs opp (win 1, draw 0.5) over 2 x match_games games, sides as the game loop assigns them."""
        n = a.match_games; tot = 0.0; wins = losses = 0
        for k in range(2):
            rng, r = jax.random.split(rng)
            traj = games_fn(rc, n, search(rc), 0.25, True)(p, r, opp)
            _, _, side_rng, _ = jax.random.split(r, 4)                     # same split as play_games_batched
            main_p1 = np.array(jax.random.uniform(side_rng, (n,)) < 0.5)
            wn = np.array(traj.winners)
            for w, m1 in zip(wn, main_p1):
                if w == 0: tot += 0.5
                elif (w == 1) == m1: tot += 1; wins += 1
                else: losses += 1
        return tot / (2 * n), wins, losses

    new_log = not (a.run_dir / "log.csv").exists()
    lf = open(a.run_dir / "log.csv", "a", newline=""); log = csv.writer(lf)
    if new_log: log.writerow(["total_it", "size", "it", "games", "examples", "mean_moves", "p1_win", "draws", "cur_policy",
                              "cur_value", "old_kl", "puz_kl", "kl_prev_c", "match_score", "win_chain", "back_chain",
                              "placements", "sec_play", "sec_train", "time", "search"])
    rng = jax.random.PRNGKey(a.seed + 7919 * st["total_it"]); nrng = np.random.default_rng(a.seed + st["total_it"])

    def checkpoint(save_buffers):
        st.update(params=params, opt_state=opt_state)
        pickle.dump(st, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").replace(ck)
        if save_buffers:
            for rc in sizes:
                if rings[rc].n: rings[rc].save(a.run_dir / f"buffers_{rc[0]}x{rc[1]}.tmp.npz"); \
                    Path(a.run_dir / f"buffers_{rc[0]}x{rc[1]}.tmp.npz").replace(a.run_dir / f"buffers_{rc[0]}x{rc[1]}.npz")

    # ---------------------------------------------------------------- main loop
    while True:
        c = control()
        if c.get("stop"): checkpoint(True); clear_flag("stop"); say("stop requested: checkpointed, exiting"); return 0
        if c.get("reload"): checkpoint(True); clear_flag("reload"); say("reload requested: checkpointed, exiting 3"); sys.exit(3)
        if c.get("pause"): time.sleep(30); continue
        if "lr" in c: opt_state.hyperparams["learning_rate"] = jnp.float32(a.lr)
        rc = sizes[st["pos"]]; last = st["pos"] == len(sizes) - 1; olds = sizes[:st["pos"]]
        t0 = time.time(); rng, r = jax.random.split(rng)
        S, P, V, gs = selfplay(rc, params, r)
        rings[rc].add(S, P, V); t_play = time.time() - t0

        n_puz = int(round(a.batch * a.share_puzzle)); n_old = int(round(a.batch * a.share_old)) if olds else 0
        n_cur = a.batch - n_puz - n_old
        layout = [("cur", rc, n_cur)]
        for k, o in enumerate(olds):
            cnt = n_old // len(olds) + (1 if k < n_old % len(olds) else 0)
            if cnt and rings[o].n: layout.append(("old", o, cnt))
        for k in range(len(puzzles)):
            cnt = n_puz // len(puzzles) + (1 if k < n_puz % len(puzzles) else 0)
            if cnt: layout.append(("puz", (21, 15), cnt))
        layout = tuple(layout); f = step_fn(layout)
        kl_prev_c = a.kl_prev * 0.5 ** (st["it"] / a.kl_prev_half) if olds else 0.0
        prev = st["promoted"] if st["promoted"] is not None else params
        n_scans = max(1, int(np.ceil(a.reuse * len(V) / n_cur / a.scan))); n_steps = n_scans * a.scan
        t1 = time.time(); acc = []; M = a.scan
        for _ in range(n_scans):
            subs = []; pi = 0
            for role, rc_, cnt in layout:
                if role == "cur": subs.append(rings[rc].sample(nrng, (M, cnt)))
                elif role == "old": subs.append(rings[rc_].sample(nrng, (M, cnt)))
                else:
                    d = puzzles[pi][1]; pi += 1; idx = nrng.integers(0, len(d[0]), (M, cnt))
                    subs.append({"states": d[0][idx], "policy_targets": d[1][idx], "value_targets": d[2][idx],
                                 "value_weights": d[3][idx]})
            params, opt_state, m = f(params, opt_state, subs, jnp.float32(kl_prev_c), jnp.float32(a.kl_puzzle), prev, puzzle_ref)
            acc.append(m)
        mm = {k: float(np.mean([float(x[k]) for x in acc])) for k in acc[0]}; t_train = time.time() - t1
        st["it"] += 1; st["total_it"] += 1
        fr = n_cur / a.batch
        row = dict(games=gs["games"], examples=len(V), mean_moves=gs["mean_moves"], p1_win=gs["p1_win"],
                   draws=gs["draws"], cur_policy=mm["cur_policy"] / fr, cur_value=mm["cur_value"] / fr,
                   old_kl=mm.get("old_kl", float("nan")), puz_kl=mm.get("puz_kl", float("nan")), kl_prev_c=kl_prev_c)
        say(f"{rc[0]}x{rc[1]} it {st['it']} (total {st['total_it']}) | {gs['games']} games ({gs['chunks']} chunks x {a.slots} slots, "
            f"{gs['pending']} positions pending), {len(V)} examples, "
            f"search {search(rc)[0]}x{search(rc)[1]} | {row['mean_moves']:.0f} moves/game, P1 {row['p1_win']:.0%} draws {row['draws']:.0%} | {n_steps} steps: policy "
            f"{row['cur_policy']:.3f} value {row['cur_value']:.3f} puzzle-KL {row['puz_kl']:.3f}"
            + (f" old-KL {row['old_kl']:.3f} (c {kl_prev_c:.2f})" if olds else "") +
            f" | play {t_play:.0f}s ({len(V) / max(t_play, 1e-9):.0f} positions/s) train {t_train:.0f}s ({n_steps / max(t_train, 1e-9):.1f} steps/s)")

        ms = wc = bc = float("nan"); pe = ""; promote = False
        if st["it"] % a.eval_every == 0 or c.get("promote_now"):
            rng, r = jax.random.split(rng)
            ms, w_, l_ = match(params, st["snap"], rc, r)
            pol = NetPolicy(net_for((21, 15)), params); ch = []
            for fam in ("win", "back"):
                res = evaluate(pol, recs, limit_per_diff=a.eval_per_diff, family=fam); tot_ = sum(v["n"] for v in res.values())
                ch.append(sum(v["chain"] * v["n"] for v in res.values()) / tot_)
            wc, bc = ch; pe = " ".join(f"{n} {top1(params, (21, 15), d):.1%}" for n, d in puzzle_eval)
            st["stale"] = st["stale"] + 1 if ms < a.promote_below else 0
            st["hist"].append({"size": rc, "it": st["it"], "match": ms}); st["snap"] = params
            say(f"  [eval] vs itself {a.eval_every} iterations ago: {ms:.1%} ({w_}W {l_}L of {2 * a.match_games}) | stale "
                f"{st['stale']}/{a.patience} | puzzles: jump chains win {wc:.1%} back {bc:.1%} | placements top-1 {pe}")
            plateau = st["it"] >= a.min_iters and st["stale"] >= a.patience
            if plateau and st["phase"] < len(SEARCH[rc]) - 1 and not c.get("promote_now"):
                st["phase"] += 1; st["stale"] = 0
                say(f"  plateau: search budget up to {search(rc)[0]} simulations x {search(rc)[1]} candidates")
            elif not last and (c.get("promote_now") or st["it"] >= a.max_iters or plateau):
                promote = True
            if c.get("promote_now"): clear_flag("promote_now")
        elif not last and st["it"] >= a.max_iters: promote = True
        log.writerow([st["total_it"], f"{rc[0]}x{rc[1]}", st["it"], *[round(row[k], 4) for k in
                     ("games", "examples", "mean_moves", "p1_win", "draws", "cur_policy", "cur_value", "old_kl", "puz_kl",
                      "kl_prev_c")], round(ms, 4), round(wc, 4), round(bc, 4), pe, round(t_play), round(t_train),
                      time.strftime("%H:%M:%S"), f"{search(rc)[0]}x{search(rc)[1]}"]); lf.flush()
        (a.run_dir / "status.json").write_text(json.dumps({"size": f"{rc[0]}x{rc[1]}", "it": st["it"],
            "total_it": st["total_it"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
            "last_match": st["hist"][-1] if st["hist"] else None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        if promote:
            pickle.dump({"params": params, "size": rc}, open(a.run_dir / f"selfplay_{rc[0]}x{rc[1]}_final.pkl", "wb"))
            say(f"  promoted after {rc[0]}x{rc[1]} ({st['it']} iterations)")
            st.update(pos=st["pos"] + 1, it=0, promoted=params, stale=0, phase=0)
        if last and st["it"] % a.milestone_every == 0:
            pickle.dump({"params": params, "size": rc}, open(a.run_dir / f"selfplay_{rc[0]}x{rc[1]}_it{st['it']}.pkl", "wb"))
        checkpoint(save_buffers=promote or st["total_it"] % 10 == 0)


if __name__ == "__main__":
    sys.exit(main() or 0)
