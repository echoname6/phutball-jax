"""PBT over GRPO (expert/grpo.py) to push the self-play final further (GPU / Colab).

Members (--members) take turns on the GPU. Each round:
  1. start pool: --pool-chunks chunks of continuous self-play (16 sims) by the current anchor; the in-progress positions
     of all slots after each chunk (with their move counts) become the pool of start states;
  2. every member runs --updates rollout+update cycles: B = slots / G start states (at least `depth` moves into
     their game), G rollouts each with a light search (`sims` x 8) sampled at temperature `tau`, H micro-actions max;
  3. fitness: a match against the anchor, both sides at --eval-sims x 16 search, --eval-games games per colour;
  4. exploit / explore: the worst member copies the best (weights, optimizer, genes) when the best is significantly
     better (one-sided z-test, z > 1.645), then perturbs the genes (x0.8 / x1.25, discrete genes step to a neighbour);
  5. ratchet: a member scoring >= --ratchet against the anchor becomes the new anchor (fitness and the KL anchor then
     measure against the stronger network); every anchor is saved as anchor_<k>.pkl.
Genes: lr, tau, sims, G, kl_final (KL to the anchor), puz_w (puzzle share weight), depth (min start move).
Live control (<run-dir>/control.json): updates, eval_games, horizon, ratchet, pause, stop, reload (exit 3).

  python -m expert.grpo_pbt --init /content/drive/MyDrive/phutball/selfplay/selfplay_21x15_final.pkl \\
      --puzzle-ref /content/data/expert_data/place_run/latest.pkl --run-dir /content/drive/MyDrive/phutball/grpo \\
      --data-root /content/data
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIMS = [8, 16, 32]; GS = [8, 16]
BOUNDS = {"lr": (1e-6, 1e-3), "tau": (0.3, 2.0), "kl_final": (0.01, 2.0), "puz_w": (0.02, 1.0), "depth": (0, 60)}
HOT = ("updates", "eval_games", "horizon", "ratchet", "eval_max_turns")


def perturb(genes, rng):
    g = dict(genes)
    for k in ("lr", "tau", "kl_final", "puz_w"):
        lo, hi = BOUNDS[k]; g[k] = float(min(hi, max(lo, g[k] * rng.choice((0.8, 1.25)))))
    g["depth"] = int(min(BOUNDS["depth"][1], max(0, g["depth"] + rng.choice((-5, 5)))))
    if rng.random() < 0.3: g["sims"] = SIMS[min(len(SIMS) - 1, max(0, SIMS.index(g["sims"]) + rng.choice((-1, 1))))]
    if rng.random() < 0.3: g["G"] = GS[min(len(GS) - 1, max(0, GS.index(g["G"]) + rng.choice((-1, 1))))]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True); ap.add_argument("--puzzle-ref", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--rows", type=int, default=21); ap.add_argument("--cols", type=int, default=15)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--members", type=int, default=4); ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=80); ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50, help="optimizer steps per rollout batch")
    ap.add_argument("--moves", type=int, default=256, help="policy-gradient (and value) moves per step")
    ap.add_argument("--puzzles", type=int, default=64, help="puzzle positions per step")
    ap.add_argument("--kl-puz", type=float, default=0.5)
    ap.add_argument("--pool-chunks", type=int, default=4); ap.add_argument("--pool-sims", type=int, default=16)
    ap.add_argument("--eval-games", type=int, default=64); ap.add_argument("--eval-sims", type=int, default=64)
    ap.add_argument("--eval-max-turns", type=int, default=360, help="fitness games past this many turns are draws")
    ap.add_argument("--ratchet", type=float, default=0.6); ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(a.run_dir / "train.log", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import jax
    if not hasattr(jax.core, "get_opaque_trace_state"):   # JAX >= 0.11 with older Flax
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    import jax.numpy as jnp
    import optax
    from network import create_transformer_network
    from phutball_env_jax import EnvConfig
    from self_play_batched import make_transformer_recurrent_fn, play_games_batched, transformer_mcts_policy
    from expert.grpo import make_rollout, make_update
    from expert.selfplay_fast import make_selfplay
    from expert.selfplay_ladder import PUZZLES, load
    say(f"devices: {jax.devices()} | jax {jax.__version__}")

    rc = (a.rows, a.cols); cfg = EnvConfig(rows=a.rows, cols=a.cols); T = 400 if rc == (21, 15) else 300
    net = create_transformer_network(rows=a.rows, cols=a.cols, d_model=a.width, n_layers=a.layers, n_heads=4,
                                     ffn_dim=a.width * 2, pos_encoding="goal_distance")
    puzzles = [load(a.data_root, p) for _, p in PUZZLES]
    pref = pickle.load(open(a.puzzle_ref, "rb"))["params"]

    def make_opt():
        return optax.inject_hyperparams(lambda learning_rate: optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=1e-4)))(learning_rate=1e-4)
    optimizer = make_opt()
    build_update = make_update(net, optimizer, K=a.steps, M=a.moves)
    rng_py = random.Random(a.seed)

    ck = a.run_dir / "state.pkl"
    if ck.exists():
        S_ = pickle.load(open(ck, "rb")); say(f"resumed at round {S_['round']} ({len(S_['members'])} members, anchor #{S_['anchor_k']})")
    else:
        init = pickle.load(open(a.init, "rb"))["params"]
        members = []
        for i in range(a.members):
            g = {"lr": 1e-4 * rng_py.choice((0.5, 1.0, 2.0)), "tau": rng_py.choice((0.8, 1.0, 1.2)),
                 "sims": SIMS[i % len(SIMS)], "G": GS[i % len(GS)], "kl_final": rng_py.choice((0.1, 0.2, 0.4)),
                 "puz_w": rng_py.choice((0.1, 0.2, 0.3)), "depth": rng_py.choice((0, 10, 20))}
            members.append({"id": i, "params": init, "opt": optimizer.init(init), "genes": g, "fit": [], "lineage": [i]})
        S_ = {"round": 0, "members": members, "anchor": init, "anchor_k": 0, "history": []}
        pickle.dump({"params": init}, open(a.run_dir / "anchor_0.pkl", "wb"))

    def control():
        f = a.run_dir / "control.json"
        try: c = json.loads(f.read_text()) if f.exists() else {}
        except Exception as e: say(f"control.json unreadable ({e})"); return {}
        for k in HOT:
            if k in c and getattr(a, k) != c[k]: say(f"control: {k} {getattr(a, k)} -> {c[k]}"); setattr(a, k, type(getattr(a, k))(c[k]))
        return c

    def clear_flag(flag):
        f = a.run_dir / "control.json"
        try: c = json.loads(f.read_text()); c.pop(flag, None); f.write_text(json.dumps(c, indent=1))
        except Exception: pass

    def save():
        pickle.dump(S_, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").replace(ck)

    # ------------------------------------------------------------------ cached compiled pieces
    rolls, updates, pools = {}, {}, {}
    def rollout_fn(G, sims):
        k = (G, sims, a.horizon)
        if k not in rolls: rolls[k] = make_rollout(net, cfg, a.slots // G, G, a.horizon, sims, 8)
        return rolls[k]

    def update_fn(G):
        if G not in updates: updates[G] = build_update(G)
        return updates[G]

    match_fns = {}
    def match(p, opp, rng):
        n = a.eval_games; Te = min(T, a.eval_max_turns); key = (n, Te)
        if key not in match_fns:
            kw = dict(network=net, env_config=cfg, batch_size=n, max_turns=Te, max_moves=2 * Te, temperature=0.25,
                      temp_threshold=a.rows, temp_final=0.25, num_simulations=a.eval_sims, max_num_considered_actions=16,
                      random_opponent_ratio=0.0, mcts_policy_fn=transformer_mcts_policy,
                      recurrent_fn=make_transformer_recurrent_fn(net, cfg))
            match_fns[key] = jax.jit(lambda q, r, o: play_games_batched({"network_params": q}, r,
                                                                          opponent_params={"network_params": o}, opponent_ratio=1.0, **kw))
        tot = 0.0; w = l = 0
        for _ in range(2):
            rng, r = jax.random.split(rng)
            traj = match_fns[key](p, r, opp)
            _, _, side_rng, _ = jax.random.split(r, 4)
            main_p1 = np.array(jax.random.uniform(side_rng, (n,)) < 0.5)
            for win, m1 in zip(np.array(traj.winners), main_p1):
                if win == 0: tot += 0.5
                elif (win == 1) == m1: tot += 1; w += 1
                else: l += 1
        return tot / (2 * n), w, l

    def start_pool(params, rng):
        """In-progress positions from continuous self-play by `params` (16 sims): all slots after each chunk."""
        if "fn" not in pools:
            pools["fn"] = make_selfplay(net, cfg, a.slots, 32, a.pool_sims, 8, 1.0, 0.3, a.rows, T)
        init, run = pools["fn"]; carry = init(); states, moves = [], []
        for i in range(a.pool_chunks + 1):
            rng, r = jax.random.split(rng); carry, _ = run(params, carry, r)
            if i > 0:                                                  # skip the first chunk (all games young)
                states.append(jax.tree_util.tree_map(np.asarray, carry["states"])); moves.append(np.asarray(carry["move"]))
        st = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *states)
        return st, np.concatenate(moves)

    def puzzle_batch(K, m):
        per = [m // len(puzzles) + (1 if i < m % len(puzzles) else 0) for i in range(len(puzzles))]
        parts = {"states": [], "policy_targets": [], "value_targets": [], "value_weights": []}
        for d, c in zip(puzzles, per):
            idx = np.random.randint(0, len(d[0]), (K, c))
            for key, arr in zip(("states", "policy_targets", "value_targets", "value_weights"), d): parts[key].append(arr[idx])
        return {k: np.concatenate(v, axis=1) for k, v in parts.items()}

    new_log = not (a.run_dir / "log.csv").exists()
    lf = open(a.run_dir / "log.csv", "a", newline=""); log = csv.writer(lf)
    if new_log: log.writerow(["round", "member", "lineage", "lr", "tau", "sims", "G", "kl_final", "puz_w", "depth", "pg", "value",
                              "kl_final_val", "puzzle_ce", "entropy", "informative", "finished", "fitness", "anchor_k", "sec", "time"])
    rng = jax.random.PRNGKey(a.seed + 7919 * S_["round"])

    while S_["round"] < a.rounds:
        c = control()
        if c.get("stop"): save(); clear_flag("stop"); say("stop requested: saved, exiting"); return 0
        if c.get("reload"): save(); clear_flag("reload"); say("reload requested: saved, exiting 3"); sys.exit(3)
        if c.get("pause"): time.sleep(30); continue
        S_["round"] += 1; R_ = S_["round"]; t_round = time.time()
        rng, r = jax.random.split(rng); pool_st, pool_mv = start_pool(S_["anchor"], r)
        say(f"=== round {R_} | anchor #{S_['anchor_k']} | start pool {len(pool_mv):,} positions (moves in game: median "
            f"{int(np.median(pool_mv))}, max {int(pool_mv.max())})")
        for mem in S_["members"]:
            g = mem["genes"]; t0 = time.time(); G = g["G"]; B = a.slots // G
            mem["opt"].hyperparams["learning_rate"] = jnp.float32(g["lr"])
            roll, upd = rollout_fn(G, g["sims"]), update_fn(G); stats_acc = []; fin_acc = []
            for _ in range(a.updates):
                ok = np.flatnonzero(pool_mv >= g["depth"]); ok = ok if len(ok) >= B else np.arange(len(pool_mv))
                pick = np.random.choice(ok, B, replace=len(ok) < B)
                starts = jax.tree_util.tree_map(lambda x: jnp.asarray(x[pick]), pool_st)
                rng, r1, r2 = jax.random.split(rng, 3)
                recs, reward, done = roll(mem["params"], starts, r1, jnp.float32(g["tau"]))
                mem["params"], mem["opt"], st = upd(mem["params"], mem["opt"], S_["anchor"], pref, recs, reward, r2,
                                                    puzzle_batch(a.steps, a.puzzles), jnp.float32(g["kl_final"]),
                                                    jnp.float32(a.kl_puz), jnp.float32(g["puz_w"]))
                stats_acc.append({k: float(v) for k, v in st.items()}); fin_acc.append(float(np.asarray(done).mean()))
            ms = {k: float(np.mean([x[k] for x in stats_acc])) for k in stats_acc[0]}; fin = float(np.mean(fin_acc))
            rng, r = jax.random.split(rng); fit, w, l = match(mem["params"], S_["anchor"], r)
            mem["fit"].append(fit); mem["last"] = (fit, w, l)
            say(f"  member {mem['id']} (lineage {'>'.join(map(str, mem['lineage'][-4:]))}) | lr {g['lr']:.1e} tau {g['tau']:.2f} "
                f"search {g['sims']}x8 G {G} kl {g['kl_final']:.2f} puz {g['puz_w']:.2f} depth {g['depth']} | pg {ms['pg']:+.3f} "
                f"value {ms['value']:.3f} kl-anchor {ms['kl_final']:.3f} entropy {ms['entropy']:.2f} | informative groups "
                f"{ms['informative_groups']:.0%}, finished {fin:.0%} | vs anchor #{S_['anchor_k']}: {fit:.1%} ({w}W {l}L "
                f"{2 * a.eval_games - w - l}D) | {time.time() - t0:.0f}s")
            log.writerow([R_, mem["id"], ">".join(map(str, mem["lineage"])), g["lr"], g["tau"], g["sims"], G, g["kl_final"],
                          g["puz_w"], g["depth"], round(ms["pg"], 4), round(ms["value"], 4), round(ms["kl_final"], 4),
                          round(ms["puzzle_ce"], 4), round(ms["entropy"], 3), round(ms["informative_groups"], 3), round(fin, 3),
                          round(fit, 4), S_["anchor_k"], round(time.time() - t0), time.strftime("%H:%M:%S")]); lf.flush()
            save()
        # ratchet: a member clearly better than the anchor becomes the anchor
        ranked = sorted(S_["members"], key=lambda m: -m["last"][0]); top, bot = ranked[0], ranked[-1]
        n = 2 * a.eval_games
        if top["last"][0] >= a.ratchet:
            S_["anchor"] = top["params"]; S_["anchor_k"] += 1
            pickle.dump({"params": top["params"], "round": R_, "member": top["id"], "fitness": top["last"][0]},
                        open(a.run_dir / f"anchor_{S_['anchor_k']}.pkl", "wb"))
            say(f"  RATCHET: member {top['id']} ({top['last'][0]:.1%}) becomes anchor #{S_['anchor_k']}")
        # exploit / explore
        p1, p2 = top["last"][0], bot["last"][0]; pp = (p1 + p2) / 2
        z = (p1 - p2) / math.sqrt(max(pp * (1 - pp), 1e-6) * 2 / n)
        if z > 1.645 and top is not bot:
            bot["params"], bot["opt"] = top["params"], copy.deepcopy(top["opt"])
            bot["genes"] = perturb(top["genes"], rng_py); bot["lineage"] = top["lineage"] + [bot["id"]]
            say(f"  exploit: member {bot['id']} ({p2:.1%}) copies member {top['id']} ({p1:.1%}), z = {z:.2f}; new genes {bot['genes']}")
        else:
            say(f"  no exploit (best {p1:.1%} vs worst {p2:.1%}, z = {z:.2f})")
        S_["history"].append({"round": R_, "fitness": {m["id"]: m["last"][0] for m in S_["members"]}, "anchor_k": S_["anchor_k"]})
        (a.run_dir / "status.json").write_text(json.dumps({"round": R_, "anchor_k": S_["anchor_k"],
            "fitness": {m["id"]: round(m["last"][0], 4) for m in S_["members"]},
            "genes": {m["id"]: m["genes"] for m in S_["members"]}, "round_minutes": round((time.time() - t_round) / 60, 1),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=float))
        pickle.dump({"params": top["params"], "round": R_, "member": top["id"], "fitness": top["last"][0], "anchor_k": S_["anchor_k"]},
                    open(a.run_dir / "best_latest.pkl", "wb"))
        save()
        say(f"  round {R_} done in {(time.time() - t_round) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
