"""Board-size ladder on expert games (laptop, CPU): 13x9 -> 15x11 -> 17x13 -> 21x15, one set of weights for all sizes.

The transformer has no size-specific parameters (per-cell tokens, computed goal-distance position encoding, per-cell
heads, mean pooling), so one parameter set is applied through a network built for each board size.

Every optimiser step sums losses over FIXED-SIZE sub-batches (fixed counts, so each rung compiles once):
  current rung games   policy CE on the expert's move + value MSE on the game result
  earlier rungs' games same, plus kl_prev(t) * KL(promoted || model): the snapshot frozen at the last promotion,
                       with a coefficient that starts at --kl-prev and halves every --kl-prev-half steps (scheduled),
                       so a size change cannot wreck what was learned, but old habits can be revised later
  puzzles (21x15)      policy CE on the proven targets, value only where known (value_weights),
                       plus --kl-puzzle * KL(puzzle model || model): the tactical anchor, on puzzle states only

Promotion: every --eval-every steps, held-out top-1 agreement with the expert on the current rung (the last game chunk
of each rung is held out). Promote when it has not improved by --min-gain for --patience evaluations in a row, or when
--max-passes passes over the rung's games are done (whichever comes first; at least --min-passes).

  python -m expert.train_ladder --init expert_data/place_run/latest.pkl --run-dir expert_data/ladder
"""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
G = "expert_data/games"
RUNGS = [((13, 9), f"{G}/13x9/chunk_[0-3].npz", f"{G}/13x9/chunk_4.npz"),
         ((15, 11), f"{G}/15x11/chunk_[0-3].npz", f"{G}/15x11/chunk_4.npz"),
         ((17, 13), f"{G}/17x13/chunk_[0-3].npz", f"{G}/17x13/chunk_4.npz"),
         ((21, 15), f"{G}/chunk_[0-8].npz", f"{G}/chunk_9.npz")]
PUZZLES = [("jumps", "expert_data/pools/J*_*.npz"), ("forced", "expert_data/place_pools/forced_*.npz"),
           ("block", "expert_data/place_pools/block_*.npz"), ("prevent", "expert_data/place_pools/prevent_*.npz")]
PUZZLE_EVAL = [("forced", "expert_data/place_pools/heldout/forced.npz"), ("block", "expert_data/place_pools/heldout/block.npz"),
               ("prevent", "expert_data/place_pools/heldout/prevent.npz")]


DATA_ROOT = ROOT


def load(pattern: str, stride: int = 1):
    files = sorted(glob.glob(str(DATA_ROOT / pattern)))
    if not files: raise FileNotFoundError(pattern)
    S, P, V, W = [], [], [], []
    for f in files:
        z = np.load(f); sl = slice(None, None, stride)
        S.append(z["states"][sl].astype(np.int8)); P.append(z["policy_targets"][sl].astype(np.float32))
        V.append(z["value_targets"][sl].astype(np.float32))
        W.append(z["value_weights"][sl].astype(np.float32) if "value_weights" in z else np.ones(len(V[-1]), np.float32))
    return [np.concatenate(x) for x in (S, P, V, W)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="puzzle-trained checkpoint (also the puzzle KL anchor)")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--share-old", type=float, default=0.15); ap.add_argument("--share-puzzle", type=float, default=0.15)
    ap.add_argument("--kl-puzzle", type=float, default=0.5)
    ap.add_argument("--kl-prev", type=float, default=1.0); ap.add_argument("--kl-prev-half", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.003)
    ap.add_argument("--min-passes", type=float, default=0.5); ap.add_argument("--max-passes", type=float, default=3.0)
    ap.add_argument("--stride-21", type=int, default=2, help="keep every k-th 21x15 position (neighbours are near-duplicates)")
    ap.add_argument("--eval-per-diff", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rungs", default="0,1,2,3", help="which rungs to run (indices into RUNGS)")
    ap.add_argument("--data-root", type=Path, default=None, help="directory holding expert_data/ (default: the repo)")
    a = ap.parse_args()
    global DATA_ROOT
    if a.data_root: DATA_ROOT = a.data_root; a.run_dir.mkdir(parents=True, exist_ok=True)
    rung_ids = [int(x) for x in a.rungs.split(",")]

    import jax
    import jax.numpy as jnp
    import optax
    from network import create_transformer_network
    from expert.net_policy import NetPolicy
    from expert.puzzle_eval import evaluate, load_set

    nets = {}
    def net_for(rc):
        if rc not in nets:
            nets[rc] = create_transformer_network(rows=rc[0], cols=rc[1], d_model=a.width, n_layers=a.layers, n_heads=4,
                                                  ffn_dim=a.width * 2, pos_encoding="goal_distance")
        return nets[rc]

    print("loading data ...", flush=True)
    games = {}
    for i in rung_ids:
        rc, tr_pat, ho_pat = RUNGS[i]; st = a.stride_21 if rc == (21, 15) else 1
        games[i] = (load(tr_pat, st), load(ho_pat, st))
        print(f"  rung {i} {rc[0]}x{rc[1]}: {len(games[i][0][0]):,} train / {len(games[i][1][0]):,} held-out positions", flush=True)
    puzzles = [(n, load(p)) for n, p in PUZZLES]
    puzzle_eval = [(n, load(p)) for n, p in PUZZLE_EVAL]
    print("  puzzles: " + ", ".join(f"{n} {len(d[0]):,}" for n, d in puzzles), flush=True)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=1e-4))
    ck = a.run_dir / "latest.pkl"
    puzzle_ref = pickle.load(open(a.init, "rb"))["params"]
    if ck.exists():
        d = pickle.load(open(ck, "rb"))
        params, opt_state, rpos, rstep, promoted, hist = d["params"], d["opt_state"], d["rung_pos"], d["rung_step"], d["promoted"], d["hist"]
        print(f"resumed: rung position {rpos}, step {rstep} in rung", flush=True)
    else:
        params = puzzle_ref; opt_state = optimizer.init(params); rpos, rstep, promoted, hist = 0, 0, None, []

    def sub_loss(p, net, b, anchor, kl_c):
        logits, v = net.apply({"params": p}, b["states"], train=True)
        logp = jax.nn.log_softmax(logits)
        pol = -jnp.mean(jnp.sum(b["policy_targets"] * logp, -1))
        w = b["value_weights"]; val = jnp.sum(w * jnp.square(v - b["value_targets"])) / jnp.maximum(jnp.sum(w), 1.0)
        kl = jnp.float32(0.0)
        if anchor is not None:
            rl = jax.lax.stop_gradient(net.apply({"params": anchor}, b["states"], train=False)[0])
            rlp = jax.nn.log_softmax(rl); kl = jnp.mean(jnp.sum(jnp.exp(rlp) * (rlp - logp), -1))
        return pol, val, kl

    def make_step(layout):
        """layout: tuple of (role, (rows, cols), count); role in cur / old / puz. Compiled once per rung."""
        def loss_fn(p, subs, kl_prev_c, prev, pref):
            tot = 0.0; m = {}
            for (role, rc, cnt), b in zip(layout, subs):
                if role == "cur": pol, val, kl = sub_loss(p, net_for(rc), b, None, 0.0); kc = 0.0
                elif role == "old": pol, val, kl = sub_loss(p, net_for(rc), b, prev, 1.0); kc = kl_prev_c
                else: pol, val, kl = sub_loss(p, net_for(rc), b, pref, 1.0); kc = a.kl_puzzle
                frac = cnt / a.batch
                tot = tot + frac * (pol + val + kc * kl)
                for k, x in (("policy", pol), ("value", val), ("kl", kl)):
                    m[f"{role}_{k}"] = m.get(f"{role}_{k}", 0.0) + x * frac
            return tot, m

        @jax.jit
        def step(p, o, subs, kl_prev_c, prev, pref):
            (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, subs, kl_prev_c, prev, pref)
            u, o = optimizer.update(g, o, p); return optax.apply_updates(p, u), o, m
        return step

    nrng = np.random.default_rng(a.seed + 7919 * len(hist))

    def draw(d, m):
        idx = nrng.integers(0, len(d[0]), m)
        return {"states": d[0][idx].astype(np.float32), "policy_targets": d[1][idx], "value_targets": d[2][idx],
                "value_weights": d[3][idx]}

    fwd_cache = {}
    def top1(p, rc, d):
        if rc not in fwd_cache:
            net = net_for(rc); fwd_cache[rc] = jax.jit(lambda q, x: net.apply({"params": q}, x, train=False))
        f = fwd_cache[rc]; hit = 0; se = 0.0; wsum = 0.0
        for i in range(0, len(d[0]), 512):
            lg, v = f(p, d[0][i:i + 512].astype(np.float32)); lg = np.array(lg); v = np.array(v)
            hit += int((d[1][i:i + 512][np.arange(len(lg)), lg.argmax(1)] > 0).sum())
            se += float((d[3][i:i + 512] * (v - d[2][i:i + 512]) ** 2).sum()); wsum += float(d[3][i:i + 512].sum())
        return hit / len(d[0]), (se / wsum if wsum else float("nan"))

    recs = load_set(); new_log = not (a.run_dir / "log.csv").exists()
    lf = open(a.run_dir / "log.csv", "a", newline=""); log = csv.writer(lf)
    if new_log: log.writerow(["rung", "size", "rung_step", "passes", "kl_prev_c", "cur_policy", "cur_value", "old_kl", "puz_kl",
                              "heldout", "win_chain", "back_chain", "placement", "minutes"])
    t0 = time.time()

    while rpos < len(rung_ids):
        ri = rung_ids[rpos]; rc = RUNGS[ri][0]; train_d, ho_d = games[ri]
        olds = [i for i in rung_ids[:rpos]]
        n_puz = int(round(a.batch * a.share_puzzle)); n_old = int(round(a.batch * a.share_old)) if olds else 0
        n_cur = a.batch - n_puz - n_old
        layout = [("cur", rc, n_cur)]
        for k, i in enumerate(olds):                                    # old share split evenly over earlier rungs
            c = n_old // len(olds) + (1 if k < n_old % len(olds) else 0)
            if c: layout.append(("old", RUNGS[i][0], c))
        for k, (n, _) in enumerate(puzzles):
            c = n_puz // len(puzzles) + (1 if k < n_puz % len(puzzles) else 0)
            if c: layout.append(("puz", (21, 15), c))
        layout = tuple(layout); step_fn = make_step(layout)
        steps_per_pass = len(train_d[0]) / n_cur
        min_steps, max_steps = int(a.min_passes * steps_per_pass), int(a.max_passes * steps_per_pass)
        prev = promoted if promoted is not None else params
        rung_hist = [h for h in hist if h["rung"] == ri]
        best = max([h["heldout"] for h in rung_hist], default=-1.0); stale = 0
        for h in rung_hist:                                             # replay the patience counter on resume
            stale = 0 if h["gain"] >= a.min_gain else stale + 1
        print(f"=== rung {ri}: {rc[0]}x{rc[1]} | batch layout {[(r, f'{x[0]}x{x[1]}', c) for r, x, c in layout]} | "
              f"{steps_per_pass:.0f} steps/pass, {min_steps}-{max_steps} steps", flush=True)
        acc = []; mm = {}
        while True:
            kl_prev_c = a.kl_prev * 0.5 ** (rstep / a.kl_prev_half) if olds else 0.0
            subs = []; pi = 0
            for role, rc_, cnt in layout:
                if role == "cur": subs.append(draw(train_d, cnt))
                elif role == "old": subs.append(draw(games[next(i for i in olds if RUNGS[i][0] == rc_)][0], cnt))
                else: subs.append(draw(puzzles[pi][1], cnt)); pi += 1
            params, opt_state, m = step_fn(params, opt_state, subs, jnp.float32(kl_prev_c), prev, puzzle_ref)
            rstep += 1; acc.append({k: float(v) for k, v in m.items()})
            if rstep % 20 == 0:
                mm = {k: np.mean([x[k] for x in acc]) for k in acc[0]}; acc = []
                extra = " ".join(f"{k} {mm[k]:.3f}" for k in ("old_kl", "puz_kl") if k in mm)
                print(f"rung {ri} {rc[0]}x{rc[1]} step {rstep:5d} ({rstep / steps_per_pass:.2f} passes) | policy "
                      f"{mm['cur_policy'] / (n_cur / a.batch):.3f} value {mm['cur_value'] / (n_cur / a.batch):.3f} | {extra} "
                      f"kl_prev_c {kl_prev_c:.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
            if rstep % a.eval_every == 0 or rstep >= max_steps:
                if acc: mm = {k: np.mean([x[k] for x in acc]) for k in acc[0]}
                ho, vm = top1(params, rc, ho_d)
                olds_s = " ".join(f"{RUNGS[i][0][0]}x{RUNGS[i][0][1]} {top1(params, RUNGS[i][0], games[i][1])[0]:.1%}" for i in olds)
                pol = NetPolicy(net_for((21, 15)), params); chains_ = []
                for fam in ("win", "back"):
                    res = evaluate(pol, recs, limit_per_diff=a.eval_per_diff, family=fam); tot_ = sum(v["n"] for v in res.values())
                    chains_.append(sum(v["chain"] * v["n"] for v in res.values()) / tot_)
                pe = " ".join(f"{n} {top1(params, (21, 15), d)[0]:.1%}" for n, d in puzzle_eval)
                gain = ho - best; best = max(best, ho); stale = 0 if gain >= a.min_gain else stale + 1
                hist.append({"rung": ri, "step": rstep, "heldout": ho, "gain": gain})
                print(f"  [eval rung {ri} step {rstep}] held-out expert top-1 {ho:.1%} (value MSE {vm:.3f}, best {best:.1%}, "
                      f"stale {stale}/{a.patience}) | earlier sizes: {olds_s or '-'} | jump chains win {chains_[0]:.1%} "
                      f"back {chains_[1]:.1%} | placements top-1: {pe}", flush=True)
                log.writerow([ri, f"{rc[0]}x{rc[1]}", rstep, round(rstep / steps_per_pass, 2), round(kl_prev_c, 3),
                              mm.get("cur_policy", ""), mm.get("cur_value", ""), mm.get("old_kl", ""), mm.get("puz_kl", ""),
                              round(ho, 4), round(chains_[0], 4), round(chains_[1], 4), pe, round((time.time() - t0) / 60, 1)])
                lf.flush()
                done = rstep >= max_steps or (rstep >= min_steps and stale >= a.patience)
                if done:
                    pickle.dump({"params": params, "rung": ri}, open(a.run_dir / f"rung{ri}_{rc[0]}x{rc[1]}.pkl", "wb"))
                    print(f"  promoted after rung {ri} ({'plateau' if rstep < max_steps else 'max passes'})", flush=True)
                    promoted = params; rpos += 1; rstep = 0
                d = {"params": params, "opt_state": opt_state, "rung_pos": rpos, "rung_step": rstep, "promoted": promoted,
                     "hist": hist, "config": vars(a)}
                pickle.dump(d, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").replace(ck)
                if done: break
    print("ladder done", flush=True)


if __name__ == "__main__":
    main()
