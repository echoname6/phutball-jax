"""Mixed-source training after the jump curriculum (laptop, CPU): placement puzzles, then expert games with a KL anchor.

Each batch is drawn from named sources by share. Every source is a set of .npz pools with states (int8 or float),
policy_targets, value_targets and optionally value_weights (default 1). The value loss is weighted per example, so
puzzles with an unknown outcome (block, prevent) train only the policy.

  --ref CKPT --kl C   adds C * KL(ref || model) on every example: the frozen reference (the puzzle-trained model)
                      keeps the tactics while the expert games teach the rest of the game.

Stage 5, placements (jump puzzles kept in the mix):
  python -m expert.train_mix --init expert_data/curriculum/latest.pkl --run-dir expert_data/place_run --steps 1500 \\
      --src "jumps:expert_data/pools/J*_*.npz:0.4" --src "forced:expert_data/place_pools/forced_*.npz:0.2" \\
      --src "block:expert_data/place_pools/block_*.npz:0.2" --src "prevent:expert_data/place_pools/prevent_*.npz:0.2"
Stage 6, expert games, 2-3 passes (--passes sets the steps from the size of the named source):
  python -m expert.train_mix --init expert_data/place_run/latest.pkl --ref expert_data/place_run/latest.pkl --kl 0.5 \\
      --run-dir expert_data/games_run --passes games:2 --src "games:expert_data/games/chunk_[0-8].npz:0.75" \\
      --src "jumps:...:0.1" --src "forced:...:0.05" --src "block:...:0.05" --src "prevent:...:0.05"
"""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_source(pattern: str):
    S, P, V, W = [], [], [], []
    files = sorted(glob.glob(str(ROOT / pattern) if not pattern.startswith("/") else pattern))
    if not files: raise FileNotFoundError(pattern)
    for f in files:
        z = np.load(f); s = z["states"]
        S.append(s.astype(np.int8)); P.append(z["policy_targets"].astype(np.float32))
        V.append(z["value_targets"].astype(np.float32))
        W.append(z["value_weights"].astype(np.float32) if "value_weights" in z else np.ones(len(s), np.float32))
    return np.concatenate(S), np.concatenate(P), np.concatenate(V), np.concatenate(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="name:glob:share (repeatable)")
    ap.add_argument("--eval-src", action="append", default=[], help="name:glob held-out pools (target-support top-1, value)")
    ap.add_argument("--init", type=Path, required=True); ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None); ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--passes", default="", help="source:passes")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--eval-per-diff", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.run_dir.mkdir(parents=True, exist_ok=True)

    srcs = []
    for spec in a.src:
        name, rest = spec.split(":", 1); pat, share = rest.rsplit(":", 1)
        srcs.append((name, load_source(pat), float(share)))
    tot = sum(s[2] for s in srcs); srcs = [(n, d, sh / tot) for n, d, sh in srcs]
    print("sources: " + ", ".join(f"{n} {len(d[0]):,} ({sh:.0%})" for n, d, sh in srcs), flush=True)
    steps = a.steps
    if a.passes:
        pn, pk = a.passes.split(":"); n_src, sh = next((len(d[0]), sh) for n, d, sh in srcs if n == pn)
        steps = int(np.ceil(float(pk) * n_src / (a.batch * sh)))
    assert steps > 0, "--steps or --passes"
    evals = []
    for spec in a.eval_src:
        name, pat = spec.split(":", 1); evals.append((name, load_source(pat)))

    import jax
    import jax.numpy as jnp
    import optax
    from train_batched import TrainConfig, TransformerTrainer
    from expert.net_policy import NetPolicy
    from expert.puzzle_eval import evaluate, load_set
    cfg = TrainConfig(rows=21, cols=15, num_channels=a.width, num_res_blocks=a.layers, use_wandb=False, learning_rate=a.lr,
                      checkpoint_dir=tempfile.mkdtemp(), batch_size_games=4, games_per_iteration=4, num_simulations=4,
                      train_steps_per_iteration=1, num_iterations=1)
    tr = TransformerTrainer(cfg); net = tr.network
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(a.lr, weight_decay=1e-4))
    ck = a.run_dir / "latest.pkl"
    if ck.exists():
        d = pickle.load(open(ck, "rb")); params, opt_state, step = d["params"], d["opt_state"], d["step"]
        print(f"resumed at step {step}", flush=True)
    else:
        params = pickle.load(open(a.init, "rb"))["params"]; opt_state = optimizer.init(params); step = 0
    ref = pickle.load(open(a.ref, "rb"))["params"] if a.ref else None
    kl_c = a.kl if ref is not None else 0.0

    def loss_fn(p, batch):
        logits, v = net.apply({"params": p}, batch["states"], train=True)
        logp = jax.nn.log_softmax(logits)
        pol = -jnp.mean(jnp.sum(batch["policy_targets"] * logp, -1))
        w = batch["value_weights"]
        val = jnp.sum(w * jnp.square(v - batch["value_targets"])) / jnp.maximum(jnp.sum(w), 1.0)
        kl = 0.0
        if kl_c:
            rl = jax.lax.stop_gradient(net.apply({"params": ref}, batch["states"], train=False)[0])
            rlogp = jax.nn.log_softmax(rl); kl = jnp.mean(jnp.sum(jnp.exp(rlogp) * (rlogp - logp), -1))
        ent = -jnp.mean(jnp.sum(jnp.exp(logp) * logp, -1))
        return pol + val + kl_c * kl, {"policy_loss": pol, "value_loss": val, "kl": kl, "entropy": ent}

    @jax.jit
    def train_step(p, o, batch):
        (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(p, batch)
        u, o = optimizer.update(g, o, p); return optax.apply_updates(p, u), o, m

    fwd = jax.jit(lambda p, x: net.apply({"params": p}, x, train=False))
    nrng = np.random.default_rng(a.seed + step)

    def batch():
        counts = nrng.multinomial(a.batch, [sh for _, _, sh in srcs]); parts = [[], [], [], []]
        for (n, d, _), m in zip(srcs, counts):
            if m == 0: continue
            idx = nrng.integers(0, len(d[0]), m)
            for k in range(4): parts[k].append(d[k][idx])
        S, P, V, W = (np.concatenate(x) for x in parts)
        return {"states": S.astype(np.float32), "policy_targets": P, "value_targets": V, "value_weights": W}

    def held_out(p):
        out = {}
        for name, (S, P, V, W) in evals:
            hit, vse, vw = 0, 0.0, 0.0
            for i in range(0, len(S), 512):
                lg, v = fwd(p, S[i:i + 512].astype(np.float32)); lg = np.array(lg); v = np.array(v)
                hit += int((P[i:i + 512][np.arange(len(lg)), lg.argmax(1)] > 0).sum())
                vse += float((W[i:i + 512] * (v - V[i:i + 512]) ** 2).sum()); vw += float(W[i:i + 512].sum())
            out[name] = (hit / len(S), vse / vw if vw else float("nan"))
        return out

    recs = load_set(); new_log = not (a.run_dir / "log.csv").exists()
    lf = open(a.run_dir / "log.csv", "a", newline=""); log = csv.writer(lf)
    if new_log: log.writerow(["step", "policy_loss", "value_loss", "kl", "entropy", "win_chain", "back_chain", "held_out", "minutes"])
    t0 = time.time(); acc = []
    print(f"{steps} steps, batch {a.batch}, lr {a.lr}, kl {kl_c}", flush=True)
    while step < steps:
        params, opt_state, m = train_step(params, opt_state, batch()); step += 1
        acc.append({k: float(v) for k, v in m.items()})
        if step % 20 == 0:
            mm = {k: np.mean([x[k] for x in acc]) for k in acc[0]}; acc = []
            print(f"step {step:5d}/{steps} | policy {mm['policy_loss']:.3f} value {mm['value_loss']:.3f} kl {mm['kl']:.3f} "
                  f"entropy {mm['entropy']:.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == steps:
            pol = NetPolicy(net, params); chains_ = []
            for fam in ("win", "back"):
                res = evaluate(pol, recs, limit_per_diff=a.eval_per_diff, family=fam); tot_ = sum(v["n"] for v in res.values())
                chains_.append(sum(v["chain"] * v["n"] for v in res.values()) / tot_)
            ho = held_out(params)
            hs = " ".join(f"{k}: top1 {v[0]:.1%}" + (f" vMSE {v[1]:.3f}" if v[1] == v[1] else "") for k, v in ho.items())
            print(f"  [eval step {step}] jump chains win {chains_[0]:.1%} back {chains_[1]:.1%} | {hs}", flush=True)
            log.writerow([step, mm["policy_loss"], mm["value_loss"], mm["kl"], mm["entropy"], round(chains_[0], 4),
                          round(chains_[1], 4), hs, round((time.time() - t0) / 60, 1)]); lf.flush()
            d = {"params": params, "opt_state": opt_state, "step": step, "config": vars(a)}
            pickle.dump(d, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").replace(ck)
    print("done", flush=True)


if __name__ == "__main__":
    main()
