"""Curriculum pretraining of the repo's transformer on engine-relabelled winning-jump puzzles (laptop, CPU).

Stages widen the jump range: stage k trains on puzzles with intended jump count 1..k (half of each batch from the
newest jump count, the rest spread over the earlier ones so they keep being practised), for --steps[k] steps.
Every --eval-every steps: held-out first-move accuracy and full-chain success by TRUE difficulty
(expert/puzzle_eval.py), a log line (expert_data/curriculum/log.csv) and a checkpoint. Resumable.

Data pools are generated up front, in parallel subprocesses (JAX + multiprocessing in one process hangs on macOS):
expert_data/pools/J{j}_{chunk}.npz via expert/puzzle_data.py --sample --jumps j.

  python -m expert.train_curriculum --steps 600,600,800,1000 --batch 64
"""
from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "expert_data"; POOLS = DATA / "pools"; RUN = DATA / "curriculum"


def build_pools(puzzles_per_j: int, chunk: int, max_j: int, procs: int):
    POOLS.mkdir(parents=True, exist_ok=True); jobs = []
    for j in range(1, max_j + 1):
        for k in range(puzzles_per_j // chunk):
            out = POOLS / f"J{j}_{k:02d}.npz"
            if not out.exists():
                jobs.append([sys.executable, "-m", "expert.puzzle_data", "--sample", str(chunk), "--jumps", str(j),
                             "--seed", str(1000 * j + k), "--out", str(out)])
    running = []; t0 = time.time()
    while jobs or running:
        while jobs and len(running) < procs:
            running.append(subprocess.Popen(jobs.pop(0), cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in running[:]:
            if p.poll() is not None:
                if p.returncode != 0: raise RuntimeError(f"pool generation failed: {p.args}")
                running.remove(p)
        time.sleep(1)
    print(f"pools ready ({time.time() - t0:.0f}s)", flush=True)


def load_pool(j: int):
    S, P = [], []
    for f in sorted(POOLS.glob(f"J{j}_*.npz")):
        z = np.load(f); s = z["states"]
        assert s.max() < 127 and s.min() > -128
        S.append(s.astype(np.int8)); P.append(z["policy_targets"])      # int8 states: ~4x less memory
    return np.concatenate(S), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="600,600,800,1000", help="training steps per stage (stage k = jumps 1..k)")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--puzzles-per-j", type=int, default=16000); ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=7); ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-per-diff", type=int, default=60); ap.add_argument("--newest-share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, default=None); ap.add_argument("--pool-dir", type=Path, default=None)
    a = ap.parse_args(); steps = [int(s) for s in a.steps.split(",")]
    global RUN, POOLS
    if a.run_dir: RUN = a.run_dir
    if a.pool_dir: POOLS = a.pool_dir
    RUN.mkdir(parents=True, exist_ok=True)
    build_pools(a.puzzles_per_j, a.chunk, len(steps), a.procs)

    import jax
    import tempfile
    from train_batched import TrainConfig, TransformerTrainer
    from expert.net_policy import NetPolicy
    from expert.puzzle_eval import evaluate, load_set
    cfg = TrainConfig(rows=21, cols=15, num_channels=a.width, num_res_blocks=a.layers, use_wandb=False, learning_rate=a.lr,
                      checkpoint_dir=tempfile.mkdtemp(), batch_size_games=4, games_per_iteration=4, num_simulations=4,
                      train_steps_per_iteration=1, num_iterations=1)
    tr = TransformerTrainer(cfg)
    print(f"model: {a.layers} layers x width {a.width}, {sum(x.size for x in jax.tree_util.tree_leaves(tr.params)):,} parameters", flush=True)
    step = 0; rng = jax.random.PRNGKey(a.seed); nrng = np.random.default_rng(a.seed)
    ck = RUN / "latest.pkl"
    if ck.exists():
        d = pickle.load(open(ck, "rb")); tr.params, tr.opt_state, step = d["params"], d["opt_state"], d["step"]
        print(f"resumed at step {step}", flush=True)
    pools = {j: load_pool(j) for j in range(1, len(steps) + 1)}
    print("examples per jump count: " + ", ".join(f"J{j} {len(p[1]):,}" for j, p in pools.items()), flush=True)
    recs = load_set(); new_log = not (RUN / "log.csv").exists()
    lf = open(RUN / "log.csv", "a", newline=""); log = csv.writer(lf)
    if new_log: log.writerow(["step", "stage", "policy_loss", "value_loss", "entropy", "first", "chain", "by_difficulty",
                              "back_first", "back_chain", "back_by_difficulty", "minutes"])
    bounds = np.cumsum(steps); t0 = time.time(); acc = []

    def batch(stage):
        k = stage + 1; n_new = int(a.batch * (a.newest_share if k > 1 else 1.0)); parts = [(k, n_new)]
        rest = a.batch - n_new
        for j in range(1, k):
            parts.append((j, rest // (k - 1) + (1 if j <= rest % (k - 1) else 0)))
        S, P = [], []
        for j, m in parts:
            if m <= 0: continue
            idx = nrng.integers(0, len(pools[j][1]), m); S.append(pools[j][0][idx]); P.append(pools[j][1][idx])
        S = np.concatenate(S).astype(np.float32); P = np.concatenate(P)
        return {"states": S, "policy_targets": P, "value_targets": np.ones(len(P), np.float32)}

    while step < bounds[-1]:
        stage = int(np.searchsorted(bounds, step, side="right"))
        rng, k = jax.random.split(rng)
        tr.params, tr.opt_state, m = tr.train_step_fn(tr.params, tr.opt_state, batch(stage), k)
        step += 1; acc.append({k_: float(v) for k_, v in m.items() if k_ in ("policy_loss", "value_loss", "policy_entropy")})
        if step % 20 == 0:
            mm = {k_: np.mean([x[k_] for x in acc]) for k_ in acc[0]}; acc = []
            print(f"step {step:5d} stage {stage + 1} (jumps 1..{stage + 1}) | policy loss {mm['policy_loss']:.3f} value loss "
                  f"{mm['value_loss']:.3f} entropy {mm['policy_entropy']:.2f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == bounds[-1] or step in bounds:
            pol = NetPolicy(tr.network, tr.params); out = []
            for fam in ("win", "back"):                      # forward-built wins vs required-backward traps
                res = evaluate(pol, recs, limit_per_diff=a.eval_per_diff, family=fam)
                tot = sum(v["n"] for v in res.values())
                f = sum(v["first"] * v["n"] for v in res.values()) / tot; c = sum(v["chain"] * v["n"] for v in res.values()) / tot
                byd = " ".join(f"{d}j:{v['first']:.2f}/{v['chain']:.2f}" for d, v in res.items()); out += [round(f, 4), round(c, 4), byd]
                print(f"  [eval step {step}] {fam:4s}: first-move {f:.1%}, chain {c:.1%} | by difficulty (first/chain) {byd}", flush=True)
            log.writerow([step, stage + 1, mm.get("policy_loss", ""), mm.get("value_loss", ""), mm.get("policy_entropy", ""),
                          *out, round((time.time() - t0) / 60, 1)]); lf.flush()
            d = {"params": tr.params, "opt_state": tr.opt_state, "step": step, "config": vars(a)}
            pickle.dump(d, open(str(ck) + ".tmp", "wb")); Path(str(ck) + ".tmp").replace(ck)
            if step in bounds: pickle.dump(d, open(RUN / f"stage{stage + 1}_end.pkl", "wb"))
    print("curriculum done", flush=True)


if __name__ == "__main__":
    main()
