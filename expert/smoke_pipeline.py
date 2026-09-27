"""End-to-end smoke test on CPU with a tiny transformer: expert data -> buffer -> train steps ->
checkpoint save/reload -> network plays a game. Run before spending GPU time."""
import glob
import sys
import tempfile
import time

sys.path.insert(0, ".")


def main():
    # Generate in a SEPARATE process: starting a multiprocessing pool after JAX is loaded in the same
    # process hangs (observed on macOS); in Colab the kernel always has JAX loaded, so do the same there.
    import subprocess
    d0 = tempfile.mkdtemp(); t0 = time.time()
    subprocess.run([sys.executable, "-m", "expert.dataset", "--games", "24", "--workers", "2", "--out", d0 + "/d.npz"], check=True)
    import numpy as np
    z = np.load(d0 + "/d.npz"); S, P, V = z["states"], z["policy_targets"], z["value_targets"]
    print(f"data: {len(V)} examples in {time.time()-t0:.0f}s, channels {S.shape[1]}", flush=True)
    import jax
    from expert.arena import RandomPolicy, play
    from expert.net_policy import NetPolicy
    from train_batched import TrainConfig, TransformerTrainer
    d = tempfile.mkdtemp()
    cfg = TrainConfig(rows=21, cols=15, num_channels=32, num_res_blocks=2, use_wandb=False, checkpoint_dir=d,
                      batch_size_games=4, games_per_iteration=4, num_simulations=4, train_steps_per_iteration=5, num_iterations=1)
    t0 = time.time(); tr = TransformerTrainer(cfg); print(f"trainer built in {time.time()-t0:.0f}s", flush=True)
    tr.replay_buffer.add(S, P, V)
    rng = jax.random.PRNGKey(0); t0 = time.time()
    for i in range(60):
        rng, k = jax.random.split(rng)
        b = tr.replay_buffer.sample(64)
        tr.params, tr.opt_state, m = tr.train_step_fn(tr.params, tr.opt_state, {k_: b[k_] for k_ in ("states", "policy_targets", "value_targets")}, k)
        if i in (0, 59):
            print(f"step {i}: " + ", ".join(f"{k_}={float(v):.3f}" for k_, v in m.items() if k_ in ("policy_loss", "value_loss", "policy_entropy")), flush=True)
    print(f"60 train steps in {time.time()-t0:.0f}s", flush=True)
    tr.save_checkpoint(); ck = glob.glob(d + "/*.pkl"); print("checkpoint:", [c.split('/')[-1] for c in ck], flush=True)
    tr2 = TransformerTrainer(cfg); tr2.load_checkpoint(ck[0]); print("reload ok; buffer", len(tr2.replay_buffer), flush=True)
    w, t = play(NetPolicy(tr.network, tr.params), RandomPolicy(1), max_turns=200)
    print(f"net vs random smoke game: winner {w} after {t} turns")


if __name__ == "__main__":
    main()
