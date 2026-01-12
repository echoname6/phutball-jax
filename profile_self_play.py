"""
Profile batched self-play to identify bottlenecks.
Works on both CPU (laptop) and TPU (Colab/Kaggle).

Usage:
  python profile_self_play.py           # Auto-detect device, small config
  python profile_self_play.py --tpu     # TPU mode with larger batches
  python profile_self_play.py --full    # Full training config benchmark
"""

import jax
import jax.numpy as jnp
import time
import argparse
import os

print("=" * 60)
print("PHUTBALL SELF-PLAY PROFILER")
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

IS_TPU = any('tpu' in str(d).lower() for d in jax.devices())
IS_GPU = any('gpu' in str(d).lower() for d in jax.devices())
print(f"TPU detected: {IS_TPU}")
print(f"GPU detected: {IS_GPU}")
print()

from phutball_env_jax import EnvConfig
from network import create_network, init_network
from self_play_batched import (
    play_games_batched,
    make_mcts_recurrent_fn,
    batched_mcts_policy,
    batched_reset,
    make_batched_step,
    make_batched_legal_actions,
    make_batched_network_input,
)


def profile_network_throughput(batch_sizes, rows, cols, channels, blocks):
    """Benchmark raw network inference speed."""
    print("\n" + "-" * 40)
    print("NETWORK THROUGHPUT")
    print("-" * 40)

    network = create_network(rows=rows, cols=cols, num_channels=channels, num_res_blocks=blocks)
    rng = jax.random.PRNGKey(42)
    variables = init_network(rng, network, num_input_channels=6)

    @jax.jit
    def forward(x):
        return network.apply(variables, x, train=False)

    for batch in batch_sizes:
        dummy_input = jax.random.normal(rng, (batch, 6, rows, cols))

        # Warmup
        policy, value = forward(dummy_input)
        policy.block_until_ready()

        # Benchmark
        num_iters = 50
        start = time.perf_counter()
        for _ in range(num_iters):
            policy, value = forward(dummy_input)
        policy.block_until_ready()
        elapsed = time.perf_counter() - start

        samples_per_sec = (batch * num_iters) / elapsed
        print(f"  batch={batch:4d}: {samples_per_sec:,.0f} samples/sec ({elapsed/num_iters*1000:.1f}ms/batch)")


def profile_mcts_policy(batch_sizes, rows, cols, channels, blocks, num_simulations):
    """Profile MCTS policy function."""
    print("\n" + "-" * 40)
    print(f"MCTS POLICY (sims={num_simulations})")
    print("-" * 40)

    env_config = EnvConfig(rows=rows, cols=cols)
    network = create_network(rows=rows, cols=cols, num_channels=channels, num_res_blocks=blocks)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }

    recurrent_fn = make_mcts_recurrent_fn(network, env_config)

    for batch_size in batch_sizes:
        try:
            states = batched_reset(env_config, batch_size)

            # Warmup
            rng, policy_rng = jax.random.split(rng)
            actions, _, _ = batched_mcts_policy(
                params, states, policy_rng, network, env_config,
                num_simulations=num_simulations, temperature=1.0,
                recurrent_fn=recurrent_fn,
            )
            actions.block_until_ready()

            # Benchmark
            num_calls = 10
            start = time.perf_counter()
            for _ in range(num_calls):
                rng, policy_rng = jax.random.split(rng)
                actions, _, _ = batched_mcts_policy(
                    params, states, policy_rng, network, env_config,
                    num_simulations=num_simulations, temperature=1.0,
                    recurrent_fn=recurrent_fn,
                )
            actions.block_until_ready()
            elapsed = time.perf_counter() - start

            total_sims = batch_size * num_simulations * num_calls
            sims_per_sec = total_sims / elapsed
            ms_per_call = (elapsed / num_calls) * 1000

            print(f"  batch={batch_size:4d}: {sims_per_sec:,.0f} sims/sec, {ms_per_call:.0f}ms/call")

        except Exception as e:
            print(f"  batch={batch_size:4d}: FAILED - {type(e).__name__}: {e}")


def profile_full_games(rows, cols, channels, blocks, batch_size, num_simulations, max_turns):
    """Profile full game playing."""
    print("\n" + "-" * 40)
    print(f"FULL SELF-PLAY ({rows}x{cols}, batch={batch_size}, sims={num_simulations})")
    print("-" * 40)

    env_config = EnvConfig(rows=rows, cols=cols)
    network = create_network(rows=rows, cols=cols, num_channels=channels, num_res_blocks=blocks)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }

    print(f"  Network: {channels} channels, {blocks} res blocks")

    # Warmup
    print("  Warming up...")
    rng, game_rng = jax.random.split(rng)
    trajectory = play_games_batched(
        params=params, rng=game_rng, network=network, env_config=env_config,
        batch_size=min(batch_size, 16), max_turns=10, max_moves=50,
        temperature=1.0, num_simulations=min(num_simulations, 25),
    )
    _ = trajectory.winners.block_until_ready()

    # Full benchmark
    print(f"  Playing {batch_size} games...")
    rng, game_rng = jax.random.split(rng)
    start = time.perf_counter()
    trajectory = play_games_batched(
        params=params, rng=game_rng, network=network, env_config=env_config,
        batch_size=batch_size, max_turns=max_turns, max_moves=max_turns * 3,
        temperature=1.0, temp_threshold=30, temp_final=0.1,
        num_simulations=num_simulations,
    )
    _ = trajectory.winners.block_until_ready()
    elapsed = time.perf_counter() - start

    total_moves = int(trajectory.valid_mask.sum())
    total_sims = total_moves * num_simulations

    print(f"\n  Results:")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Games: {batch_size}, Total moves: {total_moves}")
    print(f"    Moves/game: {total_moves/batch_size:.1f}")
    print(f"    Games/sec: {batch_size/elapsed:.3f}")
    print(f"    Moves/sec: {total_moves/elapsed:.1f}")
    print(f"    MCTS sims/sec: {total_sims/elapsed:,.0f}")

    winners = trajectory.winners
    p1_wins = int((winners == 1).sum())
    p2_wins = int((winners == 2).sum())
    draws = int((winners == 0).sum())
    print(f"    W1/W2/D: {p1_wins}/{p2_wins}/{draws}")

    return total_sims / elapsed  # Return sims/sec for extrapolation


def run_cpu_profile():
    """Small config for CPU/laptop."""
    print("\n" + "=" * 60)
    print("CPU MODE (small batches for laptop)")
    print("=" * 60)

    rows, cols = 11, 9
    channels, blocks = 32, 2

    profile_network_throughput([4, 8, 16, 32], rows, cols, channels, blocks)
    profile_mcts_policy([4, 8, 16], rows, cols, channels, blocks, num_simulations=25)
    sims_per_sec = profile_full_games(rows, cols, channels, blocks,
                                       batch_size=4, num_simulations=25, max_turns=50)

    # Extrapolation
    print("\n" + "-" * 40)
    print("EXTRAPOLATION TO TRAINING CONFIG")
    print("-" * 40)
    train_sims = 128 * 170 * 100  # batch * moves * sims
    estimated_time = train_sims / sims_per_sec
    print(f"  Training config: 128 batch, 100 sims, ~170 moves/game")
    print(f"  Estimated time at current rate: {estimated_time:.0f}s ({estimated_time/60:.1f}min)")


def run_tpu_profile():
    """Larger config for TPU."""
    print("\n" + "=" * 60)
    print("TPU MODE (larger batches)")
    print("=" * 60)

    # Test network throughput at different scales
    print("\n--- Small network (11x9) ---")
    profile_network_throughput([32, 64, 128, 256, 512], 11, 9, 64, 4)

    print("\n--- Full network (11x9, 128ch, 10blk) ---")
    profile_network_throughput([32, 64, 128, 256], 11, 9, 128, 10)

    print("\n--- Full board (21x15, 128ch, 10blk) ---")
    profile_network_throughput([32, 64, 128], 21, 15, 128, 10)

    # MCTS scaling with Gumbel-optimal sim counts
    print("\n--- MCTS Scaling (11x9, 16 sims) ---")
    profile_mcts_policy([64, 128, 256, 512, 1024], 11, 9, 64, 4, num_simulations=16)

    print("\n--- MCTS Scaling (11x9, 32 sims) ---")
    profile_mcts_policy([64, 128, 256, 512], 11, 9, 64, 4, num_simulations=32)

    # Full games with Gumbel-optimal settings
    print("\n--- Full Games ---")
    sims_per_sec_16 = profile_full_games(11, 9, 64, 4,
                                          batch_size=256, num_simulations=16, max_turns=100)

    sims_per_sec_32 = profile_full_games(11, 9, 64, 4,
                                          batch_size=512, num_simulations=32, max_turns=100)

    sims_per_sec_full = profile_full_games(11, 9, 128, 10,
                                            batch_size=256, num_simulations=32, max_turns=150)

    # Extrapolation
    print("\n" + "-" * 40)
    print("EXTRAPOLATION TO TRAINING")
    print("-" * 40)
    train_moves = 256 * 170  # batch * moves/game
    print(f"  Config: 256 batch, 32 sims, ~170 moves/game")
    print(f"  At {sims_per_sec_full:,.0f} sims/sec: {train_moves * 32 / sims_per_sec_full:.0f}s per iteration")


def profile_game_logic_vs_nn(env_config, network, params, batch_size=256):
    """Time game logic vs NN separately."""
    states = batched_reset(env_config, batch_size)
    rng = jax.random.PRNGKey(0)
    
    # Dummy actions
    actions = jnp.zeros(batch_size, dtype=jnp.int32)
    
    batched_step_fn = make_batched_step(env_config)
    batched_legal_fn = make_batched_legal_actions(env_config)
    batched_input_fn = make_batched_network_input(env_config)
    
    variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
    
    @jax.jit
    def nn_forward(inputs):
        return network.apply(variables, inputs, train=False)
    
    # Warmup
    _ = batched_step_fn(states, actions).board.block_until_ready()
    _ = batched_legal_fn(states).block_until_ready()
    inputs = batched_input_fn(states)
    _ = nn_forward(inputs)[0].block_until_ready()
    
    import time
    N = 100
    
    # Time step
    t0 = time.perf_counter()
    for _ in range(N):
        _ = batched_step_fn(states, actions)
    _.board.block_until_ready()
    t_step = (time.perf_counter() - t0) / N
    
    # Time legal actions
    t0 = time.perf_counter()
    for _ in range(N):
        _ = batched_legal_fn(states)
    _.block_until_ready()
    t_legal = (time.perf_counter() - t0) / N
    
    # Time state conversion
    t0 = time.perf_counter()
    for _ in range(N):
        inputs = batched_input_fn(states)
    inputs.block_until_ready()
    t_convert = (time.perf_counter() - t0) / N
    
    # Time NN
    t0 = time.perf_counter()
    for _ in range(N):
        policy, value = nn_forward(inputs)
    policy.block_until_ready()
    t_nn = (time.perf_counter() - t0) / N
    
    total = t_step + t_legal + t_convert + t_nn
    
    print(f"\nPer-call timing (batch={batch_size}):")
    print(f"  step:         {t_step*1000:6.2f}ms ({100*t_step/total:5.1f}%)")
    print(f"  legal_actions:{t_legal*1000:6.2f}ms ({100*t_legal/total:5.1f}%)")
    print(f"  state_convert:{t_convert*1000:6.2f}ms ({100*t_convert/total:5.1f}%)")
    print(f"  NN forward:   {t_nn*1000:6.2f}ms ({100*t_nn/total:5.1f}%)")
    print(f"  TOTAL:        {total*1000:6.2f}ms")


def run_full_training_profile():
    """Benchmark at actual training config."""
    print("\n" + "=" * 60)
    print("FULL TRAINING CONFIG BENCHMARK")
    print("=" * 60)

    # Match the actual training config from logs
    rows, cols = 11, 9  # or 21, 15 for full
    channels, blocks = 128, 10
    batch_size = 128
    num_simulations = 100
    max_turns = 200

    sims_per_sec = profile_full_games(rows, cols, channels, blocks,
                                       batch_size, num_simulations, max_turns)

    print("\n" + "-" * 40)
    print("COMPARISON TO LOGGED PERFORMANCE")
    print("-" * 40)
    logged_time = 2400  # seconds from your logs
    logged_games = 256
    logged_moves_per_game = 170
    logged_sims = logged_games * logged_moves_per_game * 100
    logged_sims_per_sec = logged_sims / logged_time

    print(f"  Logged: {logged_sims_per_sec:,.0f} sims/sec")
    print(f"  Current: {sims_per_sec:,.0f} sims/sec")
    print(f"  Speedup: {sims_per_sec/logged_sims_per_sec:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Phutball self-play")
    parser.add_argument("--tpu", action="store_true", help="TPU mode with larger batches")
    parser.add_argument("--full", action="store_true", help="Full training config benchmark")
    args = parser.parse_args()

    if args.full:
        run_full_training_profile()
    elif args.tpu or IS_TPU:
        run_tpu_profile()
    else:
        run_cpu_profile()

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
