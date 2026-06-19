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
from network import create_network, init_network, create_transformer_network, init_transformer_network
from self_play_batched import (
    play_games_batched,
    make_mcts_recurrent_fn,
    batched_mcts_policy,
    batched_reset,
    make_batched_step,
    make_batched_legal_actions,
    make_batched_network_input,
    transformer_mcts_policy,
    make_transformer_recurrent_fn,
)


def _create_network_and_params(rows, cols, channels, blocks, use_transformer=False,
                               d_model=128, n_layers=4, n_heads=4, ffn_dim=256):
    """Helper to create network + params dict for either CNN or Transformer."""
    rng = jax.random.PRNGKey(42)
    if use_transformer:
        network = create_transformer_network(
            rows=rows, cols=cols, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim,
        )
        variables = init_transformer_network(rng, network, num_input_channels=10)
        params = {'network_params': variables['params']}
    else:
        network = create_network(rows=rows, cols=cols, num_channels=channels, num_res_blocks=blocks)
        variables = init_network(rng, network, num_input_channels=10)
        params = {
            'network_params': variables['params'],
            'batch_stats': variables['batch_stats'],
        }
    return network, variables, params


def profile_network_throughput(batch_sizes, rows, cols, channels, blocks,
                               use_transformer=False, transformer_kwargs=None):
    """Benchmark raw network inference speed."""
    tkw = transformer_kwargs or {}
    label = "TRANSFORMER" if use_transformer else "CNN"
    print("\n" + "-" * 40)
    print(f"NETWORK THROUGHPUT ({label})")
    print("-" * 40)

    network, variables, _ = _create_network_and_params(
        rows, cols, channels, blocks, use_transformer=use_transformer, **tkw)

    if use_transformer:
        apply_vars = {'params': variables['params']}
    else:
        apply_vars = variables

    @jax.jit
    def forward(x):
        return network.apply(apply_vars, x, train=False)

    rng = jax.random.PRNGKey(42)
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


def profile_mcts_policy(batch_sizes, rows, cols, channels, blocks, num_simulations,
                        use_transformer=False, transformer_kwargs=None):
    """Profile MCTS policy function."""
    tkw = transformer_kwargs or {}
    label = "Transformer" if use_transformer else "CNN"
    print("\n" + "-" * 40)
    print(f"MCTS POLICY ({label}, sims={num_simulations})")
    print("-" * 40)

    env_config = EnvConfig(rows=rows, cols=cols)
    network, _, params = _create_network_and_params(
        rows, cols, channels, blocks, use_transformer=use_transformer, **tkw)

    if use_transformer:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)
        policy_fn = transformer_mcts_policy
    else:
        recurrent_fn = make_mcts_recurrent_fn(network, env_config)
        policy_fn = batched_mcts_policy

    rng = jax.random.PRNGKey(42)

    for batch_size in batch_sizes:
        try:
            states = batched_reset(env_config, batch_size)

            # Warmup
            rng, policy_rng = jax.random.split(rng)
            actions, _, _ = policy_fn(
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
                actions, _, _ = policy_fn(
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


def profile_full_games(rows, cols, channels, blocks, batch_size, num_simulations, max_turns,
                       use_transformer=False, transformer_kwargs=None):
    """Profile full game playing."""
    tkw = transformer_kwargs or {}
    label = "Transformer" if use_transformer else "CNN"
    print("\n" + "-" * 40)
    print(f"FULL SELF-PLAY ({label}, {rows}x{cols}, batch={batch_size}, sims={num_simulations})")
    print("-" * 40)

    env_config = EnvConfig(rows=rows, cols=cols)
    network, _, params = _create_network_and_params(
        rows, cols, channels, blocks, use_transformer=use_transformer, **tkw)

    if use_transformer:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)
        mcts_policy_fn = transformer_mcts_policy
        print(f"  Network: Transformer (d_model={tkw.get('d_model', 128)}, layers={tkw.get('n_layers', 4)})")
    else:
        recurrent_fn = None
        mcts_policy_fn = None
        print(f"  Network: {channels} channels, {blocks} res blocks")

    # Warmup
    print("  Warming up...")
    rng = jax.random.PRNGKey(42)
    rng, game_rng = jax.random.split(rng)
    trajectory = play_games_batched(
        params=params, rng=game_rng, network=network, env_config=env_config,
        batch_size=min(batch_size, 16), max_turns=10, max_moves=50,
        temperature=1.0, num_simulations=min(num_simulations, 25),
        mcts_policy_fn=mcts_policy_fn, recurrent_fn=recurrent_fn,
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
        mcts_policy_fn=mcts_policy_fn, recurrent_fn=recurrent_fn,
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


def run_cpu_profile(use_transformer=False, transformer_kwargs=None):
    """Small config for CPU/laptop."""
    tkw = transformer_kwargs or {}
    label = " [Transformer]" if use_transformer else ""
    print("\n" + "=" * 60)
    print(f"CPU MODE (small batches for laptop){label}")
    print("=" * 60)

    rows, cols = 11, 9
    channels, blocks = 32, 2

    profile_network_throughput([4, 8, 16, 32], rows, cols, channels, blocks,
                               use_transformer=use_transformer, transformer_kwargs=tkw)
    profile_mcts_policy([4, 8, 16], rows, cols, channels, blocks, num_simulations=25,
                        use_transformer=use_transformer, transformer_kwargs=tkw)
    sims_per_sec = profile_full_games(rows, cols, channels, blocks,
                                       batch_size=4, num_simulations=25, max_turns=50,
                                       use_transformer=use_transformer, transformer_kwargs=tkw)

    # Extrapolation
    print("\n" + "-" * 40)
    print("EXTRAPOLATION TO TRAINING CONFIG")
    print("-" * 40)
    train_sims = 128 * 170 * 100  # batch * moves * sims
    estimated_time = train_sims / sims_per_sec
    print(f"  Training config: 128 batch, 100 sims, ~170 moves/game")
    print(f"  Estimated time at current rate: {estimated_time:.0f}s ({estimated_time/60:.1f}min)")


def run_tpu_profile(use_transformer=False, transformer_kwargs=None):
    """Larger config for TPU."""
    tkw = transformer_kwargs or {}
    label = " [Transformer]" if use_transformer else ""
    print("\n" + "=" * 60)
    print(f"TPU MODE (larger batches){label}")
    print("=" * 60)

    ut = use_transformer
    tk = transformer_kwargs

    # Test network throughput at different scales
    print("\n--- Small network (11x9) ---")
    profile_network_throughput([32, 64, 128, 256, 512], 11, 9, 64, 4,
                               use_transformer=ut, transformer_kwargs=tk)

    print("\n--- Full network (11x9, 128ch, 10blk) ---")
    profile_network_throughput([32, 64, 128, 256], 11, 9, 128, 10,
                               use_transformer=ut, transformer_kwargs=tk)

    print("\n--- Full board (21x15, 128ch, 10blk) ---")
    profile_network_throughput([32, 64, 128], 21, 15, 128, 10,
                               use_transformer=ut, transformer_kwargs=tk)

    # MCTS scaling with Gumbel-optimal sim counts
    print("\n--- MCTS Scaling (11x9, 16 sims) ---")
    profile_mcts_policy([64, 128, 256, 512, 1024], 11, 9, 64, 4, num_simulations=16,
                        use_transformer=ut, transformer_kwargs=tk)

    print("\n--- MCTS Scaling (11x9, 32 sims) ---")
    profile_mcts_policy([64, 128, 256, 512], 11, 9, 64, 4, num_simulations=32,
                        use_transformer=ut, transformer_kwargs=tk)

    # Full games with Gumbel-optimal settings
    print("\n--- Full Games ---")
    sims_per_sec_16 = profile_full_games(11, 9, 64, 4,
                                          batch_size=256, num_simulations=16, max_turns=100,
                                          use_transformer=ut, transformer_kwargs=tk)

    sims_per_sec_32 = profile_full_games(11, 9, 64, 4,
                                          batch_size=512, num_simulations=32, max_turns=100,
                                          use_transformer=ut, transformer_kwargs=tk)

    sims_per_sec_full = profile_full_games(11, 9, 128, 10,
                                            batch_size=256, num_simulations=32, max_turns=150,
                                            use_transformer=ut, transformer_kwargs=tk)

    # Extrapolation
    print("\n" + "-" * 40)
    print("EXTRAPOLATION TO TRAINING")
    print("-" * 40)
    train_moves = 256 * 170  # batch * moves/game
    print(f"  Config: 256 batch, 32 sims, ~170 moves/game")
    print(f"  At {sims_per_sec_full:,.0f} sims/sec: {train_moves * 32 / sims_per_sec_full:.0f}s per iteration")


def profile_game_logic_vs_nn(env_config, network, params, batch_size=256, use_transformer=False):
    """Time game logic vs NN separately."""
    states = batched_reset(env_config, batch_size)
    rng = jax.random.PRNGKey(0)

    # Dummy actions
    actions = jnp.zeros(batch_size, dtype=jnp.int32)

    batched_step_fn = make_batched_step(env_config)
    batched_legal_fn = make_batched_legal_actions(env_config)
    batched_input_fn = make_batched_network_input(env_config)

    if use_transformer:
        variables = {'params': params['network_params']}
    else:
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


def run_full_training_profile(use_transformer=False, transformer_kwargs=None):
    """Benchmark at actual training config."""
    tkw = transformer_kwargs or {}
    label = " [Transformer]" if use_transformer else ""
    print("\n" + "=" * 60)
    print(f"FULL TRAINING CONFIG BENCHMARK{label}")
    print("=" * 60)

    # Match the actual training config from logs
    rows, cols = 11, 9  # or 21, 15 for full
    channels, blocks = 128, 10
    batch_size = 128
    num_simulations = 100
    max_turns = 200

    sims_per_sec = profile_full_games(rows, cols, channels, blocks,
                                       batch_size, num_simulations, max_turns,
                                       use_transformer=use_transformer, transformer_kwargs=tkw)

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
    parser.add_argument("--transformer", action="store_true", help="Use transformer instead of CNN")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer d_model")
    parser.add_argument("--n-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--ffn-dim", type=int, default=256, help="Transformer FFN dimension")
    args = parser.parse_args()

    tkw = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'ffn_dim': args.ffn_dim,
    } if args.transformer else None

    if args.full:
        run_full_training_profile(use_transformer=args.transformer, transformer_kwargs=tkw)
    elif args.tpu or IS_TPU:
        run_tpu_profile(use_transformer=args.transformer, transformer_kwargs=tkw)
    else:
        run_cpu_profile(use_transformer=args.transformer, transformer_kwargs=tkw)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
