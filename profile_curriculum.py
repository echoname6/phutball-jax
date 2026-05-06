"""
Profiler to compare curriculum generation speed vs MCTS self-play.

This ensures curriculum generation is not a bottleneck in training.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from phutball_env_jax import EnvConfig
from curriculum_puzzles import (
    generate_curriculum_batch,
    generate_one_move_win_state,
    generate_n_move_win_state,
)
from network import PhutballNetwork, create_transformer_network, init_transformer_network
from self_play_batched import (
    play_games_batched,
    trajectory_to_training_examples,
    transformer_mcts_policy,
    make_transformer_recurrent_fn,
)


@dataclass
class ProfilingResult:
    """Results from a profiling run."""
    name: str
    total_time: float
    num_examples: int
    examples_per_second: float
    iterations: int
    time_per_iteration: float


def profile_curriculum_generation(
    env_config: EnvConfig,
    batch_size: int = 64,
    num_iterations: int = 10,
    jump_distribution: List[float] = None,
    warmup: int = 2,
) -> Tuple[ProfilingResult, dict]:
    """
    Profile curriculum batch generation.

    Args:
        env_config: Environment configuration
        batch_size: Examples per batch
        num_iterations: Number of batches to generate
        jump_distribution: Distribution over 1-4 jump examples
        warmup: Number of warmup iterations (not counted)

    Returns:
        Tuple of (ProfilingResult with timing statistics, cumulative stats dict)
    """
    rng = jax.random.PRNGKey(42)

    # Warmup runs (compile and cache)
    for _ in range(warmup):
        rng, batch_rng = jax.random.split(rng)
        _ = generate_curriculum_batch(batch_rng, env_config, batch_size, jump_distribution)

    # Block until warmup complete
    jax.block_until_ready(rng)

    # Timed runs
    start_time = time.perf_counter()
    cumulative_stats = {
        'curriculum_1jump': 0,
        'curriculum_2jump': 0,
        'curriculum_3jump': 0,
        'curriculum_4jump': 0,
        'curriculum_total': 0,
    }

    for _ in range(num_iterations):
        rng, batch_rng = jax.random.split(rng)
        states, policies, values, stats = generate_curriculum_batch(
            batch_rng, env_config, batch_size, jump_distribution, return_stats=True
        )
        for k, v in stats.items():
            cumulative_stats[k] += v

    # Block until all computation complete
    jax.block_until_ready(states)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_examples = num_iterations * batch_size

    return ProfilingResult(
        name="Curriculum Generation",
        total_time=total_time,
        num_examples=total_examples,
        examples_per_second=total_examples / total_time,
        iterations=num_iterations,
        time_per_iteration=total_time / num_iterations,
    ), cumulative_stats


def profile_individual_generators(
    env_config: EnvConfig,
    num_samples: int = 100,
    warmup: int = 5,
) -> dict:
    """
    Profile individual N-move generators.

    Returns dict mapping jump count to examples/second.
    """
    rng = jax.random.PRNGKey(123)
    results = {}

    for num_jumps in [1, 2, 3, 4]:
        # Warmup
        for _ in range(warmup):
            rng, state_rng, player_rng = jax.random.split(rng, 3)
            player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2
            if num_jumps == 1:
                _ = generate_one_move_win_state(state_rng, env_config, player=player)
            else:
                _ = generate_n_move_win_state(state_rng, env_config, num_jumps=num_jumps, player=player)

        # Timed runs
        start_time = time.perf_counter()

        for _ in range(num_samples):
            rng, state_rng, player_rng = jax.random.split(rng, 3)
            player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2
            if num_jumps == 1:
                state, action = generate_one_move_win_state(state_rng, env_config, player=player)
            else:
                state, actions = generate_n_move_win_state(state_rng, env_config, num_jumps=num_jumps, player=player)

        jax.block_until_ready(rng)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        results[num_jumps] = {
            'time': elapsed,
            'samples': num_samples,
            'examples_per_second': num_samples / elapsed,
            'ms_per_example': (elapsed / num_samples) * 1000,
        }

    return results


def profile_mcts_selfplay(
    env_config: EnvConfig,
    network,
    params: dict,
    batch_size: int = 16,
    num_iterations: int = 3,
    num_simulations: int = 50,
    max_turns: int = 100,
    warmup: int = 1,
    use_transformer: bool = False,
) -> ProfilingResult:
    """
    Profile MCTS self-play game generation.

    Args:
        env_config: Environment configuration
        network: Neural network
        params: Network parameters
        batch_size: Games per batch
        num_iterations: Number of batches to play
        num_simulations: MCTS simulations per move
        max_turns: Max turns per game (for quicker profiling)
        warmup: Number of warmup iterations
        use_transformer: Use transformer MCTS functions

    Returns:
        ProfilingResult with timing statistics
    """
    if use_transformer:
        mcts_policy_fn = transformer_mcts_policy
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)
    else:
        mcts_policy_fn = None
        recurrent_fn = None

    rng = jax.random.PRNGKey(999)

    # Warmup
    for _ in range(warmup):
        rng, game_rng = jax.random.split(rng)
        trajectory = play_games_batched(
            params=params,
            rng=game_rng,
            network=network,
            env_config=env_config,
            batch_size=batch_size,
            max_turns=max_turns,
            num_simulations=num_simulations,
            mcts_policy_fn=mcts_policy_fn,
            recurrent_fn=recurrent_fn,
        )
        states, policies, values = trajectory_to_training_examples(trajectory)

    jax.block_until_ready(rng)

    # Timed runs
    start_time = time.perf_counter()
    total_examples = 0

    for _ in range(num_iterations):
        rng, game_rng = jax.random.split(rng)
        trajectory = play_games_batched(
            params=params,
            rng=game_rng,
            network=network,
            env_config=env_config,
            batch_size=batch_size,
            max_turns=max_turns,
            num_simulations=num_simulations,
            mcts_policy_fn=mcts_policy_fn,
            recurrent_fn=recurrent_fn,
        )
        states, policies, values = trajectory_to_training_examples(trajectory)
        total_examples += len(states)

    jax.block_until_ready(states)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    label = "Transformer" if use_transformer else "CNN"
    return ProfilingResult(
        name=f"MCTS Self-Play ({label}, {num_simulations} sims)",
        total_time=total_time,
        num_examples=total_examples,
        examples_per_second=total_examples / total_time,
        iterations=num_iterations,
        time_per_iteration=total_time / num_iterations,
    )


def profile_raw_network_selfplay(
    env_config: EnvConfig,
    network,
    params: dict,
    batch_size: int = 64,
    num_iterations: int = 5,
    max_turns: int = 200,
    warmup: int = 1,
    use_transformer: bool = False,
) -> ProfilingResult:
    """
    Profile self-play with raw network (no MCTS).
    """
    rng = jax.random.PRNGKey(777)

    # Warmup
    for _ in range(warmup):
        rng, game_rng = jax.random.split(rng)
        trajectory = play_games_batched(
            params=params,
            rng=game_rng,
            network=network,
            env_config=env_config,
            batch_size=batch_size,
            max_turns=max_turns,
            num_simulations=0,  # Raw network
        )
        states, policies, values = trajectory_to_training_examples(trajectory)

    jax.block_until_ready(rng)

    # Timed runs
    start_time = time.perf_counter()
    total_examples = 0

    for _ in range(num_iterations):
        rng, game_rng = jax.random.split(rng)
        trajectory = play_games_batched(
            params=params,
            rng=game_rng,
            network=network,
            env_config=env_config,
            batch_size=batch_size,
            max_turns=max_turns,
            num_simulations=0,
        )
        states, policies, values = trajectory_to_training_examples(trajectory)
        total_examples += len(states)

    jax.block_until_ready(states)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    label = "Transformer" if use_transformer else "CNN"
    return ProfilingResult(
        name=f"Raw Network Self-Play ({label}, no MCTS)",
        total_time=total_time,
        num_examples=total_examples,
        examples_per_second=total_examples / total_time,
        iterations=num_iterations,
        time_per_iteration=total_time / num_iterations,
    )


def format_result(result: ProfilingResult) -> str:
    """Format a profiling result for display."""
    lines = [
        f"  Total time: {result.total_time:.3f}s",
        f"  Examples generated: {result.num_examples:,}",
        f"  Examples/second: {result.examples_per_second:,.1f}",
        f"  Time per batch: {result.time_per_iteration*1000:.1f}ms",
    ]
    return "\n".join(lines)


def run_profiler(
    rows: int = 21,
    cols: int = 15,
    curriculum_batch_size: int = 64,
    curriculum_iterations: int = 20,
    selfplay_batch_size: int = 16,
    selfplay_iterations: int = 3,
    num_simulations: int = 50,
    include_mcts: bool = True,
    use_transformer: bool = False,
    transformer_kwargs: dict = None,
):
    """
    Run the full profiler and compare curriculum vs self-play.
    """
    print("=" * 60)
    print("CURRICULUM vs SELF-PLAY PROFILER")
    print("=" * 60)

    env_config = EnvConfig(rows=rows, cols=cols)
    print(f"\nBoard size: {rows}x{cols}")
    print(f"Action space: {2 * rows * cols + 1}")

    # Profile individual generators first
    print("\n" + "-" * 40)
    print("Individual Generator Performance:")
    print("-" * 40)

    individual_results = profile_individual_generators(env_config, num_samples=100)
    for num_jumps, stats in individual_results.items():
        print(f"  {num_jumps}-jump: {stats['examples_per_second']:.1f} ex/s ({stats['ms_per_example']:.2f} ms/ex)")

    # Profile curriculum batch generation
    print("\n" + "-" * 40)
    print("Curriculum Batch Generation:")
    print("-" * 40)

    curriculum_result, curriculum_stats = profile_curriculum_generation(
        env_config,
        batch_size=curriculum_batch_size,
        num_iterations=curriculum_iterations,
    )
    print(f"\n{curriculum_result.name}:")
    print(format_result(curriculum_result))
    print(f"\n  Jump distribution:")
    total = curriculum_stats['curriculum_total']
    for jumps in [1, 2, 3, 4]:
        count = curriculum_stats[f'curriculum_{jumps}jump']
        pct = (count / total * 100) if total > 0 else 0
        print(f"    {jumps}-jump: {count:,} ({pct:.1f}%)")

    # Initialize network for self-play
    print("\n" + "-" * 40)
    print("Initializing network for self-play...")
    print("-" * 40)

    if use_transformer:
        tkw = transformer_kwargs or {}
        network = create_transformer_network(
            rows=rows, cols=cols,
            d_model=tkw.get('d_model', 128),
            n_layers=tkw.get('n_layers', 4),
            n_heads=tkw.get('n_heads', 4),
            ffn_dim=tkw.get('ffn_dim', 256),
        )
        rng = jax.random.PRNGKey(0)
        variables = init_transformer_network(rng, network, num_input_channels=9)
        params = {'network_params': variables['params']}
        print(f"Transformer initialized: d_model={tkw.get('d_model', 128)}, "
              f"layers={tkw.get('n_layers', 4)}")
    else:
        network = PhutballNetwork(
            num_channels=64,  # Smaller for faster profiling
            num_res_blocks=4,
            rows=rows,
            cols=cols,
        )
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, 6, rows, cols))
        variables = network.init(rng, dummy_input)
        params = {
            'network_params': variables['params'],
            'batch_stats': variables.get('batch_stats', {}),
        }
        print(f"CNN initialized: {64} channels, {4} res blocks")

    # Profile raw network self-play
    print("\n" + "-" * 40)
    print("Raw Network Self-Play (no MCTS):")
    print("-" * 40)

    raw_result = profile_raw_network_selfplay(
        env_config,
        network,
        params,
        batch_size=selfplay_batch_size,
        num_iterations=selfplay_iterations,
        max_turns=200,
        use_transformer=use_transformer,
    )
    print(f"\n{raw_result.name}:")
    print(format_result(raw_result))

    # Profile MCTS self-play (optional, can be slow)
    mcts_result = None
    if include_mcts:
        print("\n" + "-" * 40)
        print(f"MCTS Self-Play ({num_simulations} simulations):")
        print("-" * 40)

        mcts_result = profile_mcts_selfplay(
            env_config,
            network,
            params,
            batch_size=selfplay_batch_size,
            num_iterations=selfplay_iterations,
            num_simulations=num_simulations,
            max_turns=50,  # Shorter games for MCTS profiling
            use_transformer=use_transformer,
        )
        print(f"\n{mcts_result.name}:")
        print(format_result(mcts_result))

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nCurriculum Generation: {curriculum_result.examples_per_second:,.1f} examples/second")
    print(f"Raw Network Self-Play: {raw_result.examples_per_second:,.1f} examples/second")

    speedup_raw = curriculum_result.examples_per_second / raw_result.examples_per_second
    print(f"\nCurriculum is {speedup_raw:.1f}x {'faster' if speedup_raw > 1 else 'slower'} than raw network self-play")

    if mcts_result:
        print(f"MCTS Self-Play ({num_simulations} sims): {mcts_result.examples_per_second:,.1f} examples/second")
        speedup_mcts = curriculum_result.examples_per_second / mcts_result.examples_per_second
        print(f"Curriculum is {speedup_mcts:.1f}x {'faster' if speedup_mcts > 1 else 'slower'} than MCTS self-play")

    # Bottleneck analysis
    print("\n" + "-" * 40)
    print("Bottleneck Analysis:")
    print("-" * 40)

    if speedup_raw > 1:
        print("✓ Curriculum generation is NOT a bottleneck compared to raw network self-play")
    else:
        print("⚠ WARNING: Curriculum generation is slower than raw network self-play!")

    if mcts_result and speedup_mcts > 10:
        print("✓ Curriculum generation is much faster than MCTS self-play (expected)")
    elif mcts_result:
        print(f"  Curriculum is {speedup_mcts:.1f}x faster than MCTS")

    return {
        'curriculum': curriculum_result,
        'curriculum_stats': curriculum_stats,
        'raw_selfplay': raw_result,
        'mcts_selfplay': mcts_result,
        'individual': individual_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile curriculum vs self-play")
    parser.add_argument("--rows", type=int, default=21, help="Board rows")
    parser.add_argument("--cols", type=int, default=15, help="Board cols")
    parser.add_argument("--curriculum-batch", type=int, default=64, help="Curriculum batch size")
    parser.add_argument("--curriculum-iters", type=int, default=20, help="Curriculum iterations")
    parser.add_argument("--selfplay-batch", type=int, default=16, help="Self-play batch size")
    parser.add_argument("--selfplay-iters", type=int, default=3, help="Self-play iterations")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations")
    parser.add_argument("--no-mcts", action="store_true", help="Skip MCTS profiling")
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

    run_profiler(
        rows=args.rows,
        cols=args.cols,
        curriculum_batch_size=args.curriculum_batch,
        curriculum_iterations=args.curriculum_iters,
        selfplay_batch_size=args.selfplay_batch,
        selfplay_iterations=args.selfplay_iters,
        num_simulations=args.mcts_sims,
        include_mcts=not args.no_mcts,
        use_transformer=args.transformer,
        transformer_kwargs=tkw,
    )
