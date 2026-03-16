"""
Batched round-robin ELO tournament for Phutball agents.

Runs all-pairs matches using play_tournament_match (fixed-perspective batched
games via transformer MCTS), then computes iterative ELO ratings.

Usage:
    from tournament_batched import run_tournament, play_tournament_match

    agents = [("ckpt_100", params_100), ("ckpt_200", params_200), ...]
    results, ratings = run_tournament(agents, network, env_config, rng)
"""

import time
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from phutball_env_jax import EnvConfig, reset, step, get_legal_actions
from network import PhutballTransformer
from self_play_batched import (
    batched_reset,
    _step_games_batched,
    _make_frozen_state,
    make_transformer_recurrent_fn,
    transformer_mcts_policy,
)


def play_tournament_match(
    params_a: dict,
    params_b: dict,
    network: PhutballTransformer,
    env_config: EnvConfig,
    rng: jnp.ndarray,
    games_per_perspective: int = 10,
    batch_size: int = 64,
    num_simulations: int = 32,
    max_moves: int = 2048,
    mcts_policy_fn=None,
    recurrent_fn=None,
) -> Tuple[int, int, int, dict]:
    """
    Play a head-to-head match between agents A and B with fixed perspectives.

    Plays games_per_perspective games with A as P1, then games_per_perspective
    with B as P1. Uses play_vs_checkpoint_batched-style loop with explicit
    side assignment (no random_sides).

    Returns:
        a_wins: total wins for agent A
        b_wins: total wins for agent B
        draws: total draws
        stats: dict with per-perspective breakdowns and game statistics
    """
    _mcts_fn = mcts_policy_fn or transformer_mcts_policy
    if recurrent_fn is None:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    total_a_wins = 0
    total_b_wins = 0
    total_draws = 0
    all_turns = []
    all_jump_counts = []

    # Two perspectives: A as P1, then B as P1
    for perspective in range(2):
        if perspective == 0:
            p1_params, p2_params = params_a, params_b
        else:
            p1_params, p2_params = params_b, params_a

        games_remaining = games_per_perspective
        while games_remaining > 0:
            cur_batch = min(games_remaining, batch_size)
            rng, batch_rng = jax.random.split(rng)

            a_w, b_w, dr, turns, jumps = _play_fixed_perspective_batch(
                p1_params=p1_params,
                p2_params=p2_params,
                a_is_p1=(perspective == 0),
                rng=batch_rng,
                network=network,
                env_config=env_config,
                batch_size=cur_batch,
                max_moves=max_moves,
                num_simulations=num_simulations,
                mcts_policy_fn=_mcts_fn,
                recurrent_fn=recurrent_fn,
            )
            total_a_wins += a_w
            total_b_wins += b_w
            total_draws += dr
            all_turns.extend(turns)
            all_jump_counts.extend(jumps)
            games_remaining -= cur_batch

    stats = {
        "a_wins": int(total_a_wins),
        "b_wins": int(total_b_wins),
        "draws": int(total_draws),
        "avg_game_length": float(np.mean(all_turns)) if all_turns else 0.0,
        "avg_jumps_per_game": float(np.mean(all_jump_counts)) if all_jump_counts else 0.0,
        "total_games": 2 * games_per_perspective,
    }

    return int(total_a_wins), int(total_b_wins), int(total_draws), stats


def _play_fixed_perspective_batch(
    p1_params: dict,
    p2_params: dict,
    a_is_p1: bool,
    rng: jnp.ndarray,
    network: PhutballTransformer,
    env_config: EnvConfig,
    batch_size: int,
    max_moves: int,
    num_simulations: int,
    mcts_policy_fn,
    recurrent_fn,
) -> Tuple[int, int, int, list, list]:
    """
    Play a batch of games with fixed P1/P2 param assignment.

    Returns: (a_wins, b_wins, draws, per_game_turns, per_game_jump_counts)
    """
    states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)
    jump_counts = jnp.zeros((batch_size,), dtype=jnp.int32)
    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    for step_idx in range(max_moves):
        if not bool(jnp.any(~terminated)):
            break

        current_player = states.current_player
        use_p1 = (current_player == 1)

        rng, rng_p1, rng_p2 = jax.random.split(rng, 3)

        actions_p1, _, _ = mcts_policy_fn(
            p1_params, states, rng_p1, network, env_config,
            num_simulations=num_simulations, temperature=0.0,
            dirichlet_fraction=0.0, recurrent_fn=recurrent_fn,
        )

        actions_p2, _, _ = mcts_policy_fn(
            p2_params, states, rng_p2, network, env_config,
            num_simulations=num_simulations, temperature=0.0,
            dirichlet_fraction=0.0, recurrent_fn=recurrent_fn,
        )

        actions = jnp.where(use_p1, actions_p1, actions_p2)

        # Track jumps (actions in [rows*cols, 2*rows*cols) are jump actions)
        total_positions = rows * cols
        is_jump = (actions >= total_positions) & (actions < 2 * total_positions)
        jump_counts = jump_counts + (is_jump & ~terminated).astype(jnp.int32)

        states, terminated = _step_games_batched(states, actions, terminated, env_config)

    winners = states.winner
    per_game_turns = list(np.array(states.num_turns))

    # Map winners to A/B wins
    if a_is_p1:
        a_wins = int(jnp.sum((winners == 1).astype(jnp.int32)))
        b_wins = int(jnp.sum((winners == 2).astype(jnp.int32)))
    else:
        a_wins = int(jnp.sum((winners == 2).astype(jnp.int32)))
        b_wins = int(jnp.sum((winners == 1).astype(jnp.int32)))

    draws = int(jnp.sum((winners == 0).astype(jnp.int32)))

    return a_wins, b_wins, draws, per_game_turns, list(np.array(jump_counts))


def compute_elo(
    names: List[str],
    W: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    base_elo: float = 1500.0,
    K: float = 32.0,
    iters: int = 10,
) -> Dict[str, float]:
    """
    Iterative ELO solver from W/D/L matrices.

    W[i,j] = wins of agent i vs agent j.
    Returns dict mapping agent name to ELO rating.
    """
    n = len(names)
    ratings = np.full(n, base_elo, dtype=np.float64)

    for _ in range(iters):
        for i in range(n):
            for j in range(i + 1, n):
                wins_ij = float(W[i, j])
                draws_ij = float(D[i, j])
                loss_ij = float(L[i, j])
                N_ij = wins_ij + draws_ij + loss_ij
                if N_ij == 0:
                    continue

                S_i = wins_ij + 0.5 * draws_ij
                diff = (ratings[j] - ratings[i]) / 400.0
                p_i = 1.0 / (1.0 + 10.0 ** diff)
                E_i = N_ij * p_i

                delta = K * (S_i - E_i) / max(1.0, N_ij)
                ratings[i] += delta
                ratings[j] -= delta

    return {name: float(ratings[i]) for i, name in enumerate(names)}


def run_tournament(
    agents: List[Tuple[str, dict]],
    network: PhutballTransformer,
    env_config: EnvConfig,
    rng: jnp.ndarray,
    games_per_perspective: int = 10,
    batch_size: int = 64,
    num_simulations: int = 32,
    max_moves: int = 2048,
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Run a round-robin tournament between all agent pairs.

    Args:
        agents: list of (name, params) tuples
        network: PhutballTransformer instance
        env_config: board/game configuration
        rng: JAX random key
        games_per_perspective: games per side per matchup (total = 2x this)
        batch_size: max games per MCTS batch
        num_simulations: MCTS simulations per move
        max_moves: max moves before game is declared draw

    Returns:
        match_results: list of per-match result dicts
        elo_ratings: dict mapping agent name to ELO rating
    """
    names = [name for name, _ in agents]
    n = len(agents)
    W = np.zeros((n, n), dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)
    L = np.zeros((n, n), dtype=np.float64)

    recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    match_results = []
    total_matchups = n * (n - 1) // 2
    completed = 0

    print(f"Tournament: {n} agents, {total_matchups} matchups, "
          f"{games_per_perspective * 2} games each")
    print(f"Board: {env_config.rows}x{env_config.cols}, "
          f"MCTS sims: {num_simulations}\n")

    for i in range(n):
        for j in range(i + 1, n):
            name_a, params_a = agents[i]
            name_b, params_b = agents[j]

            rng, match_rng = jax.random.split(rng)
            t0 = time.time()

            a_wins, b_wins, draws, stats = play_tournament_match(
                params_a=params_a,
                params_b=params_b,
                network=network,
                env_config=env_config,
                rng=match_rng,
                games_per_perspective=games_per_perspective,
                batch_size=batch_size,
                num_simulations=num_simulations,
                max_moves=max_moves,
                recurrent_fn=recurrent_fn,
            )
            elapsed = time.time() - t0

            W[i, j] += a_wins
            D[i, j] += draws
            L[i, j] += b_wins
            W[j, i] += b_wins
            D[j, i] += draws
            L[j, i] += a_wins

            total = a_wins + b_wins + draws
            wr_a = a_wins / total if total > 0 else 0.0

            completed += 1
            print(f"  [{completed}/{total_matchups}] {name_a} vs {name_b}: "
                  f"{a_wins}W-{draws}D-{b_wins}L "
                  f"({wr_a:.0%} for {name_a}) "
                  f"avg_len={stats['avg_game_length']:.0f} "
                  f"avg_jumps={stats['avg_jumps_per_game']:.1f} "
                  f"[{elapsed:.1f}s]")

            result = {
                "agent_a": name_a,
                "agent_b": name_b,
                **stats,
                "elapsed_s": round(elapsed, 1),
            }
            match_results.append(result)

    elo_ratings = compute_elo(names, W, D, L)

    print(f"\n{'='*50}")
    print("ELO Ratings:")
    for name in sorted(elo_ratings, key=elo_ratings.get, reverse=True):
        print(f"  {name}: {elo_ratings[name]:.0f}")
    print(f"{'='*50}")

    return match_results, elo_ratings
