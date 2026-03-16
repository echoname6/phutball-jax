"""
Batched round-robin ELO tournament for Phutball agents.

Primary entry point: run_tournament_megabatch — runs all matchup games in
one (or few) mega-batched game loops instead of sequential per-matchup loops.

Fallback: run_tournament / play_tournament_match — sequential per-matchup.

Usage:
    from tournament_batched import run_tournament_megabatch

    agents = [("ckpt_100", params_100), ("ckpt_200", params_200), ...]
    results, ratings = run_tournament_megabatch(agents, network, env_config, rng)
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


# ═══════════════════════════════════════════════════════════════
#  Mega-batched tournament (primary path)
# ═══════════════════════════════════════════════════════════════

def _build_manifest(
    num_agents: int,
    games_per_perspective: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Create arrays mapping each game to its agents and matchup.

    For K agents and G games_per_perspective:
      - K*(K-1)/2 matchups
      - Each matchup has 2*G games (G per perspective)
      - Total N = K*(K-1)/2 * 2*G games

    Returns:
        p1_agent_idx: (N,) int — which agent plays P1
        p2_agent_idx: (N,) int — which agent plays P2
        matchup_idx:  (N,) int — which matchup this game belongs to
        perspective:  (N,) int — 0 = agent_i is P1, 1 = agent_j is P1
        matchup_pairs: list of (i, j) — agent index pairs per matchup
    """
    p1_ids, p2_ids, m_ids, persp = [], [], [], []
    matchup_pairs = []
    m = 0

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            matchup_pairs.append((i, j))
            # Perspective 0: agent i is P1, agent j is P2
            for _ in range(games_per_perspective):
                p1_ids.append(i)
                p2_ids.append(j)
                m_ids.append(m)
                persp.append(0)
            # Perspective 1: agent j is P1, agent i is P2
            for _ in range(games_per_perspective):
                p1_ids.append(j)
                p2_ids.append(i)
                m_ids.append(m)
                persp.append(1)
            m += 1

    return (
        np.array(p1_ids, dtype=np.int32),
        np.array(p2_ids, dtype=np.int32),
        np.array(m_ids, dtype=np.int32),
        np.array(persp, dtype=np.int32),
        matchup_pairs,
    )


def _run_megabatch(
    all_params: list,
    p1_agent_idx: jnp.ndarray,
    p2_agent_idx: jnp.ndarray,
    network: PhutballTransformer,
    env_config: EnvConfig,
    rng: jnp.ndarray,
    num_simulations: int = 32,
    max_moves: int = 2048,
    recurrent_fn=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Play a batch of games where different games use different agent params.

    Uses jax.lax.while_loop for XLA compilation of the entire game loop.
    All K agents are evaluated every step (no dynamic Python branching),
    but XLA compiles and fuses everything for 10-100x speedup over
    Python-dispatched selective calls.

    The Python `for agent_id in range(num_agents)` inside body_fn is
    traced ONCE at compile time and unrolled into the XLA program.
    num_agents must be known at trace time (recompiles if changed).

    Returns:
        winners:     (batch,) — 1=P1 won, 2=P2 won, 0=draw
        num_turns:   (batch,) — game lengths
        jump_counts: (batch,) — total jumps per game
    """
    from jax import lax

    batch_size = p1_agent_idx.shape[0]
    num_agents = len(all_params)
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    if recurrent_fn is None:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    batch_arange = jnp.arange(batch_size)

    init_states = batched_reset(env_config, batch_size)
    init_terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
    init_jumps = jnp.zeros(batch_size, dtype=jnp.int32)
    init_step = jnp.int32(0)

    def cond_fn(carry):
        states, terminated, jump_counts, rng, step_idx = carry
        any_active = jnp.any(~terminated)
        within_budget = step_idx < max_moves
        return any_active & within_budget

    def body_fn(carry):
        states, terminated, jump_counts, rng, step_idx = carry

        current_agent = jnp.where(
            states.current_player == 1, p1_agent_idx, p2_agent_idx
        )

        # Run ALL agents on full batch — traced once, compiled into XLA
        rngs = jax.random.split(rng, num_agents + 1)
        rng_next = rngs[0]
        agent_rngs = rngs[1:]

        agent_actions = []
        for agent_id in range(num_agents):
            a, _, _ = transformer_mcts_policy(
                all_params[agent_id], states, agent_rngs[agent_id],
                network, env_config,
                num_simulations=num_simulations, temperature=0.0,
                dirichlet_fraction=0.0, recurrent_fn=recurrent_fn,
            )
            agent_actions.append(a)

        stacked = jnp.stack(agent_actions)  # (K, batch)
        actions = stacked[current_agent, batch_arange]

        # Track jumps
        is_jump = (actions >= total_positions) & (actions < 2 * total_positions)
        jump_counts = jump_counts + (is_jump & ~terminated).astype(jnp.int32)

        # Step envs
        new_states_raw = jax.vmap(
            lambda s, a: step(s, a, env_config)
        )(states, actions)
        new_terminated = terminated | new_states_raw.terminated
        new_states = _make_frozen_state(
            states, new_states_raw, new_terminated, env_config
        )

        return (new_states, new_terminated, jump_counts, rng_next, step_idx + 1)

    final = lax.while_loop(
        cond_fn, body_fn,
        (init_states, init_terminated, init_jumps, rng, init_step)
    )
    states, terminated, jump_counts, _, _ = final

    return states.winner, states.num_turns, jump_counts


def _collect_results(
    winners: np.ndarray,
    num_turns: np.ndarray,
    jump_counts: np.ndarray,
    matchup_idx: np.ndarray,
    matchup_pairs: List[Tuple[int, int]],
    p1_agent_idx: np.ndarray,
    p2_agent_idx: np.ndarray,
    num_agents: int,
    agent_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Unpack game outcomes into W/D/L matrices and per-matchup stats.

    Returns:
        W: (K, K) win matrix
        D: (K, K) draw matrix
        L: (K, K) loss matrix
        match_results: list of per-matchup stat dicts
    """
    W = np.zeros((num_agents, num_agents), dtype=np.float64)
    D = np.zeros((num_agents, num_agents), dtype=np.float64)
    L = np.zeros((num_agents, num_agents), dtype=np.float64)
    match_results = []

    for m_idx, (i, j) in enumerate(matchup_pairs):
        mask = matchup_idx == m_idx
        m_winners = winners[mask]
        m_turns = num_turns[mask]
        m_jumps = jump_counts[mask]
        m_p1 = p1_agent_idx[mask]

        # Agent i wins: P1 won and i was P1, or P2 won and i was P2
        i_wins = int(np.sum(((m_winners == 1) & (m_p1 == i)) |
                            ((m_winners == 2) & (m_p1 != i))))
        j_wins = int(np.sum(((m_winners == 1) & (m_p1 == j)) |
                            ((m_winners == 2) & (m_p1 != j))))
        draws = int(np.sum(m_winners == 0))

        W[i, j] += i_wins
        D[i, j] += draws
        L[i, j] += j_wins
        W[j, i] += j_wins
        D[j, i] += draws
        L[j, i] += i_wins

        total = i_wins + j_wins + draws
        match_results.append({
            "agent_a": agent_names[i],
            "agent_b": agent_names[j],
            "a_wins": i_wins,
            "b_wins": j_wins,
            "draws": draws,
            "total_games": total,
            "avg_game_length": float(np.mean(m_turns)) if len(m_turns) > 0 else 0.0,
            "avg_jumps_per_game": float(np.mean(m_jumps)) if len(m_jumps) > 0 else 0.0,
        })

    return W, D, L, match_results


def run_tournament_megabatch(
    agents: List[Tuple[str, dict]],
    network: PhutballTransformer,
    env_config: EnvConfig,
    rng: jnp.ndarray,
    games_per_perspective: int = 10,
    max_batch_size: int = 256,
    num_simulations: int = 32,
    max_moves: int = 2048,
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Full round-robin tournament using mega-batched games.

    All matchup games run in one (or few) mega-batched game loops instead of
    sequential per-matchup loops. At each step, all K agents evaluate the full
    batch and actions are selected per-game based on the agent assignment.

    Args:
        agents: list of (name, params) tuples
        network: PhutballTransformer instance
        env_config: board/game configuration
        rng: JAX random key
        games_per_perspective: games per side per matchup (total = 2x per pair)
        max_batch_size: chunk games into batches of this size
        num_simulations: MCTS simulations per move
        max_moves: max moves before game is declared draw

    Returns:
        match_results: list of per-matchup result dicts
        elo_ratings: dict mapping agent name to ELO rating
    """
    names = [name for name, _ in agents]
    all_params = [params for _, params in agents]
    num_agents = len(agents)

    # Phase 1: Build manifest
    p1_idx, p2_idx, m_idx, persp, matchup_pairs = _build_manifest(
        num_agents, games_per_perspective
    )
    total_games = len(p1_idx)
    num_matchups = len(matchup_pairs)

    print(f"Tournament: {num_agents} agents, {num_matchups} matchups, "
          f"{games_per_perspective * 2} games each, {total_games} total games")
    print(f"Board: {env_config.rows}x{env_config.cols}, "
          f"MCTS sims: {num_simulations}")

    recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    # Phase 2: Run mega-batches
    all_winners = []
    all_turns = []
    all_jumps = []
    all_m_idx = []
    all_p1_idx = []

    num_chunks = (total_games + max_batch_size - 1) // max_batch_size
    print(f"Running in {num_chunks} chunk(s) of up to {max_batch_size} games\n")

    t0 = time.time()

    for chunk_i in range(num_chunks):
        start = chunk_i * max_batch_size
        end = min(start + max_batch_size, total_games)
        chunk_size = end - start

        chunk_p1 = jnp.array(p1_idx[start:end])
        chunk_p2 = jnp.array(p2_idx[start:end])

        rng, chunk_rng = jax.random.split(rng)

        print(f"  Chunk {chunk_i + 1}/{num_chunks}: compiling + running {chunk_size} games...")
        ct0 = time.time()
        winners, turns, jumps = _run_megabatch(
            all_params=all_params,
            p1_agent_idx=chunk_p1,
            p2_agent_idx=chunk_p2,
            network=network,
            env_config=env_config,
            rng=chunk_rng,
            num_simulations=num_simulations,
            max_moves=max_moves,
            recurrent_fn=recurrent_fn,
        )
        winners.block_until_ready()
        ct1 = time.time()

        print(f"  Chunk {chunk_i + 1}/{num_chunks}: {chunk_size} games [{ct1 - ct0:.1f}s]")

        all_winners.append(np.array(winners))
        all_turns.append(np.array(turns))
        all_jumps.append(np.array(jumps))
        all_m_idx.append(m_idx[start:end])
        all_p1_idx.append(p1_idx[start:end])

    elapsed = time.time() - t0
    print(f"\nAll games complete [{elapsed:.1f}s total]\n")

    # Phase 3: Collect results
    cat_winners = np.concatenate(all_winners)
    cat_turns = np.concatenate(all_turns)
    cat_jumps = np.concatenate(all_jumps)
    cat_m_idx = np.concatenate(all_m_idx)
    cat_p1_idx = np.concatenate(all_p1_idx)

    W, D, L, match_results = _collect_results(
        winners=cat_winners,
        num_turns=cat_turns,
        jump_counts=cat_jumps,
        matchup_idx=cat_m_idx,
        matchup_pairs=matchup_pairs,
        p1_agent_idx=cat_p1_idx,
        p2_agent_idx=p2_idx,
        num_agents=num_agents,
        agent_names=names,
    )

    # Print per-matchup results
    for r in match_results:
        total = r["a_wins"] + r["b_wins"] + r["draws"]
        wr = r["a_wins"] / total if total > 0 else 0.0
        print(f"  {r['agent_a']} vs {r['agent_b']}: "
              f"{r['a_wins']}W-{r['draws']}D-{r['b_wins']}L "
              f"({wr:.0%} for {r['agent_a']}) "
              f"avg_len={r['avg_game_length']:.0f} "
              f"avg_jumps={r['avg_jumps_per_game']:.1f}")

    # Phase 4: Compute ELO
    elo_ratings = compute_elo(names, W, D, L)

    print(f"\n{'='*50}")
    print("ELO Ratings:")
    for name in sorted(elo_ratings, key=elo_ratings.get, reverse=True):
        print(f"  {name}: {elo_ratings[name]:.0f}")
    print(f"{'='*50}")

    return match_results, elo_ratings


# ═══════════════════════════════════════════════════════════════
#  Sequential per-matchup tournament (fallback)
# ═══════════════════════════════════════════════════════════════

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
    Sequential fallback — prefer run_tournament_megabatch for full tournaments.
    """
    _mcts_fn = mcts_policy_fn or transformer_mcts_policy
    if recurrent_fn is None:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    total_a_wins = 0
    total_b_wins = 0
    total_draws = 0
    all_turns = []
    all_jump_counts = []

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
    """Play a batch of games with fixed P1/P2 param assignment."""
    states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)
    jump_counts = jnp.zeros((batch_size,), dtype=jnp.int32)
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    for step_idx in range(max_moves):
        if not bool(jnp.any(~terminated)):
            break

        is_p1_turn = (states.current_player == 1)
        p1_active = bool(jnp.any(is_p1_turn & ~terminated))
        p2_active = bool(jnp.any(~is_p1_turn & ~terminated))

        rng, mcts_rng = jax.random.split(rng)

        if p1_active and not p2_active:
            # Only P1 needs actions — skip P2 MCTS entirely
            actions, _, _ = mcts_policy_fn(
                p1_params, states, mcts_rng, network, env_config,
                num_simulations=num_simulations, temperature=0.0,
                dirichlet_fraction=0.0, recurrent_fn=recurrent_fn,
            )
        elif p2_active and not p1_active:
            # Only P2 needs actions — skip P1 MCTS entirely
            actions, _, _ = mcts_policy_fn(
                p2_params, states, mcts_rng, network, env_config,
                num_simulations=num_simulations, temperature=0.0,
                dirichlet_fraction=0.0, recurrent_fn=recurrent_fn,
            )
        else:
            # Both players active — run both
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
            actions = jnp.where(is_p1_turn, actions_p1, actions_p2)

        is_jump = (actions >= total_positions) & (actions < 2 * total_positions)
        jump_counts = jump_counts + (is_jump & ~terminated).astype(jnp.int32)

        states, terminated = _step_games_batched(states, actions, terminated, env_config)

    winners = states.winner
    per_game_turns = list(np.array(states.num_turns))

    if a_is_p1:
        a_wins = int(jnp.sum((winners == 1).astype(jnp.int32)))
        b_wins = int(jnp.sum((winners == 2).astype(jnp.int32)))
    else:
        a_wins = int(jnp.sum((winners == 2).astype(jnp.int32)))
        b_wins = int(jnp.sum((winners == 1).astype(jnp.int32)))

    draws = int(jnp.sum((winners == 0).astype(jnp.int32)))
    return a_wins, b_wins, draws, per_game_turns, list(np.array(jump_counts))


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
    """Sequential per-matchup tournament. Prefer run_tournament_megabatch."""
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
                params_a=params_a, params_b=params_b,
                network=network, env_config=env_config, rng=match_rng,
                games_per_perspective=games_per_perspective,
                batch_size=batch_size, num_simulations=num_simulations,
                max_moves=max_moves, recurrent_fn=recurrent_fn,
            )
            elapsed = time.time() - t0

            W[i, j] += a_wins; D[i, j] += draws; L[i, j] += b_wins
            W[j, i] += b_wins; D[j, i] += draws; L[j, i] += a_wins

            total = a_wins + b_wins + draws
            wr_a = a_wins / total if total > 0 else 0.0
            completed += 1

            print(f"  [{completed}/{total_matchups}] {name_a} vs {name_b}: "
                  f"{a_wins}W-{draws}D-{b_wins}L "
                  f"({wr_a:.0%} for {name_a}) "
                  f"avg_len={stats['avg_game_length']:.0f} "
                  f"avg_jumps={stats['avg_jumps_per_game']:.1f} "
                  f"[{elapsed:.1f}s]")

            match_results.append({
                "agent_a": name_a, "agent_b": name_b,
                **stats, "elapsed_s": round(elapsed, 1),
            })

    elo_ratings = compute_elo(names, W, D, L)

    print(f"\n{'='*50}")
    print("ELO Ratings:")
    for name in sorted(elo_ratings, key=elo_ratings.get, reverse=True):
        print(f"  {name}: {elo_ratings[name]:.0f}")
    print(f"{'='*50}")

    return match_results, elo_ratings


# ═══════════════════════════════════════════════════════════════
#  ELO computation (shared)
# ═══════════════════════════════════════════════════════════════

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
