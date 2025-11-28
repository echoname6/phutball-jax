"""
Round-robin evaluation between multiple Phutball AlphaZero checkpoints.

Usage in Colab (from repo root):

    %cd /content/phutball-jax

    from evaluate_round_robin import run_round_robin

    checkpoint_ids = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]
    run_round_robin(
        checkpoint_ids,
        checkpoint_dir="checkpoints",
        num_simulations=128,   # 128–256 is a good eval range
        games_per_color=2,     # 2 as P1 + 2 as P2 = 4 games per pair
        max_moves=1024,
    )
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

# Ensure TrainConfig exists for unpickling checkpoints
import train_batched  # noqa: F401

from phutball_env_jax import EnvConfig, reset, step
from network import create_network
from mcts import MCTSConfig, select_action


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    params: dict  # {'network_params': ..., 'batch_stats': ...}


def load_agent(
    checkpoint_path: str,
    rows: int,
    cols: int,
    num_channels: int,
    num_res_blocks: int,
) -> Tuple[Agent, object]:
    """
    Load a single agent from a checkpoint.

    Returns:
        (agent, network) but the same network instance can be reused
        across multiple agents (only params differ).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    # ckpt has 'params' and 'batch_stats' from training
    network_params = ckpt["params"]
    batch_stats = ckpt["batch_stats"]

    params = {
        "network_params": network_params,
        "batch_stats": batch_stats,
    }

    agent = Agent(
        name=os.path.basename(checkpoint_path),
        params=params,
    )

    # Network architecture must match training
    network = create_network(
        rows=rows,
        cols=cols,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
    )

    return agent, network


# ---------------------------------------------------------------------------
# Single-game and match logic
# ---------------------------------------------------------------------------

def play_game(
    agent_p1: Agent,
    agent_p2: Agent,
    network,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
    max_moves: int = 1024,
    temperature: float = 0.0,
    rng: jnp.ndarray | None = None,
) -> float:
    """
    Play a single game between two agents.

    Args:
        agent_p1: Agent controlling player 1 (state.current_player == 1)
        agent_p2: Agent controlling player 2 (state.current_player == 2)
        network: Shared PhutballNetwork instance
        env_config: EnvConfig
        mcts_config: MCTSConfig (num_simulations, etc.)
        max_moves: hard cap on ply to avoid infinite marathons
        temperature: MCTS sampling temperature (0.0 = greedy)
        rng: JAX random key

    Returns:
        result_from_p1_perspective:
            +1.0 if player 1 wins
            -1.0 if player 2 wins
             0.0 for draw / max-move cutoff
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Fresh game
    state = reset(env_config)
    move_count = 0

    while (not bool(state.terminated)) and (move_count < max_moves):
        rng, action_rng = jax.random.split(rng)

        current_player = int(state.current_player)
        agent = agent_p1 if current_player == 1 else agent_p2

        action, policy, value = select_action(
            agent.params,
            action_rng,
            state,
            network,
            env_config,
            mcts_config,
            temperature=temperature,
        )

        # apply move
        state = step(state, jnp.array(action, dtype=jnp.int32), env_config)
        move_count += 1

    winner = int(state.winner)
    if winner == 0 or move_count >= max_moves:
        return 0.0
    elif winner == 1:
        return 1.0
    else:
        # winner == 2
        return -1.0


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    network,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
    games_per_color: int = 2,
    max_moves: int = 1024,
    temperature: float = 0.0,
    base_seed: int = 0,
) -> Tuple[float, int]:
    """
    Play a small match between A and B.

    games_per_color:
        - Number of games where A is player 1 and B is player 2
        - Same number of games where B is player 1 and A is player 2

    Returns:
        (score_for_A, total_games)

        Each game:
            +1 if A wins
            -1 if A loses
             0 on draw / cutoff
    """
    score_a = 0.0
    total_games = 0

    rng = jax.random.PRNGKey(base_seed)

    # A as player 1
    for g in range(games_per_color):
        rng, game_rng = jax.random.split(rng)
        res = play_game(
            agent_p1=agent_a,
            agent_p2=agent_b,
            network=network,
            env_config=env_config,
            mcts_config=mcts_config,
            max_moves=max_moves,
            temperature=temperature,
            rng=game_rng,
        )
        score_a += res
        total_games += 1

    # B as player 1 (so result we get is from B's perspective)
    for g in range(games_per_color):
        rng, game_rng = jax.random.split(rng)
        res_b = play_game(
            agent_p1=agent_b,
            agent_p2=agent_a,
            network=network,
            env_config=env_config,
            mcts_config=mcts_config,
            max_moves=max_moves,
            temperature=temperature,
            rng=game_rng,
        )
        # Flip perspective back to A
        score_a -= res_b
        total_games += 1

    return score_a, total_games


# ---------------------------------------------------------------------------
# Round-robin driver
# ---------------------------------------------------------------------------

def run_round_robin(
    checkpoint_ids: List[int],
    checkpoint_dir: str = "checkpoints",
    rows: int = 21,
    cols: int = 15,
    num_channels: int = 64,
    num_res_blocks: int = 8,
    num_simulations: int = 128,
    games_per_color: int = 2,
    max_moves: int = 1024,
    temperature: float = 0.0,
) -> Tuple[Dict[str, Dict], Dict[Tuple[str, str], Tuple[float, int]]]:
    """
    Run a round-robin tournament over a list of checkpoint indices.

    Args:
        checkpoint_ids: e.g. [9, 19, 29, 39, 49, ...]
        checkpoint_dir: directory where checkpoint_XXXXXX.pkl live
        rows, cols, num_channels, num_res_blocks: must match training
        num_simulations: MCTS simulations per move for evaluation
        games_per_color: games per color per pair (total = 2 * games_per_color)
        max_moves: game cutoff
        temperature: MCTS temperature (0.0 = greedy, 1.0 = sample by visits)

    Returns:
        (results, head_to_head)
        results[name] = {"points": float, "games": int}
        head_to_head[(name_a, name_b)] = (score_for_a, games_played)
    """
    # Shared env/network/MCTS config
    env_config = EnvConfig(rows=rows, cols=cols)
    mcts_config = MCTSConfig(
        num_simulations=num_simulations,
        max_num_considered_actions=32,
        dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )

    # Load agents
    agents: List[Agent] = []
    network = None

    for idx in checkpoint_ids:
        ckpt_name = f"checkpoint_{idx:06d}.pkl"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing {ckpt_path}, skipping")
            continue

        agent, net = load_agent(
            ckpt_path,
            rows=rows,
            cols=cols,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
        )
        agents.append(agent)
        if network is None:
            network = net

    if network is None or not agents:
        raise RuntimeError("No checkpoints loaded; check paths / IDs.")

    # Init result structures
    results: Dict[str, Dict[str, float | int]] = {
        a.name: {"points": 0.0, "games": 0} for a in agents
    }
    head_to_head: Dict[Tuple[str, str], Tuple[float, int]] = {}

    # Round-robin: each unique pair
    seed_counter = 0
    for i, agent_a in enumerate(agents):
        for j in range(i + 1, len(agents)):
            agent_b = agents[j]
            base_seed = seed_counter
            seed_counter += 1

            score_a, games = play_match(
                agent_a,
                agent_b,
                network=network,
                env_config=env_config,
                mcts_config=mcts_config,
                games_per_color=games_per_color,
                max_moves=max_moves,
                temperature=temperature,
                base_seed=base_seed,
            )

            results[agent_a.name]["points"] += score_a
            results[agent_a.name]["games"] += games

            results[agent_b.name]["points"] -= score_a
            results[agent_b.name]["games"] += games

            head_to_head[(agent_a.name, agent_b.name)] = (score_a, games)

            print(
                f"{agent_a.name} vs {agent_b.name}: "
                f"score_for_A={score_a:.1f} over {games} games"
            )

    # Overall standings
    print("\n=== Overall standings (win=+1, loss=-1, draw=0) ===")
    def avg_score(name: str) -> float:
        g = results[name]["games"]
        return results[name]["points"] / g if g > 0 else 0.0

    for agent in sorted(agents, key=lambda a: avg_score(a.name), reverse=True):
        name = agent.name
        pts = results[name]["points"]
        g = results[name]["games"]
        avg = avg_score(name)
        print(f"{name}: {pts:+.1f} over {g} games (avg {avg:+.3f})")

    return results, head_to_head
