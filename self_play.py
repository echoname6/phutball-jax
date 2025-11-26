"""
Self-play for Phutball AlphaZero.

Plays games using MCTS, collects training data.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np

from phutball_env_jax import (
    PhutballState, EnvConfig,
    reset, step, get_legal_actions,
)
from network import PhutballNetwork, init_network, create_network
from mcts import MCTSConfig, select_action, state_to_network_input


class TrainingExample(NamedTuple):
    """Single training example."""
    state: np.ndarray          # (channels, rows, cols)
    policy: np.ndarray         # (action_space_size,)
    value: float               # Game outcome from this player's perspective


@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""
    # Temperature schedule
    temp_threshold: int = 15        # Use temp=1 for first N moves, then temp→0
    temp_high: float = 1.0          # Temperature for exploration phase
    temp_low: float = 0.1           # Temperature for exploitation phase
    
    # Game limits
    max_moves: int = 7200           # Max moves per game (safety cap, not expected to hit)
    
    # MCTS settings (inherited)
    mcts_config: MCTSConfig = None
    
    def __post_init__(self):
        if self.mcts_config is None:
            self.mcts_config = MCTSConfig()


def play_game(
    params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    self_play_config: SelfPlayConfig,
) -> List[TrainingExample]:
    """
    Play a single game of self-play.
    
    Args:
        params: Network parameters
        rng: Random key
        network: Neural network
        env_config: Environment config
        self_play_config: Self-play config
        
    Returns:
        List of TrainingExample tuples
    """
    state = reset(env_config)
    
    # Collect game history: (state_array, policy, player_at_turn)
    history = []
    move_count = 0
    
    while not state.terminated and move_count < self_play_config.max_moves:
        rng, action_rng = jax.random.split(rng)
        
        # Temperature schedule
        if move_count < self_play_config.temp_threshold:
            temperature = self_play_config.temp_high
        else:
            temperature = self_play_config.temp_low
        
        # Get MCTS policy
        action, policy, value = select_action(
            params,
            action_rng,
            state,
            network,
            env_config,
            self_play_config.mcts_config,
            temperature=temperature,
        )
        
        # Store state and policy
        state_array = np.array(state_to_network_input(state, env_config))
        player = int(state.current_player)
        history.append((state_array, np.array(policy), player))
        
        # Take action
        state = step(state, jnp.array(action, dtype=jnp.int32), env_config)
        move_count += 1
    
    # Determine game outcome
    winner = int(state.winner)
    
    if winner == 0:
        # Draw (max moves reached)
        outcome_p1 = 0.0
        outcome_p2 = 0.0
    elif winner == 1:
        outcome_p1 = 1.0
        outcome_p2 = -1.0
    else:  # winner == 2
        outcome_p1 = -1.0
        outcome_p2 = 1.0
    
    # Convert history to training examples with correct value signs
    examples = []
    for state_array, policy, player in history:
        if player == 1:
            value = outcome_p1
        else:
            value = outcome_p2
        
        examples.append(TrainingExample(
            state=state_array,
            policy=policy,
            value=value,
        ))
    
    return examples


def play_games(
    params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    self_play_config: SelfPlayConfig,
    num_games: int,
) -> List[TrainingExample]:
    """
    Play multiple games of self-play.
    
    Args:
        params: Network parameters
        rng: Random key
        network: Neural network
        env_config: Environment config
        self_play_config: Self-play config
        num_games: Number of games to play
        
    Returns:
        List of all TrainingExamples from all games
    """
    all_examples = []
    
    for i in range(num_games):
        rng, game_rng = jax.random.split(rng)
        examples = play_game(
            params, game_rng, network, env_config, self_play_config
        )
        all_examples.extend(examples)
        
        # Simple progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Played {i + 1}/{num_games} games, {len(all_examples)} examples")
    
    return all_examples


class ReplayBuffer:
    """Simple replay buffer for training examples."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: List[TrainingExample] = []
    
    def add(self, examples: List[TrainingExample]):
        """Add examples to buffer."""
        self.buffer.extend(examples)
        
        # Remove oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def sample(self, batch_size: int) -> dict:
        """Sample a batch for training."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        states = np.stack([self.buffer[i].state for i in indices])
        policies = np.stack([self.buffer[i].policy for i in indices])
        values = np.array([self.buffer[i].value for i in indices])
        
        return {
            'states': jnp.array(states),
            'policy_targets': jnp.array(policies),
            'value_targets': jnp.array(values),
        }
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Tests
# ============================================================================

def test_play_single_game():
    """Test playing a single self-play game."""
    # Small board, few simulations for speed
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    
    mcts_config = MCTSConfig(num_simulations=4, max_num_considered_actions=8)
    self_play_config = SelfPlayConfig(
        temp_threshold=5,
        max_moves=50,
        mcts_config=mcts_config,
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Play game
    rng, game_rng = jax.random.split(rng)
    examples = play_game(params, game_rng, network, env_config, self_play_config)
    
    print(f"✓ Single game test passed")
    print(f"  Game length: {len(examples)} moves")
    print(f"  Final value: {examples[-1].value}")
    
    # Check example format
    ex = examples[0]
    assert ex.state.shape == (6, 9, 9), f"State shape: {ex.state.shape}"
    assert ex.policy.shape == (163,), f"Policy shape: {ex.policy.shape}"
    assert -1 <= ex.value <= 1, f"Value: {ex.value}"
    
    # Check policy sums to ~1
    assert np.isclose(np.sum(ex.policy), 1.0, atol=0.01), f"Policy sum: {np.sum(ex.policy)}"


def test_replay_buffer():
    """Test replay buffer operations."""
    buffer = ReplayBuffer(max_size=100)
    
    # Add some fake examples
    action_space = 163
    examples = [
        TrainingExample(
            state=np.random.randn(6, 9, 9).astype(np.float32),
            policy=np.ones(action_space, dtype=np.float32) / action_space,
            value=np.random.uniform(-1, 1),
        )
        for _ in range(50)
    ]
    
    buffer.add(examples)
    assert len(buffer) == 50, f"Buffer length: {len(buffer)}"
    
    # Sample a batch
    batch = buffer.sample(batch_size=16)
    assert batch['states'].shape == (16, 6, 9, 9), f"States shape: {batch['states'].shape}"
    assert batch['policy_targets'].shape == (16, 163), f"Policy shape: {batch['policy_targets'].shape}"
    assert batch['value_targets'].shape == (16,), f"Values shape: {batch['value_targets'].shape}"
    
    # Test overflow
    buffer.add(examples)  # Add 50 more
    assert len(buffer) == 100, f"Buffer should be capped at 100, got {len(buffer)}"
    
    buffer.add(examples)  # Add 50 more, should remove oldest
    assert len(buffer) == 100, f"Buffer should still be 100, got {len(buffer)}"
    
    print(f"✓ Replay buffer test passed")


def test_play_multiple_games():
    """Test playing multiple games."""
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    
    mcts_config = MCTSConfig(num_simulations=4, max_num_considered_actions=8)
    self_play_config = SelfPlayConfig(
        temp_threshold=5,
        max_moves=30,
        mcts_config=mcts_config,
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Play games
    print("Playing 3 games...")
    rng, games_rng = jax.random.split(rng)
    examples = play_games(
        params, games_rng, network, env_config, self_play_config, num_games=3
    )
    
    print(f"✓ Multiple games test passed")
    print(f"  Total examples: {len(examples)}")


if __name__ == "__main__":
    print("Testing Self-Play...\n")
    
    test_replay_buffer()
    print()
    
    print("Playing single game (may take a moment)...")
    test_play_single_game()
    print()
    
    test_play_multiple_games()
    
    print("\n✓ All self-play tests passed!")