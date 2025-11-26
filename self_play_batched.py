"""
Batched Self-play for Phutball AlphaZero.

Runs multiple games in parallel using vmap for TPU efficiency.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple
from functools import partial
import numpy as np

from phutball_env_jax import (
    PhutballState, EnvConfig,
    reset, step, get_legal_actions,
)
from network import PhutballNetwork, predict
from mcts import state_to_network_input


class GameState(NamedTuple):
    """State for a batch of games being played."""
    env_states: PhutballState    # Batched environment states
    terminated: jnp.ndarray      # (batch,) which games are done
    move_count: jnp.ndarray      # (batch,) moves played per game
    

class TrajectoryData(NamedTuple):
    """Collected trajectory data from games."""
    states: jnp.ndarray          # (batch, max_moves, channels, rows, cols)
    policies: jnp.ndarray        # (batch, max_moves, action_space)
    players: jnp.ndarray         # (batch, max_moves) which player moved
    valid_mask: jnp.ndarray      # (batch, max_moves) which moves are real
    winners: jnp.ndarray         # (batch,) game outcomes


def batched_reset(env_config: EnvConfig, batch_size: int) -> PhutballState:
    """Reset multiple environments in parallel."""
    single_state = reset(env_config)
    # Broadcast to batch
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape).copy(),
        single_state
    )


def make_batched_step(env_config: EnvConfig):
    """Create batched step function."""
    @jax.jit
    def batched_step(states: PhutballState, actions: jnp.ndarray) -> PhutballState:
        return jax.vmap(lambda s, a: step(s, a, env_config))(states, actions)
    return batched_step


def make_batched_legal_actions(env_config: EnvConfig):
    """Create batched legal actions function."""
    @jax.jit
    def batched_legal(states: PhutballState) -> jnp.ndarray:
        return jax.vmap(lambda s: get_legal_actions(s, env_config))(states)
    return batched_legal


def make_batched_network_input(env_config: EnvConfig):
    """Create batched state-to-input conversion."""
    @jax.jit
    def batched_to_input(states: PhutballState) -> jnp.ndarray:
        return jax.vmap(lambda s: state_to_network_input(s, env_config))(states)
    return batched_to_input


@partial(jax.jit, static_argnums=(3, 4, 5))
def batched_network_policy(
    params: dict,
    states: PhutballState,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get policy from network for a batch of states.
    
    For speed, we skip MCTS and just use the raw network policy.
    This is faster for data generation; MCTS quality comes from training.
    
    Returns:
        actions: (batch,) selected actions
        policies: (batch, action_space) policy distributions
        values: (batch,) value estimates
    """
    batch_size = states.board.shape[0]
    
    # Convert states to network input
    batched_to_input = make_batched_network_input(env_config)
    network_inputs = batched_to_input(states)
    
    # Get network predictions
    variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
    policy_logits, values = network.apply(variables, network_inputs, train=False)
    
    # Get legal action mask
    batched_legal = make_batched_legal_actions(env_config)
    legal_mask = batched_legal(states)
    
    # Mask illegal actions
    masked_logits = jnp.where(
        legal_mask == 1,
        policy_logits,
        jnp.float32(-1e9)
    )
    
    # Convert to probabilities
    policies = jax.nn.softmax(masked_logits / temperature)
    
    # Sample actions
    rngs = jax.random.split(rng, batch_size)
    actions = jax.vmap(lambda r, p: jax.random.choice(r, len(p), p=p))(rngs, policies)
    
    return actions, policies, values


def play_games_batched(
    params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    batch_size: int = 64,
    max_moves: int = 500,
    temperature: float = 1.0,
) -> TrajectoryData:
    """
    Play multiple games in parallel.
    
    Args:
        params: Network parameters
        rng: Random key
        network: Neural network
        env_config: Environment config
        batch_size: Number of games to play in parallel
        max_moves: Maximum moves per game
        temperature: Sampling temperature
        
    Returns:
        TrajectoryData with all game trajectories
    """
    action_space_size = 2 * env_config.rows * env_config.cols + 1
    num_channels = 6
    
    # Initialize storage for trajectories
    all_states = jnp.zeros((batch_size, max_moves, num_channels, env_config.rows, env_config.cols))
    all_policies = jnp.zeros((batch_size, max_moves, action_space_size))
    all_players = jnp.zeros((batch_size, max_moves), dtype=jnp.int32)
    valid_mask = jnp.zeros((batch_size, max_moves), dtype=jnp.bool_)
    
    # Initialize game states
    env_states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
    move_count = jnp.zeros(batch_size, dtype=jnp.int32)
    
    # Get batched functions
    batched_step = make_batched_step(env_config)
    batched_to_input = make_batched_network_input(env_config)
    
    # Play loop
    def game_step(carry, _):
        env_states, terminated, move_count, all_states, all_policies, all_players, valid_mask, rng = carry
        
        rng, step_rng = jax.random.split(rng)
        
        # Get current observations
        current_obs = batched_to_input(env_states)
        current_players = env_states.current_player
        
        # Get actions from network
        actions, policies, values = batched_network_policy(
            params, env_states, step_rng, network, env_config, temperature
        )
        
        # Store data (only for non-terminated games)
        active = ~terminated
        
        # Update storage using dynamic indexing
        def update_at_move(storage, new_data, move_idx, active_mask):
            # For each batch element, update at its move_idx if active
            batch_indices = jnp.arange(batch_size)
            return storage.at[batch_indices, move_idx].set(
                jnp.where(active_mask[:, None] if new_data.ndim > 1 else active_mask, 
                         new_data, 
                         storage[batch_indices, move_idx])
            )
        
        # Simpler approach: use a fixed move index for all
        # This wastes some space but is simpler
        move_idx = move_count  # (batch,) array
        
        # Update each game's trajectory at its current move index
        all_states = jax.vmap(lambda s, obs, idx, act: 
            s.at[idx].set(jnp.where(act, obs, s[idx]))
        )(all_states, current_obs, move_idx, active)
        
        all_policies = jax.vmap(lambda s, pol, idx, act:
            s.at[idx].set(jnp.where(act, pol, s[idx]))
        )(all_policies, policies, move_idx, active)
        
        all_players = jax.vmap(lambda s, pl, idx, act:
            s.at[idx].set(jnp.where(act, pl, s[idx]))
        )(all_players, current_players, move_idx, active)
        
        valid_mask = jax.vmap(lambda s, idx, act:
            s.at[idx].set(act)
        )(valid_mask, move_idx, active)
        
        # Step environments
        new_env_states = batched_step(env_states, actions)
        
        # Update termination status
        new_terminated = terminated | new_env_states.terminated
        new_move_count = move_count + active.astype(jnp.int32)
        
        carry = (new_env_states, new_terminated, new_move_count, 
                all_states, all_policies, all_players, valid_mask, rng)
        return carry, None
    
    # Run the game loop
    initial_carry = (env_states, terminated, move_count, 
                    all_states, all_policies, all_players, valid_mask, rng)
    
    final_carry, _ = lax.scan(game_step, initial_carry, None, length=max_moves)
    
    (final_env_states, _, _, 
     all_states, all_policies, all_players, valid_mask, _) = final_carry
    
    # Get winners
    winners = final_env_states.winner
    
    return TrajectoryData(
        states=all_states,
        policies=all_policies,
        players=all_players,
        valid_mask=valid_mask,
        winners=winners,
    )


def trajectory_to_training_examples(
    trajectory: TrajectoryData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert trajectory data to training examples.
    
    Returns:
        states: (N, channels, rows, cols)
        policies: (N, action_space)
        values: (N,)
    """
    batch_size, max_moves = trajectory.valid_mask.shape
    
    all_states = []
    all_policies = []
    all_values = []
    
    # Convert to numpy for easier manipulation
    states_np = np.array(trajectory.states)
    policies_np = np.array(trajectory.policies)
    players_np = np.array(trajectory.players)
    valid_np = np.array(trajectory.valid_mask)
    winners_np = np.array(trajectory.winners)
    
    for game_idx in range(batch_size):
        winner = winners_np[game_idx]
        
        for move_idx in range(max_moves):
            if not valid_np[game_idx, move_idx]:
                continue
                
            state = states_np[game_idx, move_idx]
            policy = policies_np[game_idx, move_idx]
            player = players_np[game_idx, move_idx]
            
            # Value from this player's perspective
            if winner == 0:
                value = 0.0  # Draw
            elif winner == player:
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            all_states.append(state)
            all_policies.append(policy)
            all_values.append(value)
    
    if len(all_states) == 0:
        # Return empty arrays with correct shapes
        channels, rows, cols = states_np.shape[2:]
        action_space = policies_np.shape[2]
        return (
            np.zeros((0, channels, rows, cols), dtype=np.float32),
            np.zeros((0, action_space), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    
    return (
        np.stack(all_states).astype(np.float32),
        np.stack(all_policies).astype(np.float32),
        np.array(all_values, dtype=np.float32),
    )


class ReplayBuffer:
    """Simple replay buffer for training examples."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.states = None
        self.policies = None
        self.values = None
        self.size = 0
        self.idx = 0
    
    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """Add examples to buffer."""
        n = len(states)
        if n == 0:
            return
            
        if self.states is None:
            # Initialize buffers
            self.states = np.zeros((self.max_size,) + states.shape[1:], dtype=np.float32)
            self.policies = np.zeros((self.max_size,) + policies.shape[1:], dtype=np.float32)
            self.values = np.zeros(self.max_size, dtype=np.float32)
        
        # Add data
        if self.idx + n <= self.max_size:
            self.states[self.idx:self.idx + n] = states
            self.policies[self.idx:self.idx + n] = policies
            self.values[self.idx:self.idx + n] = values
            self.idx += n
        else:
            # Wrap around
            first_part = self.max_size - self.idx
            self.states[self.idx:] = states[:first_part]
            self.policies[self.idx:] = policies[:first_part]
            self.values[self.idx:] = values[:first_part]
            
            second_part = n - first_part
            self.states[:second_part] = states[first_part:]
            self.policies[:second_part] = policies[first_part:]
            self.values[:second_part] = values[first_part:]
            self.idx = second_part
        
        self.size = min(self.size + n, self.max_size)
    
    def sample(self, batch_size: int) -> dict:
        """Sample a batch for training."""
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        return {
            'states': jnp.array(self.states[indices]),
            'policy_targets': jnp.array(self.policies[indices]),
            'value_targets': jnp.array(self.values[indices]),
        }
    
    def __len__(self):
        return self.size


# ============================================================================
# Tests
# ============================================================================

def test_batched_reset():
    """Test batched environment reset."""
    env_config = EnvConfig(rows=9, cols=9)
    batch_size = 8
    
    states = batched_reset(env_config, batch_size)
    
    assert states.board.shape == (batch_size, 9, 9), f"Board shape: {states.board.shape}"
    assert states.ball_pos.shape == (batch_size, 2), f"Ball pos shape: {states.ball_pos.shape}"
    
    print("✓ Batched reset works")


def test_batched_games():
    """Test playing batched games."""
    from network import create_network, init_network
    
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Play games
    rng, game_rng = jax.random.split(rng)
    
    import time
    start = time.time()
    
    trajectory = play_games_batched(
        params=params,
        rng=game_rng,
        network=network,
        env_config=env_config,
        batch_size=16,
        max_moves=50,
        temperature=1.0,
    )
    
    elapsed = time.time() - start
    
    print(f"✓ Batched games completed in {elapsed:.2f}s")
    print(f"  States shape: {trajectory.states.shape}")
    print(f"  Winners: {trajectory.winners}")
    print(f"  Valid moves per game: {trajectory.valid_mask.sum(axis=1)}")
    
    # Convert to training examples
    states, policies, values = trajectory_to_training_examples(trajectory)
    print(f"  Training examples: {len(states)}")


def test_replay_buffer():
    """Test the new replay buffer."""
    buffer = ReplayBuffer(max_size=100)
    
    # Add some data
    states = np.random.randn(50, 6, 9, 9).astype(np.float32)
    policies = np.random.rand(50, 163).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = np.random.uniform(-1, 1, 50).astype(np.float32)
    
    buffer.add(states, policies, values)
    assert len(buffer) == 50
    
    # Sample
    batch = buffer.sample(16)
    assert batch['states'].shape == (16, 6, 9, 9)
    
    # Add more (should wrap)
    buffer.add(states, policies, values)
    assert len(buffer) == 100
    
    print("✓ Replay buffer works")


def benchmark_batched_games():
    """Benchmark batched game speed."""
    from network import create_network, init_network
    
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Warmup
    rng, game_rng = jax.random.split(rng)
    _ = play_games_batched(params, game_rng, network, env_config, batch_size=8, max_moves=10)
    
    # Benchmark
    import time
    
    batch_sizes = [16, 64, 256]
    max_moves = 100
    
    print("\nBenchmark: Batched Self-Play")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        rng, game_rng = jax.random.split(rng)
        
        start = time.time()
        trajectory = play_games_batched(
            params, game_rng, network, env_config,
            batch_size=batch_size, max_moves=max_moves
        )
        # Force computation
        _ = trajectory.winners.block_until_ready()
        elapsed = time.time() - start
        
        total_moves = int(trajectory.valid_mask.sum())
        moves_per_sec = total_moves / elapsed
        games_per_sec = batch_size / elapsed
        
        print(f"  Batch {batch_size}: {elapsed:.2f}s, {games_per_sec:.1f} games/sec, {moves_per_sec:.0f} moves/sec")


if __name__ == "__main__":
    print("Testing Batched Self-Play...\n")
    
    test_batched_reset()
    test_replay_buffer()
    test_batched_games()
    benchmark_batched_games()
    
    print("\n✓ All tests passed!")