"""
MCTS wrapper using mctx for Phutball AlphaZero.

This connects our JAX environment and neural network to mctx's
batched MCTS implementation.
"""

import jax
import jax.numpy as jnp
import mctx
from typing import NamedTuple, Tuple
from functools import partial

from phutball_env_jax import (
    PhutballState, EnvConfig, 
    reset, step, get_legal_actions,
    state_to_network_input,
    EMPTY, BALL, MAN, END_HI, END_LO
)
from network import PhutballNetwork, predict


class MCTSConfig(NamedTuple):
    """Configuration for MCTS."""
    num_simulations: int = 800
    max_num_considered_actions: int = 32  # For Gumbel MuZero
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25


def make_recurrent_fn(network: PhutballNetwork, env_config: EnvConfig):
    """
    Create the recurrent function for mctx.
    
    This function defines how MCTS simulates forward:
    1. Take an action from current state
    2. Get the next state from the real environment
    3. Evaluate the next state with the neural network
    
    Returns:
        recurrent_fn compatible with mctx
    """
    
    def recurrent_fn(params, rng, action, embedding):
        """
        Args:
            params: Dict with 'network_params' and 'batch_stats'
            rng: Random key (unused in deterministic env)
            action: Action to take, shape (batch_size,)
            embedding: Current PhutballState (batched)
            
        Returns:
            RecurrentFnOutput, new_embedding
        """
        network_params = params['network_params']
        batch_stats = params['batch_stats']
        
        # embedding is the batched PhutballState
        # We need to step each environment
        batched_step = jax.vmap(lambda s, a: step(s, a, env_config))
        next_states = batched_step(embedding, action)
        
        # Convert states to network input
        batched_to_input = jax.vmap(lambda s: state_to_network_input(s, env_config))
        network_inputs = batched_to_input(next_states)
        
        # Get network predictions
        variables = {'params': network_params, 'batch_stats': batch_stats}
        policy_logits, values = network.apply(variables, network_inputs, train=False)
        
        # Get legal action mask for next states
        batched_legal = jax.vmap(lambda s: get_legal_actions(s, env_config))
        legal_mask = batched_legal(next_states)
        
        # Mask illegal actions with large negative logits
        masked_logits = jnp.where(
            legal_mask == 1,
            policy_logits,
            jnp.float32(-1e9)
        )
        
        # Check for terminal states
        terminated = next_states.terminated
        
        # For terminal states, compute value from current_player's perspective.
        # During jumps (the only way to win), current_player is the player who just moved.
        # If winner == current_player, that player won.
        # If winner != current_player and != 0, that player lost (own goal).
        terminal_value = jnp.where(
            next_states.winner == next_states.current_player,
            1.0,   # Current player won
            jnp.where(next_states.winner == 0, 0.0, -1.0)  # Draw or opponent won
        )
        values = jnp.where(terminated, terminal_value, values)
        
        # Discount: 1 for ongoing, 0 for terminal
        discount = jnp.where(terminated, 0.0, 1.0)
        
        # Reward: 0 for AlphaZero (no intermediate rewards)
        reward = jnp.zeros_like(values)
        
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=masked_logits,
            value=values,
        )
        
        return recurrent_fn_output, next_states
    
    return recurrent_fn


def make_root_fn(network: PhutballNetwork, env_config: EnvConfig):
    """
    Create the root function for mctx.
    
    This evaluates the root state before MCTS begins.
    """
    
    def root_fn(params, rng, root_state):
        """
        Args:
            params: Dict with 'network_params' and 'batch_stats'  
            rng: Random key
            root_state: Batched PhutballState
            
        Returns:
            RootFnOutput with prior_logits, value, embedding
        """
        network_params = params['network_params']
        batch_stats = params['batch_stats']
        
        # Convert states to network input
        batched_to_input = jax.vmap(lambda s: state_to_network_input(s, env_config))
        network_inputs = batched_to_input(root_state)
        
        # Get network predictions
        variables = {'params': network_params, 'batch_stats': batch_stats}
        policy_logits, values = network.apply(variables, network_inputs, train=False)
        
        # Get legal action mask
        batched_legal = jax.vmap(lambda s: get_legal_actions(s, env_config))
        legal_mask = batched_legal(root_state)
        
        # Mask illegal actions
        masked_logits = jnp.where(
            legal_mask == 1,
            policy_logits,
            jnp.float32(-1e9)
        )
        
        root_fn_output = mctx.RootFnOutput(
            prior_logits=masked_logits,
            value=values,
            embedding=root_state,  # The state itself is the embedding
        )
        
        return root_fn_output
    
    return root_fn


def run_mcts(
    params: dict,
    rng: jnp.ndarray,
    root_state: PhutballState,
    network: PhutballNetwork,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run MCTS from a root state and return improved policy.
    
    Args:
        params: Dict with 'network_params' and 'batch_stats'
        rng: Random key
        root_state: Single PhutballState (will be batched internally)
        network: PhutballNetwork instance
        env_config: Environment config
        mcts_config: MCTS config
        
    Returns:
        action_weights: Improved policy from MCTS visit counts
        root_value: Value estimate at root
    """
    # Batch the single state (mctx expects batched input)
    batched_state = jax.tree.map(lambda x: x[None, ...], root_state)
    
    root_fn = make_root_fn(network, env_config)
    recurrent_fn = make_recurrent_fn(network, env_config)
    
    # Run MCTS
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng,
        root=root_fn(params, rng, batched_state),
        recurrent_fn=recurrent_fn,
        num_simulations=mcts_config.num_simulations,
        max_num_considered_actions=mcts_config.max_num_considered_actions,
        gumbel_scale=1.0,
    )
    
    # Extract results (unbatch)
    action_weights = policy_output.action_weights[0]  # (action_space_size,)
    root_value = policy_output.search_tree.node_values[0, 0]  # Root value
    
    return action_weights, root_value


def select_action(
    params: dict,
    rng: jnp.ndarray,
    state: PhutballState,
    network: PhutballNetwork,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
    temperature: float = 1.0,
) -> Tuple[int, jnp.ndarray, float]:
    """
    Select an action using MCTS.
    
    Args:
        params: Network parameters
        rng: Random key
        state: Current game state
        network: Neural network
        env_config: Environment config
        mcts_config: MCTS config
        temperature: Sampling temperature (0 = greedy, 1 = proportional to visits)
        
    Returns:
        action: Selected action
        policy: MCTS policy (visit counts normalized)
        value: Root value estimate
    """
    rng, mcts_rng, sample_rng = jax.random.split(rng, 3)
    
    # Run MCTS
    action_weights, root_value = run_mcts(
        params, mcts_rng, state, network, env_config, mcts_config
    )
    
    # Apply temperature
    if temperature < 0.01:
        # Greedy
        action = jnp.argmax(action_weights)
    else:
        # Sample proportionally to visit counts
        # Raise to power of 1/temperature
        weights_temp = jnp.power(action_weights + 1e-8, 1.0 / temperature)
        weights_temp = weights_temp / jnp.sum(weights_temp)
        action = jax.random.choice(sample_rng, len(action_weights), p=weights_temp)
    
    return int(action), action_weights, float(root_value)


# ============================================================================
# Tests
# ============================================================================

def test_mcts_basic():
    """Test that MCTS runs without crashing."""
    from network import create_network, init_network
    
    # Setup
    env_config = EnvConfig(rows=9, cols=9)  # Small board for fast test
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    mcts_config = MCTSConfig(num_simulations=8, max_num_considered_actions=16)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Create initial state
    state = reset(env_config)
    
    # Run MCTS
    rng, mcts_rng = jax.random.split(rng)
    action_weights, root_value = run_mcts(
        params, mcts_rng, state, network, env_config, mcts_config
    )
    
    # Check outputs
    action_space_size = 2 * 9 * 9 + 1  # 163
    assert action_weights.shape == (action_space_size,), \
        f"Action weights shape: {action_weights.shape}"
    assert jnp.isfinite(root_value), f"Root value not finite: {root_value}"
    
    # Action weights should sum to ~1 (they're normalized visit counts)
    weight_sum = jnp.sum(action_weights)
    assert jnp.isclose(weight_sum, 1.0, atol=0.01), f"Weights sum to {weight_sum}"
    
    print(f"✓ MCTS basic test passed")
    print(f"  Action weights shape: {action_weights.shape}")
    print(f"  Root value: {root_value:.4f}")
    print(f"  Top 5 actions: {jnp.argsort(action_weights)[-5:][::-1]}")


def test_select_action():
    """Test action selection with MCTS."""
    from network import create_network, init_network
    
    # Setup
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    mcts_config = MCTSConfig(num_simulations=8, max_num_considered_actions=16)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Create initial state
    state = reset(env_config)
    
    # Select action
    rng, action_rng = jax.random.split(rng)
    action, policy, value = select_action(
        params, action_rng, state, network, env_config, mcts_config, temperature=1.0
    )
    
    # Check action is valid
    legal_mask = get_legal_actions(state, env_config)
    assert legal_mask[action] == 1, f"Selected illegal action {action}"
    
    print(f"✓ Select action test passed")
    print(f"  Selected action: {action}")
    print(f"  Value estimate: {value:.4f}")


def test_play_game():
    """Test playing a short game with MCTS."""
    from network import create_network, init_network
    
    # Setup - small board, few simulations for speed
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    mcts_config = MCTSConfig(num_simulations=4, max_num_considered_actions=8)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=6)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Play a few moves
    state = reset(env_config)
    num_moves = 10
    
    print(f"Playing {num_moves} moves...")
    for i in range(num_moves):
        rng, action_rng = jax.random.split(rng)
        action, policy, value = select_action(
            params, action_rng, state, network, env_config, mcts_config, temperature=1.0
        )
        
        state = step(state, jnp.array(action, dtype=jnp.int32), env_config)
        print(f"  Move {i+1}: action={action}, value={value:.3f}, player={int(state.current_player)}, terminated={bool(state.terminated)}")
        
        if state.terminated:
            print(f"  Game ended! Winner: {int(state.winner)}")
            break
    
    print(f"✓ Play game test passed")


if __name__ == "__main__":
    print("Testing MCTS...\n")
    
    test_mcts_basic()
    print()
    
    test_select_action()
    print()
    
    test_play_game()
    
    print("\n✓ All MCTS tests passed!")