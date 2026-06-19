"""
Batched Self-play for Phutball AlphaZero.
Runs multiple games in parallel using vmap for TPU efficiency.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Tuple, Dict
from functools import partial
import numpy as np

from phutball_env_jax import (
    PhutballState, EnvConfig,
    reset, step, get_legal_actions,
    state_to_network_input,
    calculate_jumped_men,
    MAX_JUMP_SEQUENCE_LENGTH,
    MAN, BALL, EMPTY, END_HI, END_LO
)
from network import PhutballNetwork, PhutballTransformer, predict, predict_transformer

import mctx


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
    actions: jnp.ndarray         # (batch, max_moves) chosen action ids


def transform_policy_for_p2(policy_logits: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
    """
    Transform policy logits from visual coords (P2's 180° rotated view) back to physical coords.

    The network input is 180° rotated for P2, so the policy outputs are in rotated coords.
    We need to unrotate them to match the physical board.

    Action space:
    - [0, rows*cols): placement actions - position (a // cols, a % cols)
    - [rows*cols, 2*rows*cols): jump actions - jump to (a-N) // cols, (a-N) % cols)
    - [2*rows*cols]: halt action (no transformation needed)

    For 180° rotation, position (r, c) -> (rows-1-r, cols-1-c)
    This is equivalent to reversing the flattened index within each action type.
    """
    N = rows * cols

    # Split into placement, jump, halt
    placement_logits = policy_logits[..., :N]
    jump_logits = policy_logits[..., N:2*N]
    halt_logits = policy_logits[..., 2*N:]

    # Reverse placement and jump indices (180° rotation = reverse flattened order)
    physical_placement = jnp.flip(placement_logits, axis=-1)
    physical_jump = jnp.flip(jump_logits, axis=-1)

    return jnp.concatenate([physical_placement, physical_jump, halt_logits], axis=-1)


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


def make_mcts_recurrent_fn(network: PhutballNetwork, env_config: EnvConfig):
    """Create recurrent function for mctx that uses real env dynamics."""

    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    def recurrent_fn(params, rng, action, embedding):
        """
        Args:
            params: Network params dict
            rng: Random key (unused - env is deterministic)
            action: Actions to take, shape (batch_size,)
            embedding: Current PhutballState (batched)

        Returns:
            RecurrentFnOutput, new_embedding
        """
        # Step the environment
        def single_step(state, act):
            return step(state, act, env_config)

        next_states = jax.vmap(single_step)(embedding, action)

        # Convert to network input (NOTE: this rotates 180° for P2)
        network_inputs = jax.vmap(lambda s: state_to_network_input(s, env_config))(next_states)

        # Get network predictions
        variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
        policy_logits, values = network.apply(variables, network_inputs, train=False)

        # Transform policy logits back to physical coords for P2
        # The network input was rotated 180° for P2, so policy is in rotated coords
        is_p2 = (next_states.current_player == 2)[:, None]  # (batch, 1) for broadcasting
        policy_logits_p2 = transform_policy_for_p2(policy_logits, rows, cols)
        policy_logits = jnp.where(is_p2, policy_logits_p2, policy_logits)

        # Get legal action mask (in physical coords)
        def single_legal(state):
            return get_legal_actions(state, env_config)

        legal_mask = jax.vmap(single_legal)(next_states)

        # Mask illegal actions
        masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)

        # Check for terminal states
        terminated = next_states.terminated

        # Two-player game: discount=-1 when player changes (negates opponent's value),
        # +1 when same player (mid-jump), 0 for terminal.
        player_changed = (embedding.current_player != next_states.current_player)
        discount = jnp.where(
            terminated, 0.0,
            jnp.where(player_changed, -1.0, 1.0)
        )

        # Terminal value from the acting player's perspective.
        terminal_value = jnp.where(
            next_states.winner == next_states.current_player,
            1.0,   # Acting player won
            jnp.where(next_states.winner == 0, 0.0, -1.0)  # Draw or own-goal
        )

        # Terminal value goes into reward (not value), because with discount=0:
        # Q = reward + 0*value = reward. Putting it in value would lose it.
        reward = jnp.where(terminated, terminal_value, 0.0)

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=masked_logits,
            value=values,
        )

        return recurrent_output, next_states

    return recurrent_fn


def batched_mcts_policy(
    params: dict,
    states: PhutballState,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    num_simulations: int = 50,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25,
    max_num_considered_actions: int = 32,
    recurrent_fn=None,  # Pass pre-created recurrent_fn to avoid recompilation
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get MCTS-improved policy for a batch of states.
    
    Returns:
        actions: (batch,) selected actions
        policies: (batch, action_space) MCTS visit count policies
        values: (batch,) root value estimates
    """
    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1
    batch_size = states.board.shape[0]
    
    # Convert states to network input for root evaluation
    # Uses state_to_network_input with jump sequence (NOTE: rotates 180° for P2)
    network_inputs = jax.vmap(lambda s: state_to_network_input(s, env_config))(states)

    # Get root network predictions
    variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
    policy_logits, values = network.apply(variables, network_inputs, train=False)

    # Transform policy logits back to physical coords for P2
    is_p2 = (states.current_player == 2)[:, None]  # (batch, 1) for broadcasting
    policy_logits_p2 = transform_policy_for_p2(policy_logits, rows, cols)
    policy_logits = jnp.where(is_p2, policy_logits_p2, policy_logits)

    # Get legal action mask (in physical coords)
    def single_legal(state):
        return get_legal_actions(state, env_config)
    
    legal_mask = jax.vmap(single_legal)(states)
    masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)

    rng, noise_rng = jax.random.split(rng)
    
    priors = jax.nn.softmax(masked_logits, axis=-1)
    
    # Sample Dirichlet noise for each state in batch
    noise_rngs = jax.random.split(noise_rng, batch_size)
    noise = jax.vmap(
        lambda r: jax.random.dirichlet(r, jnp.full(action_space_size, dirichlet_alpha))
    )(noise_rngs)
    
    # Mix: (1 - ε) * prior + ε * noise, but only on legal actions
    noisy_priors = (1 - dirichlet_fraction) * priors + dirichlet_fraction * noise
    noisy_priors = jnp.where(legal_mask == 1, noisy_priors, 0.0)
    noisy_priors = noisy_priors / (noisy_priors.sum(axis=-1, keepdims=True) + 1e-8)

    noisy_logits = jnp.log(noisy_priors + 1e-8)
    noisy_logits = jnp.where(legal_mask == 1, noisy_logits, -1e9)

    root = mctx.RootFnOutput(
        prior_logits=noisy_logits,
        value=values,
        embedding=states,
    )

    # Use pre-created recurrent_fn if provided, otherwise create one
    if recurrent_fn is None:
        recurrent_fn = make_mcts_recurrent_fn(network, env_config)

    rng, mcts_rng = jax.random.split(rng)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=mcts_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=1.0,
    )

    mcts_policy = policy_output.action_weights
    
    root_values = policy_output.search_tree.node_values[:, 0]
    
    rng, sample_rng = jax.random.split(rng)
    
    # Always compute both paths, select with jnp.where (JAX-traceable)
    greedy_actions = jnp.argmax(mcts_policy, axis=-1)
    
    # Safe temperature division (avoid div by zero)
    safe_temp = jnp.maximum(temperature, 1e-8)
    logits = jnp.log(mcts_policy + 1e-8) / safe_temp
    sample_rngs = jax.random.split(sample_rng, batch_size)
    sampled_actions = jax.vmap(lambda r, l: jax.random.categorical(r, l))(sample_rngs, logits)
    
    # Select based on temperature
    actions = jnp.where(temperature < 0.01, greedy_actions, sampled_actions)
        
    return actions, mcts_policy, root_values


# ============================================================================
# Transformer variants (no batch_stats)
# ============================================================================

def make_transformer_recurrent_fn(network: PhutballTransformer, env_config: EnvConfig):
    """Create recurrent function for mctx that uses Transformer (no batch_stats)."""

    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    def recurrent_fn(params, rng, action, embedding):
        """
        Args:
            params: Network params dict (only 'network_params', no 'batch_stats')
            rng: Random key (unused - env is deterministic)
            action: Actions to take, shape (batch_size,)
            embedding: Current PhutballState (batched)

        Returns:
            RecurrentFnOutput, new_embedding
        """
        # Step the environment
        def single_step(state, act):
            return step(state, act, env_config)

        next_states = jax.vmap(single_step)(embedding, action)

        # Convert to network input (NOTE: rotates 180° for P2)
        network_inputs = jax.vmap(lambda s: state_to_network_input(s, env_config))(next_states)

        # Get network predictions (no batch_stats)
        variables = {'params': params['network_params']}
        policy_logits, values = network.apply(variables, network_inputs, train=False)

        # Transform policy logits back to physical coords for P2
        is_p2 = (next_states.current_player == 2)[:, None]  # (batch, 1) for broadcasting
        policy_logits_p2 = transform_policy_for_p2(policy_logits, rows, cols)
        policy_logits = jnp.where(is_p2, policy_logits_p2, policy_logits)

        # Get legal action mask (in physical coords)
        def single_legal(state):
            return get_legal_actions(state, env_config)

        legal_mask = jax.vmap(single_legal)(next_states)

        # Mask illegal actions
        masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)

        # Check for terminal states
        terminated = next_states.terminated

        # Two-player game: discount=-1 when player changes (negates opponent's value),
        # +1 when same player (mid-jump), 0 for terminal.
        player_changed = (embedding.current_player != next_states.current_player)
        discount = jnp.where(
            terminated, 0.0,
            jnp.where(player_changed, -1.0, 1.0)
        )

        # Terminal value from the acting player's perspective.
        terminal_value = jnp.where(
            next_states.winner == next_states.current_player,
            1.0,
            jnp.where(next_states.winner == 0, 0.0, -1.0)
        )

        # Terminal value goes into reward (not value), because with discount=0:
        # Q = reward + 0*value = reward. Putting it in value would lose it.
        reward = jnp.where(terminated, terminal_value, 0.0)

        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=masked_logits,
            value=values,
        )

        return recurrent_output, next_states

    return recurrent_fn


def transformer_mcts_policy(
    params: dict,
    states: PhutballState,
    rng: jnp.ndarray,
    network: PhutballTransformer,
    env_config: EnvConfig,
    num_simulations: int = 50,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.25,
    max_num_considered_actions: int = 32,
    recurrent_fn=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Get MCTS-improved policy for a batch of states using Transformer (no batch_stats).

    Returns:
        actions: (batch,) selected actions
        policies: (batch, action_space) MCTS visit count policies
        values: (batch,) root value estimates
    """
    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1
    batch_size = states.board.shape[0]

    # Convert states to network input for root evaluation (NOTE: rotates 180° for P2)
    network_inputs = jax.vmap(lambda s: state_to_network_input(s, env_config))(states)

    # Get root network predictions (no batch_stats)
    variables = {'params': params['network_params']}
    policy_logits, values = network.apply(variables, network_inputs, train=False)

    # Transform policy logits back to physical coords for P2
    is_p2 = (states.current_player == 2)[:, None]  # (batch, 1) for broadcasting
    policy_logits_p2 = transform_policy_for_p2(policy_logits, rows, cols)
    policy_logits = jnp.where(is_p2, policy_logits_p2, policy_logits)

    # Get legal action mask (in physical coords)
    def single_legal(state):
        return get_legal_actions(state, env_config)

    legal_mask = jax.vmap(single_legal)(states)
    masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)

    rng, noise_rng = jax.random.split(rng)

    priors = jax.nn.softmax(masked_logits, axis=-1)

    # Sample Dirichlet noise for each state in batch
    noise_rngs = jax.random.split(noise_rng, batch_size)
    noise = jax.vmap(
        lambda r: jax.random.dirichlet(r, jnp.full(action_space_size, dirichlet_alpha))
    )(noise_rngs)

    # Mix: (1 - ε) * prior + ε * noise, but only on legal actions
    noisy_priors = (1 - dirichlet_fraction) * priors + dirichlet_fraction * noise
    noisy_priors = jnp.where(legal_mask == 1, noisy_priors, 0.0)
    noisy_priors = noisy_priors / (noisy_priors.sum(axis=-1, keepdims=True) + 1e-8)

    noisy_logits = jnp.log(noisy_priors + 1e-8)
    noisy_logits = jnp.where(legal_mask == 1, noisy_logits, -1e9)

    root = mctx.RootFnOutput(
        prior_logits=noisy_logits,
        value=values,
        embedding=states,
    )

    # Use pre-created recurrent_fn if provided, otherwise create one
    if recurrent_fn is None:
        recurrent_fn = make_transformer_recurrent_fn(network, env_config)

    rng, mcts_rng = jax.random.split(rng)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=mcts_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        gumbel_scale=1.0,
    )

    mcts_policy = policy_output.action_weights

    root_values = policy_output.search_tree.node_values[:, 0]

    rng, sample_rng = jax.random.split(rng)

    # Always compute both paths, select with jnp.where (JAX-traceable)
    greedy_actions = jnp.argmax(mcts_policy, axis=-1)

    # Safe temperature division (avoid div by zero)
    safe_temp = jnp.maximum(temperature, 1e-8)
    logits = jnp.log(mcts_policy + 1e-8) / safe_temp
    sample_rngs = jax.random.split(sample_rng, batch_size)
    sampled_actions = jax.vmap(lambda r, l: jax.random.categorical(r, l))(sample_rngs, logits)

    # Select based on temperature
    actions = jnp.where(temperature < 0.01, greedy_actions, sampled_actions)

    return actions, mcts_policy, root_values


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
    
    # Convert states to network input - Uses state_to_network_input
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


def _make_frozen_state(old_states: PhutballState, new_states_raw: PhutballState, 
                       done_mask: jnp.ndarray, env_config: EnvConfig) -> PhutballState:
    """Helper to create a new state that freezes terminated games."""
    rows, cols = env_config.rows, env_config.cols
    batch_size = done_mask.shape[0]
    
    return PhutballState(
        board=jnp.where(done_mask[:, None, None], old_states.board, new_states_raw.board),
        ball_pos=jnp.where(done_mask[:, None], old_states.ball_pos, new_states_raw.ball_pos),
        current_player=jnp.where(done_mask, old_states.current_player, new_states_raw.current_player),
        is_jumping=jnp.where(done_mask, old_states.is_jumping, new_states_raw.is_jumping),
        terminated=done_mask,
        winner=jnp.where(done_mask & old_states.terminated, old_states.winner, new_states_raw.winner),
        num_turns=jnp.where(done_mask, old_states.num_turns, new_states_raw.num_turns),
        jump_sequence=jnp.where(
            done_mask[:, None, None], 
            old_states.jump_sequence, 
            new_states_raw.jump_sequence
        ),
        jump_sequence_length=jnp.where(
            done_mask, 
            old_states.jump_sequence_length, 
            new_states_raw.jump_sequence_length
        ),
    )


def play_games_batched(
    params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    batch_size: int = 64,
    max_turns: int = 7200,
    max_moves: int = 50000,  # Safety cap on total moves (for memory)
    temperature: float = 1.0,
    temp_threshold: int = 30,  # Moves before temperature drops
    temp_final: float = 0.1,   # Temperature after threshold
    num_simulations: int = 0,  # 0 = no MCTS, >0 = use MCTS with this many sims
    random_opponent_ratio: float = 0.0,  # Fraction of games vs random opponent
    opponent_params: dict = None,  # If provided, use for P2 (league play)
    opponent_ratio: float = 0.0,  # Fraction of games vs opponent_params
    mcts_policy_fn=None,  # Custom MCTS policy function (for transformer)
    recurrent_fn=None,  # Pre-created recurrent_fn (for transformer)
) -> TrajectoryData:
    """
    Play multiple games in parallel with optional random/league opponents.

    Args:
        params: Network parameters (used for P1, and P2 in self-play)
        rng: Random key
        network: Neural network
        env_config: Environment config
        batch_size: Number of games to play in parallel
        max_turns: Maximum turns per game (placement or halt = 1 turn)
        max_moves: Maximum moves to store (memory cap)
        temperature: Initial sampling temperature (for exploration)
        temp_threshold: Number of moves before temperature drops
        temp_final: Temperature after threshold (for exploitation)
        num_simulations: MCTS simulations per move (0 = raw network policy)
        random_opponent_ratio: Fraction of games where opponent plays randomly
        opponent_params: If provided, use these params for P2 in some games (league play)
        opponent_ratio: Fraction of games where P2 uses opponent_params

    Returns:
        TrajectoryData with all game trajectories
    """
    action_space_size = 2 * env_config.rows * env_config.cols + 1
    num_channels = 10  # see state_to_network_input; +1 vs old layout for single-jump reachability mask
    rows, cols = env_config.rows, env_config.cols
    use_mcts = num_simulations > 0

    # Determine which games have random/league opponents
    rng, mask_rng, side_rng, league_rng = jax.random.split(rng, 4)

    # Random opponents
    num_vs_random = int(batch_size * random_opponent_ratio)
    if random_opponent_ratio > 0 and num_vs_random == 0:
        num_vs_random = 1
    has_random_opponent = jnp.arange(batch_size) < num_vs_random

    # League opponents (only if opponent_params provided)
    num_vs_league = int(batch_size * opponent_ratio) if opponent_params is not None else 0
    if opponent_ratio > 0 and num_vs_league == 0 and opponent_params is not None:
        num_vs_league = 1
    # League games start after random games
    has_league_opponent = (jnp.arange(batch_size) >= num_vs_random) & \
                          (jnp.arange(batch_size) < num_vs_random + num_vs_league)

    # For vs-random/league games, randomly decide if main network plays P1 (50/50)
    random_sides = jax.random.uniform(side_rng, (batch_size,)) < 0.5
    has_any_opponent = has_random_opponent | has_league_opponent
    network_is_P1 = jnp.where(
        has_any_opponent,
        random_sides,  # Random assignment for vs-opponent games
        jnp.ones(batch_size, dtype=jnp.bool_),  # Self-play: network plays both
    )
    network_is_P2 = jnp.where(
        has_any_opponent,
        ~random_sides,
        jnp.ones(batch_size, dtype=jnp.bool_),
    )

    # Initialize storage for trajectories
    all_states = jnp.zeros((batch_size, max_moves, num_channels, rows, cols))
    all_policies = jnp.zeros((batch_size, max_moves, action_space_size))
    all_players = jnp.zeros((batch_size, max_moves), dtype=jnp.int32)
    all_actions = jnp.zeros((batch_size, max_moves), dtype=jnp.int32)
    valid_mask = jnp.zeros((batch_size, max_moves), dtype=jnp.bool_)

    # Initialize game states
    env_states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
    move_count = jnp.zeros(batch_size, dtype=jnp.int32)

    # Pre-create batched functions outside the loop
    def single_step(state, action):
        return step(state, action, env_config)

    def single_legal(state):
        return get_legal_actions(state, env_config)

    # Pre-create recurrent_fn once to avoid recompilation inside the loop
    if recurrent_fn is not None:
        mcts_recurrent_fn = recurrent_fn
    elif use_mcts:
        mcts_recurrent_fn = make_mcts_recurrent_fn(network, env_config)
    else:
        mcts_recurrent_fn = None

    # Use provided mcts_policy_fn or default to batched_mcts_policy
    _mcts_policy_fn = mcts_policy_fn if mcts_policy_fn is not None else batched_mcts_policy

    # Play loop - now takes step_idx for temperature scheduling
    def game_step(carry, _):
        (env_states, terminated, move_count, all_states, all_policies,
         all_players, all_actions, valid_mask, rng, step_idx) = carry

        rng, step_rng, sample_rng, rand_rng = jax.random.split(rng, 4)

        # Compute effective temperature based on step count
        effective_temp = jnp.where(
            step_idx < temp_threshold,
            temperature,
            temp_final
        )

        # Check if games have exceeded max_turns
        turns_exceeded = env_states.num_turns >= max_turns
        effectively_done = terminated | turns_exceeded
        active = ~effectively_done

        # Get current observations: (batch, 6, rows, cols)
        current_obs = jax.vmap(lambda s: state_to_network_input(s, env_config))(env_states)
        current_players = env_states.current_player

        # Determine who makes this move:
        # - use_main_network: current params (self-play or main side of opponent game)
        # - use_league_opponent: opponent_params (league game, opponent's turn)
        # - use_random: random (random game, opponent's turn)
        is_network_turn = jnp.where(current_players == 1, network_is_P1, network_is_P2)
        use_main_network = jnp.where(
            has_any_opponent,
            is_network_turn,
            jnp.ones(batch_size, dtype=jnp.bool_),  # Self-play: always main
        )
        use_league_opponent = has_league_opponent & ~is_network_turn
        use_random = has_random_opponent & ~is_network_turn

        # Get legal action mask
        legal_mask = jax.vmap(single_legal)(env_states)

        if use_mcts:
            # Use MCTS to get improved policy (main network)
            actions_net, policies_net, values = _mcts_policy_fn(
                params, env_states, step_rng, network, env_config,
                num_simulations=num_simulations, temperature=effective_temp,
                recurrent_fn=mcts_recurrent_fn,
            )
            # League opponent MCTS (if needed)
            if opponent_params is not None:
                actions_league, policies_league, _ = _mcts_policy_fn(
                    opponent_params, env_states, step_rng, network, env_config,
                    num_simulations=num_simulations, temperature=effective_temp,
                    recurrent_fn=mcts_recurrent_fn,
                )
            else:
                actions_league, policies_league = actions_net, policies_net
        else:
            # Use raw network policy (main)
            variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
            policy_logits, values = network.apply(variables, current_obs, train=False)
            masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)
            policies_net = jax.nn.softmax(masked_logits / effective_temp)
            sample_rngs = jax.random.split(sample_rng, batch_size)
            actions_net = jax.vmap(lambda r, p: jax.random.choice(r, action_space_size, p=p))(sample_rngs, policies_net)

            # League opponent policy (if provided)
            if opponent_params is not None:
                opp_variables = {'params': opponent_params['network_params'], 'batch_stats': opponent_params['batch_stats']}
                opp_logits, _ = network.apply(opp_variables, current_obs, train=False)
                opp_masked = jnp.where(legal_mask == 1, opp_logits, -1e9)
                policies_league = jax.nn.softmax(opp_masked / effective_temp)
                opp_rngs = jax.random.split(sample_rng, batch_size)
                actions_league = jax.vmap(lambda r, p: jax.random.choice(r, action_space_size, p=p))(opp_rngs, policies_league)
            else:
                actions_league, policies_league = actions_net, policies_net

        # Random actions (uniform over legal)
        def get_random_action(state, rng_key):
            legal = get_legal_actions(state, env_config)
            probs = legal.astype(jnp.float32)
            probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
            return jax.random.choice(rng_key, action_space_size, p=probs)

        rand_rngs = jax.random.split(rand_rng, batch_size)
        actions_rand = jax.vmap(get_random_action)(env_states, rand_rngs)

        # Random policy (for storage - uniform over legal moves)
        policies_rand = legal_mask.astype(jnp.float32)
        policies_rand = policies_rand / jnp.maximum(jnp.sum(policies_rand, axis=-1, keepdims=True), 1e-8)

        # Select actions and policies: main > league > random
        actions = jnp.where(use_main_network, actions_net,
                  jnp.where(use_league_opponent, actions_league, actions_rand))
        policies = jnp.where(use_main_network[:, None], policies_net,
                   jnp.where(use_league_opponent[:, None], policies_league, policies_rand))
        
        # Clamp move_count to valid range for indexing
        safe_move_idx = jnp.minimum(move_count, max_moves - 1)
        
        # Update trajectories using scatter
        batch_idx = jnp.arange(batch_size)
        
        # Only update where active
        all_states = all_states.at[batch_idx, safe_move_idx].set(
            jnp.where(active[:, None, None, None], current_obs, all_states[batch_idx, safe_move_idx])
        )
        all_policies = all_policies.at[batch_idx, safe_move_idx].set(
            jnp.where(active[:, None], policies, all_policies[batch_idx, safe_move_idx])
        )
        all_players = all_players.at[batch_idx, safe_move_idx].set(
            jnp.where(active, current_players, all_players[batch_idx, safe_move_idx])
        )
        valid_mask = valid_mask.at[batch_idx, safe_move_idx].set(active)

        stored_actions = jnp.where(active, actions, all_actions[batch_idx, safe_move_idx])
        all_actions = all_actions.at[batch_idx, safe_move_idx].set(stored_actions)
        
        # Step environments
        new_env_states_raw = jax.vmap(single_step)(env_states, actions)

        # Capture termination status BEFORE freezing (includes games that just terminated)
        just_terminated = new_env_states_raw.terminated

        # For terminated games, keep old state
        new_env_states = _make_frozen_state(env_states, new_env_states_raw, effectively_done, env_config)

        # Update termination status using raw terminated, not the frozen one
        new_terminated = terminated | just_terminated
        new_move_count = move_count + active.astype(jnp.int32)
        
        carry = (new_env_states, new_terminated, new_move_count, 
                all_states, all_policies, all_players, all_actions, valid_mask, 
                rng, step_idx + jnp.int32(1))
        return carry, None
    
    # Run the game loop with early stopping.
    step0 = jnp.int32(0)

    def cond_fn(carry):
        (env_states, terminated, move_count, all_states, all_policies, 
         all_players, all_actions, valid_mask, rng, step_idx) = carry

        # Same notion of "done" as inside game_step
        turns_exceeded = env_states.num_turns >= max_turns
        effectively_done = terminated | turns_exceeded
        any_active = jnp.any(~effectively_done)

        # Continue while we still have active games and haven't hit max_moves
        return (step_idx < max_moves) & any_active

    def body_fn(carry):
        # Re-use the existing game_step logic
        carry, _ = game_step(carry, None)
        return carry

    initial_carry = (env_states, terminated, move_count,
                     all_states, all_policies, all_players, all_actions, valid_mask, 
                     rng, step0)

    final_carry = lax.while_loop(cond_fn, body_fn, initial_carry)

    (final_env_states, _, _,
     all_states, all_policies, all_players, all_actions, valid_mask, _, _) = final_carry

    # Get winners from the final env states
    winners = final_env_states.winner

    return TrajectoryData(
        states=all_states,
        policies=all_policies,
        players=all_players,
        valid_mask=valid_mask,
        winners=winners,
        actions=all_actions,
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def play_match_batched(
    home_params: dict,
    away_params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    games_per_color: int,
    max_moves: int,
    num_simulations: int,
    temperature: float = 0.0,
):
    """
    Batched match between two checkpoints: `home` vs `away`.

    We launch 2 * games_per_color games:

      - games [0 .. g-1]:   home plays as P1, away as P2
      - games [g .. 2g-1]:  away plays as P1, home as P2

    All games are stepped in parallel using a JAX while_loop.

    Returns (all JAX arrays):

        total_score_home: sum of scores (+1 win, -1 loss, 0 draw) from *home* POV
        total_games:      number of games played (= 2 * games_per_color)
        home_wins, draws, away_wins: counts
        per_game_turns:         num_turns from env
        per_game_jumps_p1/p2:   jump counts by env P1/P2
        per_game_jumps_total:   total jumps (P1+P2)
        per_game_removed:       total men removed by jumps
        winners:                {0=draw, 1=P1 win, 2=P2 win}
        home_is_P1:             bool per game, True if home was P1 in that game
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    batch_size = 2 * games_per_color

    # Which agent is P1/P2 in each game? (True = home, False = away)
    home_is_P1 = jnp.concatenate(
        [
            jnp.ones((games_per_color,), dtype=jnp.bool_),   # first block: home is P1
            jnp.zeros((games_per_color,), dtype=jnp.bool_),  # second block: away is P1
        ],
        axis=0,
    )
    home_is_P2 = ~home_is_P1  # complement

    # Initial batched env state
    states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # Stats we accumulate per game
    jumps_p1 = jnp.zeros((batch_size,), dtype=jnp.int32)
    jumps_p2 = jnp.zeros((batch_size,), dtype=jnp.int32)
    jumps_total = jnp.zeros((batch_size,), dtype=jnp.int32)
    jump_removed_total = jnp.zeros((batch_size,), dtype=jnp.int32)

    step0 = jnp.int32(0)

    # Pre-create recurrent_fn once to avoid recompilation inside the loop
    mcts_recurrent_fn = make_mcts_recurrent_fn(network, env_config)

    def cond_fn(carry):
        (states, terminated, rng, step_idx,
         jumps_p1, jumps_p2, jumps_total, jump_removed_total) = carry
        any_active = jnp.any(~terminated)
        return jnp.logical_and(step_idx < max_moves, any_active)

    def body_fn(carry):
        (states, terminated, rng, step_idx,
         jumps_p1, jumps_p2, jumps_total, jump_removed_total) = carry

        current_player = states.current_player  # (batch,)

        # For each game, decide whether *home* is to move
        use_home = jnp.where(
            current_player == 1,
            home_is_P1,   # if P1 to move, this tells us if home is P1
            home_is_P2,   # if P2 to move, this tells us if home is P2
        )

        rng, rng_home, rng_away = jax.random.split(rng, 3)

        # Run MCTS for ALL games with home, and ALL games with away,
        # then select per-game outputs based on use_home.
        actions_home, _, _ = batched_mcts_policy(
            home_params,
            states,
            rng_home,
            network,
            env_config,
            num_simulations=num_simulations,
            temperature=temperature,
            recurrent_fn=mcts_recurrent_fn,
        )
        actions_away, _, _ = batched_mcts_policy(
            away_params,
            states,
            rng_away,
            network,
            env_config,
            num_simulations=num_simulations,
            temperature=temperature,
            recurrent_fn=mcts_recurrent_fn,
        )

        actions = jnp.where(use_home, actions_home, actions_away)  # (batch,)

        # Step all envs in parallel
        board_before = states.board                    # (batch, R, C)
        new_states_raw = jax.vmap(lambda s, a: step(s, a, env_config))(states, actions)
        board_after = new_states_raw.board             # (batch, R, C)

        active = ~terminated

        # Action encoding: [0..N-1] placements, [N..2N-1] jumps, others = halt, etc.
        is_jump = (actions >= total_positions) & (actions < 2 * total_positions)
        jump_mask = is_jump & active

        # Count jumps by env player
        p1_mask = (current_player == 1) & jump_mask
        p2_mask = (current_player == 2) & jump_mask

        jumps_p1 = jumps_p1 + p1_mask.astype(jnp.int32)
        jumps_p2 = jumps_p2 + p2_mask.astype(jnp.int32)
        jumps_total = jumps_total + jump_mask.astype(jnp.int32)

        # Men removed by a jump (only when a jump actually occurred)
        men_before = jnp.sum((board_before == MAN).astype(jnp.int32), axis=(1, 2))
        men_after = jnp.sum((board_after == MAN).astype(jnp.int32), axis=(1, 2))
        removed_step = jnp.maximum(men_before - men_after, 0)
        removed_step = removed_step * jump_mask.astype(jnp.int32)
        jump_removed_total = jump_removed_total + removed_step

        # Once a game is terminated, freeze it
        done_mask = terminated | new_states_raw.terminated

        new_states = _make_frozen_state(states, new_states_raw, done_mask, env_config)

        return (
            new_states,
            done_mask,
            rng,
            step_idx + jnp.int32(1),
            jumps_p1,
            jumps_p2,
            jumps_total,
            jump_removed_total,
        )

    init_carry = (states, terminated, rng, step0,
                  jumps_p1, jumps_p2, jumps_total, jump_removed_total)

    (final_states,
     final_terminated,
     _,
     _,
     jumps_p1,
     jumps_p2,
     jumps_total,
     jump_removed_total) = lax.while_loop(cond_fn, body_fn, init_carry)

    winners = final_states.winner      # (batch,) in {0,1,2}
    per_game_turns = final_states.num_turns

    # From env: winner==1 ⇒ P1, winner==2 ⇒ P2, 0 ⇒ draw/cutoff
    win_p1 = (winners == 1).astype(jnp.float32)
    win_p2 = (winners == 2).astype(jnp.float32)

    # Score from HOME's POV
    # If home is P1, +1 when P1 wins, -1 when P2 wins
    score_if_home_P1 = win_p1 - win_p2
    # If home is P2, +1 when P2 wins, -1 when P1 wins
    score_if_home_P2 = win_p2 - win_p1

    per_game_score_home = jnp.where(home_is_P1, score_if_home_P1, score_if_home_P2)
    total_score_home = jnp.sum(per_game_score_home)
    total_games = jnp.int32(batch_size)

    # W/D/L counts from home POV
    home_win = ((winners == 1) & home_is_P1) | ((winners == 2) & ~home_is_P1)
    away_win = ((winners == 1) & ~home_is_P1) | ((winners == 2) & home_is_P1)
    draw_mask = (winners == 0)

    home_wins = jnp.sum(home_win.astype(jnp.int32))
    away_wins = jnp.sum(away_win.astype(jnp.int32))
    draws = jnp.sum(draw_mask.astype(jnp.int32))

    return (
        total_score_home,
        total_games,
        home_wins,
        draws,
        away_wins,
        per_game_turns,
        jumps_p1,
        jumps_p2,
        jumps_total,
        jump_removed_total,
        winners,
        home_is_P1,
    )


# Module-level JIT functions for play_vs_random_batched
# Moved outside to avoid recompilation on each call (causes XLA stack overflow on TPU v6e)

@partial(jax.jit, static_argnums=(2, 3, 4))
def _get_random_actions_batched(states, rng_rand, action_space_size, batch_size, env_config):
    """Get random legal actions for all games."""
    def get_random_action(state, rng_key):
        legal = get_legal_actions(state, env_config)
        probs = legal.astype(jnp.float32)
        probs = probs / jnp.sum(probs)
        return jax.random.choice(rng_key, action_space_size, p=probs)
    rand_rngs = jax.random.split(rng_rand, batch_size)
    return jax.vmap(get_random_action)(states, rand_rngs)


@partial(jax.jit, static_argnums=(3,))
def _step_games_batched(states, actions, terminated, env_config):
    """Step all games and freeze terminated ones."""
    new_states_raw = jax.vmap(lambda s, a: step(s, a, env_config))(states, actions)
    done_mask = terminated | new_states_raw.terminated
    new_states = _make_frozen_state(states, new_states_raw, done_mask, env_config)
    return new_states, done_mask


def play_vs_random_batched(
    checkpoint_params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    num_games: int,
    max_moves: int,
    num_simulations: int,
    temperature: float = 0.0,
    dirichlet_fraction: float = 0.0,  # 0.0 for eval (no exploration noise)
    mcts_policy_fn=None,  # Custom MCTS policy function (for transformer)
    recurrent_fn=None,  # Pre-created recurrent_fn (for transformer)
):
    """
    Play checkpoint vs random policy.
    
    Runs num_games total:
      - games [0 .. num_games//2 - 1]: checkpoint as P1, random as P2
      - games [num_games//2 .. num_games - 1]: random as P1, checkpoint as P2
    
    Returns:
        checkpoint_wins: number of wins for checkpoint
        draws: number of draws  
        random_wins: number of wins for random
        per_game_turns: turns per game
        winners: raw winner array {0=draw, 1=P1, 2=P2}
        checkpoint_is_P1: bool array indicating which color checkpoint played
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols
    action_space_size = 2 * total_positions + 1
    
    games_per_color = num_games // 2
    batch_size = 2 * games_per_color
    
    # Which games have checkpoint as P1?
    checkpoint_is_P1 = jnp.concatenate([
        jnp.ones((games_per_color,), dtype=jnp.bool_),
        jnp.zeros((games_per_color,), dtype=jnp.bool_),
    ])
    checkpoint_is_P2 = ~checkpoint_is_P1
    
    # Initial states
    states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)
    step0 = jnp.int32(0)

    # Pre-create recurrent_fn once to avoid recompilation inside the loop
    if recurrent_fn is not None:
        mcts_recurrent_fn = recurrent_fn
    else:
        mcts_recurrent_fn = make_mcts_recurrent_fn(network, env_config)

    # Use provided mcts_policy_fn or default to batched_mcts_policy
    _mcts_policy_fn = mcts_policy_fn if mcts_policy_fn is not None else batched_mcts_policy

    # Python loop - avoids stack overflow from deep JIT compilation
    # Note: get_random_actions and step_games moved to module level to prevent
    # recompilation on each call (which caused XLA stack overflow on TPU v6e)
    for step_idx in range(max_moves):
        # Check if any games still active
        if not bool(jnp.any(~terminated)):
            break

        current_player = states.current_player
        use_checkpoint = jnp.where(
            current_player == 1,
            checkpoint_is_P1,
            checkpoint_is_P2,
        )

        rng, rng_ckpt, rng_rand = jax.random.split(rng, 3)

        # Get checkpoint actions via MCTS
        actions_ckpt, _, _ = _mcts_policy_fn(
            checkpoint_params,
            states,
            rng_ckpt,
            network,
            env_config,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_fraction=dirichlet_fraction,
            recurrent_fn=mcts_recurrent_fn,
        )

        # Get random actions
        actions_rand = _get_random_actions_batched(states, rng_rand, action_space_size, batch_size, env_config)

        # Select action based on who's moving
        actions = jnp.where(use_checkpoint, actions_ckpt, actions_rand)

        # Step environments
        states, terminated = _step_games_batched(states, actions, terminated, env_config)

    final_states = states
    
    winners = final_states.winner
    per_game_turns = final_states.num_turns

    # Compute side-aware stats (from checkpoint's POV)
    draw_mask = (winners == 0)

    # When checkpoint is P1: win if winner==1, loss if winner==2
    p1_win_mask = checkpoint_is_P1 & (winners == 1)
    p1_draw_mask = checkpoint_is_P1 & draw_mask
    p1_loss_mask = checkpoint_is_P1 & (winners == 2)

    # When checkpoint is P2: win if winner==2, loss if winner==1
    p2_win_mask = checkpoint_is_P2 & (winners == 2)
    p2_draw_mask = checkpoint_is_P2 & draw_mask
    p2_loss_mask = checkpoint_is_P2 & (winners == 1)

    p1_wins = jnp.sum(p1_win_mask.astype(jnp.int32))
    p1_draws = jnp.sum(p1_draw_mask.astype(jnp.int32))
    p1_losses = jnp.sum(p1_loss_mask.astype(jnp.int32))

    p2_wins = jnp.sum(p2_win_mask.astype(jnp.int32))
    p2_draws = jnp.sum(p2_draw_mask.astype(jnp.int32))
    p2_losses = jnp.sum(p2_loss_mask.astype(jnp.int32))

    return (
        p1_wins, p1_draws, p1_losses,
        p2_wins, p2_draws, p2_losses,
        per_game_turns,
    )


def play_vs_checkpoint_batched(
    current_params: dict,
    opponent_params: dict,
    rng: jnp.ndarray,
    network,
    env_config: EnvConfig,
    num_games: int,
    max_moves: int,
    num_simulations: int,
    temperature: float = 0.0,
    dirichlet_fraction: float = 0.0,
    mcts_policy_fn=None,
    recurrent_fn=None,
):
    """
    Play current checkpoint vs opponent checkpoint with MCTS on both sides.

    Runs num_games total:
      - games [0 .. num_games//2 - 1]: current as P1, opponent as P2
      - games [num_games//2 .. num_games - 1]: opponent as P1, current as P2

    Returns:
        p1_wins, p1_draws, p1_losses: current's record when playing as P1
        p2_wins, p2_draws, p2_losses: current's record when playing as P2
        per_game_turns: turns per game
    """
    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    games_per_color = num_games // 2
    batch_size = 2 * games_per_color

    # Which games have current as P1?
    current_is_P1 = jnp.concatenate([
        jnp.ones((games_per_color,), dtype=jnp.bool_),
        jnp.zeros((games_per_color,), dtype=jnp.bool_),
    ])
    current_is_P2 = ~current_is_P1

    # Initial states
    states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # Pre-create recurrent_fn once
    if recurrent_fn is not None:
        mcts_recurrent_fn = recurrent_fn
    else:
        mcts_recurrent_fn = make_mcts_recurrent_fn(network, env_config)

    _mcts_policy_fn = mcts_policy_fn if mcts_policy_fn is not None else batched_mcts_policy

    for step_idx in range(max_moves):
        if not bool(jnp.any(~terminated)):
            break

        current_player = states.current_player
        use_current = jnp.where(
            current_player == 1,
            current_is_P1,
            current_is_P2,
        )

        rng, rng_cur, rng_opp = jax.random.split(rng, 3)

        # MCTS for current params
        actions_cur, _, _ = _mcts_policy_fn(
            current_params,
            states,
            rng_cur,
            network,
            env_config,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_fraction=dirichlet_fraction,
            recurrent_fn=mcts_recurrent_fn,
        )

        # MCTS for opponent params
        actions_opp, _, _ = _mcts_policy_fn(
            opponent_params,
            states,
            rng_opp,
            network,
            env_config,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_fraction=dirichlet_fraction,
            recurrent_fn=mcts_recurrent_fn,
        )

        # Select action per game
        actions = jnp.where(use_current, actions_cur, actions_opp)

        # Step environments
        states, terminated = _step_games_batched(states, actions, terminated, env_config)

    winners = states.winner
    per_game_turns = states.num_turns

    # Compute side-aware stats (from current's POV)
    draw_mask = (winners == 0)

    p1_win_mask = current_is_P1 & (winners == 1)
    p1_draw_mask = current_is_P1 & draw_mask
    p1_loss_mask = current_is_P1 & (winners == 2)

    p2_win_mask = current_is_P2 & (winners == 2)
    p2_draw_mask = current_is_P2 & draw_mask
    p2_loss_mask = current_is_P2 & (winners == 1)

    p1_wins = jnp.sum(p1_win_mask.astype(jnp.int32))
    p1_draws = jnp.sum(p1_draw_mask.astype(jnp.int32))
    p1_losses = jnp.sum(p1_loss_mask.astype(jnp.int32))

    p2_wins = jnp.sum(p2_win_mask.astype(jnp.int32))
    p2_draws = jnp.sum(p2_draw_mask.astype(jnp.int32))
    p2_losses = jnp.sum(p2_loss_mask.astype(jnp.int32))

    return (
        p1_wins, p1_draws, p1_losses,
        p2_wins, p2_draws, p2_losses,
        per_game_turns,
    )


def play_games_vs_random_training(
    params: dict,
    rng: jnp.ndarray,
    network: PhutballNetwork,
    env_config: EnvConfig,
    batch_size: int = 64,
    max_turns: int = 7200,
    max_moves: int = 50000,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    temp_final: float = 0.1,
    num_simulations: int = 0,
) -> TrajectoryData:
    """
    Play games against a random opponent and collect training data.

    Half the games have network as P1 (random as P2), half have random as P1.
    Collects trajectory data from ALL moves (both network and random) - learning
    from random's moves with correct outcome signal can teach what NOT to do.

    Args:
        params: Network parameters
        rng: Random key
        network: Neural network
        env_config: Environment config
        batch_size: Number of games to play in parallel
        max_turns: Maximum turns per game
        max_moves: Maximum moves to store (memory cap)
        temperature: Initial sampling temperature
        temp_threshold: Moves before temperature drops
        temp_final: Temperature after threshold
        num_simulations: MCTS simulations per move (0 = raw network)

    Returns:
        TrajectoryData with all game trajectories
    """
    action_space_size = 2 * env_config.rows * env_config.cols + 1
    num_channels = 10  # see state_to_network_input; +1 vs old layout for single-jump reachability mask
    rows, cols = env_config.rows, env_config.cols
    use_mcts = num_simulations > 0

    # Which games have network as P1?
    games_per_side = batch_size // 2
    network_is_P1 = jnp.concatenate([
        jnp.ones((games_per_side,), dtype=jnp.bool_),
        jnp.zeros((batch_size - games_per_side,), dtype=jnp.bool_),
    ])
    network_is_P2 = ~network_is_P1

    # Initialize storage
    all_states = jnp.zeros((batch_size, max_moves, num_channels, rows, cols))
    all_policies = jnp.zeros((batch_size, max_moves, action_space_size))
    all_players = jnp.zeros((batch_size, max_moves), dtype=jnp.int32)
    all_actions = jnp.zeros((batch_size, max_moves), dtype=jnp.int32)
    valid_mask = jnp.zeros((batch_size, max_moves), dtype=jnp.bool_)

    # Initialize game states
    env_states = batched_reset(env_config, batch_size)
    terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
    move_count = jnp.zeros(batch_size, dtype=jnp.int32)

    def single_step(state, action):
        return step(state, action, env_config)

    def single_legal(state):
        return get_legal_actions(state, env_config)

    # Pre-create recurrent_fn once to avoid recompilation inside the loop
    mcts_recurrent_fn = make_mcts_recurrent_fn(network, env_config) if use_mcts else None

    def game_step(carry):
        (env_states, terminated, move_count, all_states, all_policies,
         all_players, all_actions, valid_mask, rng, step_idx) = carry

        rng, step_rng, sample_rng, rand_rng = jax.random.split(rng, 4)

        # Temperature schedule
        effective_temp = jnp.where(step_idx < temp_threshold, temperature, temp_final)

        # Check if games are done
        turns_exceeded = env_states.num_turns >= max_turns
        effectively_done = terminated | turns_exceeded
        active = ~effectively_done

        # Current observations
        current_obs = jax.vmap(lambda s: state_to_network_input(s, env_config))(env_states)
        current_players = env_states.current_player

        # Determine if network should play this move
        use_network = jnp.where(
            current_players == 1,
            network_is_P1,
            network_is_P2,
        )

        # Get legal actions
        legal_mask = jax.vmap(single_legal)(env_states)

        if use_mcts:
            # Network actions via MCTS
            actions_net, policies_net, _ = batched_mcts_policy(
                params, env_states, step_rng, network, env_config,
                num_simulations=num_simulations, temperature=effective_temp,
                recurrent_fn=mcts_recurrent_fn,
            )
        else:
            # Network actions via raw policy
            variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
            policy_logits, _ = network.apply(variables, current_obs, train=False)
            masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)
            policies_net = jax.nn.softmax(masked_logits / effective_temp)
            sample_rngs = jax.random.split(sample_rng, batch_size)
            actions_net = jax.vmap(lambda r, p: jax.random.choice(r, action_space_size, p=p))(sample_rngs, policies_net)

        # Random actions (uniform over legal)
        def get_random_action(state, rng_key):
            legal = get_legal_actions(state, env_config)
            probs = legal.astype(jnp.float32)
            probs = probs / jnp.sum(probs)
            return jax.random.choice(rng_key, action_space_size, p=probs)

        rand_rngs = jax.random.split(rand_rng, batch_size)
        actions_rand = jax.vmap(get_random_action)(env_states, rand_rngs)

        # Random policy (for storage)
        policies_rand = legal_mask.astype(jnp.float32)
        policies_rand = policies_rand / jnp.sum(policies_rand, axis=-1, keepdims=True)

        # Select actions and policies based on who's moving
        actions = jnp.where(use_network, actions_net, actions_rand)
        policies = jnp.where(use_network[:, None], policies_net, policies_rand)

        # Update trajectories
        safe_move_idx = jnp.minimum(move_count, max_moves - 1)
        batch_idx = jnp.arange(batch_size)

        all_states = all_states.at[batch_idx, safe_move_idx].set(
            jnp.where(active[:, None, None, None], current_obs, all_states[batch_idx, safe_move_idx])
        )
        all_policies = all_policies.at[batch_idx, safe_move_idx].set(
            jnp.where(active[:, None], policies, all_policies[batch_idx, safe_move_idx])
        )
        all_players = all_players.at[batch_idx, safe_move_idx].set(
            jnp.where(active, current_players, all_players[batch_idx, safe_move_idx])
        )
        valid_mask = valid_mask.at[batch_idx, safe_move_idx].set(active)
        stored_actions = jnp.where(active, actions, all_actions[batch_idx, safe_move_idx])
        all_actions = all_actions.at[batch_idx, safe_move_idx].set(stored_actions)

        # Step environments
        new_env_states_raw = jax.vmap(single_step)(env_states, actions)
        just_terminated = new_env_states_raw.terminated
        new_env_states = _make_frozen_state(env_states, new_env_states_raw, effectively_done, env_config)
        new_terminated = terminated | just_terminated
        new_move_count = move_count + active.astype(jnp.int32)

        carry = (new_env_states, new_terminated, new_move_count,
                all_states, all_policies, all_players, all_actions, valid_mask,
                rng, step_idx + jnp.int32(1))
        return carry

    step0 = jnp.int32(0)

    def cond_fn(carry):
        (env_states, terminated, move_count, *_) = carry
        any_active = jnp.any(~terminated)
        moves_ok = jnp.all(move_count < max_moves)
        return jnp.logical_and(any_active, moves_ok)

    init_carry = (env_states, terminated, move_count, all_states, all_policies,
                  all_players, all_actions, valid_mask, rng, step0)

    final_carry = lax.while_loop(cond_fn, game_step, init_carry)
    (final_states, _, _, all_states, all_policies, all_players, all_actions, valid_mask, _, _) = final_carry

    return TrajectoryData(
        states=all_states,
        policies=all_policies,
        players=all_players,
        valid_mask=valid_mask,
        winners=final_states.winner,
        actions=all_actions,
    )


def trajectory_to_training_examples(
    trajectory: TrajectoryData,
    draw_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert trajectory data to training examples.

    NOTE: MCTS policies are stored in physical coords. For P2, we need to
    transform them back to visual coords (180° rotation) to match the network
    input which is also rotated for P2.

    Args:
        trajectory: Game trajectory data from self-play.
        draw_value: Value assigned to drawn games from each player's perspective.
            Default 0.0 (standard). Use negative values (e.g. -0.3) to penalize
            draws and discourage passive play / draw equilibria.

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

    # Infer board dimensions from state shape: (batch, moves, channels, rows, cols)
    _, _, _, rows, cols = states_np.shape
    N = rows * cols

    def transform_policy_to_visual(policy):
        """Transform policy from physical coords back to visual coords for P2."""
        # Same transformation as physical->visual (180° rotation is self-inverse)
        placement = policy[:N][::-1]
        jump = policy[N:2*N][::-1]
        halt = policy[2*N:]
        return np.concatenate([placement, jump, halt])

    for game_idx in range(batch_size):
        winner = winners_np[game_idx]

        for move_idx in range(max_moves):
            if not valid_np[game_idx, move_idx]:
                continue

            state = states_np[game_idx, move_idx]
            policy = policies_np[game_idx, move_idx]
            player = players_np[game_idx, move_idx]

            # Transform P2 policies back to visual coords for training
            # (network input is rotated 180° for P2, so target policy should match)
            if player == 2:
                policy = transform_policy_to_visual(policy)

            # Value from this player's perspective
            if winner == 0:
                value = draw_value  # Draw (penalized when draw_value < 0)
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


def compute_phutball_stats(
    trajectory: TrajectoryData,
    env_config: EnvConfig,
) -> Dict[str, float]:
    """
    Compute high-level Phutball stats from a batch of self-play games.

    Returns *totals* which you can aggregate across batches:
      - num_games
      - total_moves
      - total_placements
      - total_jumps
      - num_jump_sequences
      - sum_jump_sequence_lengths
      - sum_jump_removed_tiles
      - adjacency_opportunities
      - adjacency_conversions
    """
    states_np = np.asarray(trajectory.states)      # (B, M, C, R, C)
    actions_np = np.asarray(trajectory.actions)    # (B, M)
    valid_np = np.asarray(trajectory.valid_mask)   # (B, M)
    players_np = np.asarray(trajectory.players)    # (B, M) which player moved

    batch_size, max_moves = valid_np.shape
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    total_moves = 0
    total_placements = 0
    total_jumps = 0
    num_jump_sequences = 0
    sum_jump_sequence_lengths = 0
    sum_jump_removed_tiles = 0
    adjacency_opportunities = 0
    adjacency_conversions = 0

    for g in range(batch_size):
        prev_action_was_jump = False
        current_seq_len = 0

        for t in range(max_moves):
            if not valid_np[g, t]:
                continue

            total_moves += 1
            a = int(actions_np[g, t])
            # Channel 0 = ball (binary), Channel 1 = men (binary)
            ball_ch = states_np[g, t, 0]
            men_ch = states_np[g, t, 1]

            # For P2's moves, the observation was 180° rotated for perspective normalization
            # Unflip to get physical coordinates that match the action encoding
            if players_np[g, t] == 2:
                ball_ch = np.flip(np.flip(ball_ch, axis=0), axis=1)
                men_ch = np.flip(np.flip(men_ch, axis=0), axis=1)

            is_placement = a < total_positions
            is_jump = (total_positions <= a < 2 * total_positions)

            if is_placement:
                # --- % occupying-adjacent-tile conversion ---
                ball_pos = np.argwhere(ball_ch > 0.5)
                if ball_pos.size > 0:
                    br, bc = ball_pos[0]

                    r0 = max(br - 1, 0)
                    r1 = min(br + 1, rows - 1)
                    c0 = max(bc - 1, 0)
                    c1 = min(bc + 1, cols - 1)
                    neighborhood = men_ch[r0:r1 + 1, c0:c1 + 1]
                    has_adjacent_man = np.any(neighborhood > 0.5)

                    pr = a // cols
                    pc = a % cols
                    is_adjacent = (
                        abs(pr - br) <= 1
                        and abs(pc - bc) <= 1
                        and not (pr == br and pc == bc)
                    )

                    # "Opportunity" = ball currently has no adjacent men
                    if not has_adjacent_man:
                        adjacency_opportunities += 1
                        if is_adjacent:
                            adjacency_conversions += 1

                total_placements += 1

            if is_jump:
                total_jumps += 1

                # --- jump length (tiles removed) using env helper ---
                jump_idx = a - total_positions
                land_r = jump_idx // cols
                land_c = jump_idx % cols
                landing_pos = np.array([land_r, land_c], dtype=np.int32)

                ball_pos_arr = np.argwhere(ball_ch > 0.5)
                if ball_pos_arr.size > 0:
                    ball_pos = ball_pos_arr[0]

                    # Reconstruct board for calculate_jumped_men
                    # (it needs integer tile values to identify men)
                    board_reconstructed = np.zeros_like(ball_ch, dtype=np.int32)
                    board_reconstructed[ball_ch > 0.5] = BALL
                    board_reconstructed[men_ch > 0.5] = MAN

                    jumped = calculate_jumped_men(
                        jnp.array(ball_pos, dtype=jnp.int32),
                        jnp.array(landing_pos, dtype=jnp.int32),
                        jnp.array(board_reconstructed),
                    )
                    jumped_np = np.asarray(jumped)
                    valid_jumped = jumped_np[jumped_np[:, 0] >= 0]
                    removed = int(valid_jumped.shape[0])
                else:
                    removed = 0

                sum_jump_removed_tiles += removed

                # --- jump sequence length (contiguous jump run) ---
                if prev_action_was_jump:
                    current_seq_len += 1
                else:
                    if current_seq_len > 0:
                        num_jump_sequences += 1
                        sum_jump_sequence_lengths += current_seq_len
                    current_seq_len = 1
                prev_action_was_jump = True
            else:
                # placement / halt ends a jump sequence
                if prev_action_was_jump and current_seq_len > 0:
                    num_jump_sequences += 1
                    sum_jump_sequence_lengths += current_seq_len
                    current_seq_len = 0
                prev_action_was_jump = False

        # flush if game ended mid-sequence
        if prev_action_was_jump and current_seq_len > 0:
            num_jump_sequences += 1
            sum_jump_sequence_lengths += current_seq_len

    return {
        "num_games": batch_size,
        "total_moves": total_moves,
        "total_placements": total_placements,
        "total_jumps": total_jumps,
        "num_jump_sequences": num_jump_sequences,
        "sum_jump_sequence_lengths": sum_jump_sequence_lengths,
        "sum_jump_removed_tiles": sum_jump_removed_tiles,
        "adjacency_opportunities": adjacency_opportunities,
        "adjacency_conversions": adjacency_conversions,
    }


class ReplayBuffer:
    """Simple replay buffer for training examples."""

    def __init__(self, max_size: int = 500000, cols: int = None, augment_flip: bool = True):
        self.max_size = max_size
        self.cols = cols  # needed for policy flip augmentation
        self.augment_flip = augment_flip
        self.states = None
        self.policies = None
        self.values = None
        self.size = 0
        self.idx = 0
        self._flip_indices = None  # cached policy flip mapping
    
    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """Add examples to buffer."""
        n = len(states)
        if n == 0:
            return

        if n >= self.max_size:
            states = states[-self.max_size:]
            policies = policies[-self.max_size:]
            values = values[-self.max_size:]
            n = self.max_size
            
        if self.states is None:
            # Initialize buffers
            self.states = np.zeros((self.max_size,) + states.shape[1:], dtype=np.float32)
            self.policies = np.zeros((self.max_size,) + policies.shape[1:], dtype=np.float32)
            self.values = np.zeros(self.max_size, dtype=np.float32)
        
        # Case 1: no wrap needed
        if self.idx + n <= self.max_size:
            self.states[self.idx:self.idx + n] = states
            self.policies[self.idx:self.idx + n] = policies
            self.values[self.idx:self.idx + n] = values
            self.idx += n
        else:
            # Case 2: wrap around the circular buffer
            first_part = self.max_size - self.idx
            if first_part > 0:
                self.states[self.idx:] = states[:first_part]
                self.policies[self.idx:] = policies[:first_part]
                self.values[self.idx:] = values[:first_part]
            
            remaining = n - first_part
            if remaining > 0:
                self.states[:remaining] = states[first_part:first_part + remaining]
                self.policies[:remaining] = policies[first_part:first_part + remaining]
                self.values[:remaining] = values[first_part:first_part + remaining]
            
            self.idx = (self.idx + n) % self.max_size
        
        self.size = min(self.size + n, self.max_size)
    
    def _get_hflip_indices(self, policy_size: int) -> np.ndarray:
        """Get cached index mapping for horizontal flip of policy vector.

        Phutball action space: 2 * rows * cols + 1
        - [0, rows*cols): placements
        - [rows*cols, 2*rows*cols): jumps
        - 2*rows*cols: halt
        """
        if not hasattr(self, '_hflip_indices') or self._hflip_indices is None:
            if self.cols is not None:
                # Infer rows from policy_size = 2 * rows * cols + 1
                total_positions = (policy_size - 1) // 2
                rows = total_positions // self.cols

                flip_indices = np.arange(policy_size)

                # Flip placements (0 to total_positions-1)
                place_idx = np.arange(total_positions)
                place_row = place_idx // self.cols
                place_col = place_idx % self.cols
                place_flipped = place_row * self.cols + (self.cols - 1 - place_col)
                flip_indices[:total_positions] = place_flipped

                # Flip jumps (total_positions to 2*total_positions-1)
                flip_indices[total_positions:2*total_positions] = total_positions + place_flipped

                # Halt action stays the same (last index)
                # flip_indices[-1] = policy_size - 1  (already set)

                self._hflip_indices = flip_indices
            else:
                self._hflip_indices = None
        return self._hflip_indices

    def _get_rot180_indices(self, policy_size: int) -> np.ndarray:
        """Get cached index mapping for 180° rotation of policy vector."""
        if not hasattr(self, '_rot180_indices') or self._rot180_indices is None:
            if self.cols is not None:
                # Infer rows from policy_size = 2 * rows * cols + 1
                total_positions = (policy_size - 1) // 2
                rows = total_positions // self.cols

                rot_indices = np.arange(policy_size)

                # Rotate placements: (r, c) -> (rows-1-r, cols-1-c)
                place_idx = np.arange(total_positions)
                place_row = place_idx // self.cols
                place_col = place_idx % self.cols
                new_row = rows - 1 - place_row
                new_col = self.cols - 1 - place_col
                place_rotated = new_row * self.cols + new_col
                rot_indices[:total_positions] = place_rotated

                # Rotate jumps
                rot_indices[total_positions:2*total_positions] = total_positions + place_rotated

                # Halt action stays the same

                self._rot180_indices = rot_indices
            else:
                self._rot180_indices = None
        return self._rot180_indices

    def sample(self, batch_size: int) -> dict:
        """Sample a batch for training with augmentation (horizontal flip + 180° rotation)."""
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)

        states = self.states[indices].copy()
        policies = self.policies[indices].copy()
        values = self.values[indices].copy()

        if self.augment_flip and self.cols is not None:
            # Random augmentation: 0=none, 1=hflip, 2=rot180, 3=hflip+rot180
            aug_choice = np.random.randint(0, 4, size=len(indices))

            # Horizontal flip (choices 1 and 3)
            hflip_mask = (aug_choice == 1) | (aug_choice == 3)
            if hflip_mask.any():
                states[hflip_mask] = np.flip(states[hflip_mask], axis=-1)
                hflip_idx = self._get_hflip_indices(policies.shape[-1])
                if hflip_idx is not None:
                    policies[hflip_mask] = policies[hflip_mask][:, hflip_idx]

            # 180° rotation (choices 2 and 3)
            # This effectively shows the board from opponent's perspective
            rot180_mask = (aug_choice == 2) | (aug_choice == 3)
            if rot180_mask.any():
                # Flip state along both axes (rows and cols)
                states[rot180_mask] = np.flip(states[rot180_mask], axis=(-2, -1))
                # Swap my-goal and opp-goal channels (channels 2 and 3)
                # since opponent's perspective reverses which endzone is "mine"
                temp_ch = states[rot180_mask, 2].copy()
                states[rot180_mask, 2] = states[rot180_mask, 3]
                states[rot180_mask, 3] = temp_ch
                # Negate value (opponent's perspective has opposite outcome)
                values[rot180_mask] = -values[rot180_mask]
                # Flip policy
                rot180_idx = self._get_rot180_indices(policies.shape[-1])
                if rot180_idx is not None:
                    policies[rot180_mask] = policies[rot180_mask][:, rot180_idx]

        return {
            'states': jnp.array(states),
            'policy_targets': jnp.array(policies),
            'value_targets': jnp.array(values),
        }

    def __len__(self):
        return self.size

    def get_data(self) -> dict:
        """Get buffer contents for checkpointing."""
        if self.states is None or self.size == 0:
            return {'states': None, 'policies': None, 'values': None, 'size': 0, 'idx': 0}

        return {
            'states': self.states[:self.size],
            'policies': self.policies[:self.size],
            'values': self.values[:self.size],
            'size': self.size,
            'idx': self.idx,
        }

    def set_data(self, data: dict):
        """Restore buffer contents from checkpoint."""
        if data is None or data.get('states') is None:
            return

        states = data['states']
        policies = data['policies']
        values = data['values']
        n = len(states)

        # Initialize arrays if needed
        if self.states is None:
            self.states = np.zeros((self.max_size,) + states.shape[1:], dtype=np.float32)
            self.policies = np.zeros((self.max_size,) + policies.shape[1:], dtype=np.float32)
            self.values = np.zeros(self.max_size, dtype=np.float32)

        # Copy data into buffer
        self.states[:n] = states
        self.policies[:n] = policies
        self.values[:n] = values
        self.size = n
        self.idx = data.get('idx', n % self.max_size)


# --------============
# Tests
# --------============

def test_batched_reset():
    """Test batched environment reset."""
    env_config = EnvConfig(rows=9, cols=9)
    batch_size = 8
    
    states = batched_reset(env_config, batch_size)
    
    assert states.board.shape == (batch_size, 9, 9), f"Board shape: {states.board.shape}"
    assert states.ball_pos.shape == (batch_size, 2), f"Ball pos shape: {states.ball_pos.shape}"
    # Check jump sequence fields
    assert states.jump_sequence.shape == (batch_size, MAX_JUMP_SEQUENCE_LENGTH, 2), \
        f"Jump sequence shape: {states.jump_sequence.shape}"
    assert states.jump_sequence_length.shape == (batch_size,), \
        f"Jump sequence length shape: {states.jump_sequence_length.shape}"
    
    print("✓ Batched reset works")


def test_batched_games():
    """Test playing batched games."""
    from network import create_network, init_network
    
    env_config = EnvConfig(rows=9, cols=9)
    network = create_network(rows=9, cols=9, num_channels=32, num_res_blocks=2)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    variables = init_network(init_rng, network, num_input_channels=10)
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
        max_turns=30,     # 30 turns per game
        max_moves=200,    # Memory cap
        temperature=1.0,
        temp_threshold=15,  # Drop temp after 15 moves
        temp_final=0.1,
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
    states = np.random.randn(50, 9, 9, 9).astype(np.float32)
    policies = np.random.rand(50, 163).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = np.random.uniform(-1, 1, 50).astype(np.float32)
    
    buffer.add(states, policies, values)
    assert len(buffer) == 50
    
    # Sample
    batch = buffer.sample(16)
    assert batch['states'].shape == (16, 9, 9, 9)
    
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
    
    variables = init_network(init_rng, network, num_input_channels=10)
    params = {
        'network_params': variables['params'],
        'batch_stats': variables['batch_stats'],
    }
    
    # Warmup
    rng, game_rng = jax.random.split(rng)
    _ = play_games_batched(params, game_rng, network, env_config, batch_size=8, max_turns=5, max_moves=20)
    
    # Benchmark
    import time
    
    batch_sizes = [16, 64, 256]
    max_turns = 50
    max_moves = 300  # Rough estimate of moves for 50 turns
    
    print("\nBenchmark: Batched Self-Play")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        rng, game_rng = jax.random.split(rng)
        
        start = time.time()
        trajectory = play_games_batched(
            params, game_rng, network, env_config,
            batch_size=batch_size, max_turns=max_turns, max_moves=max_moves,
            temp_threshold=20, temp_final=0.1,
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