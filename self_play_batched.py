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
    actions: jnp.ndarray         # (batch, max_moves) chosen action ids


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


import mctx


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
        
        # Convert to network input
        def single_to_input(state):
            board = state.board.astype(jnp.float32)
            jump_layers = jnp.zeros((5, rows, cols), dtype=jnp.float32)
            return jnp.concatenate([board[None, :, :], jump_layers], axis=0)
        
        network_inputs = jax.vmap(single_to_input)(next_states)
        
        # Get network predictions
        variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
        policy_logits, values = network.apply(variables, network_inputs, train=False)
        
        # Get legal action mask
        def single_legal(state):
            return get_legal_actions(state, env_config)
        
        legal_mask = jax.vmap(single_legal)(next_states)
        
        # Mask illegal actions
        masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)
        
        # Check for terminal states
        terminated = next_states.terminated
        
        # Discount: 1 for ongoing, 0 for terminal
        discount = jnp.where(terminated, 0.0, 1.0)
        
        # For terminal states, set value based on winner
        # Winner from perspective of player who just moved
        # This is tricky - winner=1 means P1 won, winner=2 means P2 won
        # The "current_player" after the move is the OTHER player
        # So if winner == current_player, the player who just moved LOST
        terminal_value = jnp.where(
            next_states.winner == next_states.current_player,
            -1.0,  # Current player won = previous player (who moved) lost
            jnp.where(next_states.winner == 0, 0.0, 1.0)  # Draw or previous player won
        )
        values = jnp.where(terminated, terminal_value, values)
        
        recurrent_output = mctx.RecurrentFnOutput(
            reward=jnp.zeros_like(values),  # AlphaZero doesn't use intermediate rewards
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
    def single_to_input(state):
        board = state.board.astype(jnp.float32)
        jump_layers = jnp.zeros((5, rows, cols), dtype=jnp.float32)
        return jnp.concatenate([board[None, :, :], jump_layers], axis=0)
    
    network_inputs = jax.vmap(single_to_input)(states)
    
    # Get root network predictions
    variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
    policy_logits, values = network.apply(variables, network_inputs, train=False)
    
    # Get legal action mask
    def single_legal(state):
        return get_legal_actions(state, env_config)
    
    legal_mask = jax.vmap(single_legal)(states)
    masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)
    
    # Create root for MCTS
    root = mctx.RootFnOutput(
        prior_logits=masked_logits,
        value=values,
        embedding=states,
    )
    
    # Create recurrent function
    recurrent_fn = make_mcts_recurrent_fn(network, env_config)
    
    # Run MCTS
    rng, mcts_rng = jax.random.split(rng)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=mcts_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=32,
        gumbel_scale=1.0,
    )
    
    # Get MCTS policy (visit counts normalized)
    mcts_policy = policy_output.action_weights
    
    # Get root value from search
    root_values = policy_output.search_tree.node_values[:, 0]
    
    # Sample action based on temperature
    rng, sample_rng = jax.random.split(rng)
    
    if temperature < 0.01:
        # Greedy
        actions = jnp.argmax(mcts_policy, axis=-1)
    else:
        # Sample with temperature
        logits = jnp.log(mcts_policy + 1e-8) / temperature
        sample_rngs = jax.random.split(sample_rng, batch_size)
        actions = jax.vmap(lambda r, l: jax.random.categorical(r, l))(sample_rngs, logits)
    
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
    max_turns: int = 7200,
    max_moves: int = 50000,  # Safety cap on total moves (for memory)
    temperature: float = 1.0,
    num_simulations: int = 0,  # 0 = no MCTS, >0 = use MCTS with this many sims
) -> TrajectoryData:
    """
    Play multiple games in parallel.
    
    Args:
        params: Network parameters
        rng: Random key
        network: Neural network
        env_config: Environment config
        batch_size: Number of games to play in parallel
        max_turns: Maximum turns per game (placement or halt = 1 turn)
        max_moves: Maximum moves to store (memory cap)
        temperature: Sampling temperature
        num_simulations: MCTS simulations per move (0 = raw network policy)
        
    Returns:
        TrajectoryData with all game trajectories
    """
    action_space_size = 2 * env_config.rows * env_config.cols + 1
    num_channels = 6
    rows, cols = env_config.rows, env_config.cols
    use_mcts = num_simulations > 0
    
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
    
    def single_to_input(state):
        board = state.board.astype(jnp.float32)
        jump_layers = jnp.zeros((5, rows, cols), dtype=jnp.float32)
        return jnp.concatenate([board[None, :, :], jump_layers], axis=0)
    
    def single_legal(state):
        return get_legal_actions(state, env_config)
    
    # Play loop
    def game_step(carry, _):
        env_states, terminated, move_count, all_states, all_policies, all_players, all_actions, valid_mask, rng = carry
        
        rng, step_rng, sample_rng = jax.random.split(rng, 3)
        
        # Check if games have exceeded max_turns
        turns_exceeded = env_states.num_turns >= max_turns
        effectively_done = terminated | turns_exceeded
        active = ~effectively_done
        
        # Get current observations: (batch, 6, rows, cols)
        current_obs = jax.vmap(single_to_input)(env_states)
        current_players = env_states.current_player
        
        if use_mcts:
            # Use MCTS to get improved policy
            actions, policies, values = batched_mcts_policy(
                params, env_states, step_rng, network, env_config,
                num_simulations=num_simulations, temperature=temperature
            )
        else:
            # Use raw network policy
            # Get network predictions
            variables = {'params': params['network_params'], 'batch_stats': params['batch_stats']}
            policy_logits, values = network.apply(variables, current_obs, train=False)
            
            # Get legal action mask
            legal_mask = jax.vmap(single_legal)(env_states)
            
            # Mask illegal actions
            masked_logits = jnp.where(legal_mask == 1, policy_logits, -1e9)
            
            # Convert to probabilities
            policies = jax.nn.softmax(masked_logits / temperature)
            
            # Sample actions
            sample_rngs = jax.random.split(sample_rng, batch_size)
            actions = jax.vmap(lambda r, p: jax.random.choice(r, action_space_size, p=p))(sample_rngs, policies)
        
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
        new_env_states = jax.vmap(single_step)(env_states, actions)
        
        # For terminated games, keep old state using proper broadcasting
        new_board = jnp.where(effectively_done[:, None, None], env_states.board, new_env_states.board)
        new_ball_pos = jnp.where(effectively_done[:, None], env_states.ball_pos, new_env_states.ball_pos)
        new_current_player = jnp.where(effectively_done, env_states.current_player, new_env_states.current_player)
        new_is_jumping = jnp.where(effectively_done, env_states.is_jumping, new_env_states.is_jumping)
        new_terminated_flag = jnp.where(effectively_done, env_states.terminated, new_env_states.terminated)
        new_winner = jnp.where(effectively_done, env_states.winner, new_env_states.winner)
        new_num_turns = jnp.where(effectively_done, env_states.num_turns, new_env_states.num_turns)
        
        new_env_states = PhutballState(
            board=new_board,
            ball_pos=new_ball_pos,
            current_player=new_current_player,
            is_jumping=new_is_jumping,
            terminated=new_terminated_flag,
            winner=new_winner,
            num_turns=new_num_turns,
        )
        
        # Update termination status
        new_terminated = terminated | new_env_states.terminated
        new_move_count = move_count + active.astype(jnp.int32)
        
        carry = (new_env_states, new_terminated, new_move_count, 
                all_states, all_policies, all_players, all_actions, valid_mask, rng)
        return carry, None
    
    # Run the game loop with early stopping.
    #
    # We keep the big max_moves / max_turns caps for safety, but we do NOT
    # always run max_moves steps. Instead we stop once:
    #   - all games are effectively done (terminated or turns_exceeded), OR
    #   - we hit max_moves.
    #
    # This makes runtime scale with actual game length instead of the cap.

    def cond_fn(carry):
        (env_states, terminated, move_count,
         all_states, all_policies, all_players, all_actions, valid_mask, rng, step_idx) = carry

        # Same notion of "done" as inside game_step
        turns_exceeded = env_states.num_turns >= max_turns
        effectively_done = terminated | turns_exceeded
        any_active = jnp.any(~effectively_done)

        # Continue while we still have active games and haven't hit max_moves
        return (step_idx < max_moves) & any_active

    def body_fn(carry):
        (env_states, terminated, move_count,
         all_states, all_policies, all_players, all_actions, valid_mask, rng, step_idx) = carry

        inner_carry = (env_states, terminated, move_count,
                       all_states, all_policies, all_players, valid_mask, rng)

        # Re-use the existing game_step logic
        inner_carry, _ = game_step(inner_carry, None)

        (env_states, terminated, move_count,
         all_states, all_policies, all_players, valid_mask, rng) = inner_carry

        return (env_states, terminated, move_count,
                all_states, all_policies, all_players, all_actions, valid_mask, rng,
                step_idx + jnp.int32(1))

    initial_carry = (env_states, terminated, move_count,
                     all_states, all_policies, all_players, all_actions, valid_mask, rng)

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

    batch_size, max_moves = valid_np.shape
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    # From phutball_env_jax
    BALL = -1
    MAN = 1

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
            board_before = states_np[g, t, 0]  # channel 0 is board

            is_placement = a < total_positions
            is_jump = (total_positions <= a < 2 * total_positions)
            # halt = 2*total_positions, we don't need it explicitly here

            if is_placement:
                # --- % occupying-adjacent-tile conversion ---
                ball_pos = np.argwhere(board_before == BALL)
                if ball_pos.size > 0:
                    br, bc = ball_pos[0]

                    r0 = max(br - 1, 0)
                    r1 = min(br + 1, rows - 1)
                    c0 = max(bc - 1, 0)
                    c1 = min(bc + 1, cols - 1)
                    neighborhood = board_before[r0:r1 + 1, c0:c1 + 1]
                    has_adjacent_man = np.any(neighborhood == MAN)

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

                # --- jump length (tiles removed) ---
                if t + 1 < max_moves and valid_np[g, t + 1]:
                    board_after = states_np[g, t + 1, 0]
                    men_before = np.sum(board_before == MAN)
                    men_after = np.sum(board_after == MAN)
                    removed = max(0, int(men_before - men_after))
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
        max_turns=30,     # 30 turns per game
        max_moves=200,    # Memory cap
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
            batch_size=batch_size, max_turns=max_turns, max_moves=max_moves
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