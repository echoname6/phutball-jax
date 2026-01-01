"""
AlphaZero Training Loop for Phutball - Batched Version

Uses batched self-play for TPU efficiency.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import pickle
from mcts import MCTSConfig
import glob

from phutball_env_jax import (
    EnvConfig, reset, step, get_legal_actions, state_to_network_input,
    PhutballState, EMPTY, BALL, MAN, END_HI, END_LO, MAX_JUMP_SEQUENCE_LENGTH,
)
from network import (
    PhutballNetwork, create_network, init_network,
    create_optimizer, make_train_step_fn, predict
)
from self_play_batched import (
    play_games_batched,
    trajectory_to_training_examples,
    ReplayBuffer,
    compute_phutball_stats,
    play_match_batched,
    play_vs_random_batched, 
    batched_mcts_policy,
    batched_reset,
)


try:
    import wandb
except ImportError:
    wandb = None


# ============================================================================
# Curriculum Learning: 1-Move Winning State Generation
# ============================================================================

def generate_one_move_win_state(
    rng: jax.Array,
    env_config: EnvConfig,
    player: int = 1,
    min_jump_len: int = 1,
    max_jump_len: int = 8,
    add_noise_men: bool = True,
    max_noise_men: int = 10,
) -> Tuple[PhutballState, int]:
    """
    Procedurally generate a state where `player` can win in exactly one jump.

    Generates diverse scenarios:
    - Vertical (column) jumps of varying depths
    - Diagonal jumps (both directions) of varying depths
    - Landing on different positions across the endzone width
    - Sometimes jumping over men placed ON the goal line rows
    - Noise men placed elsewhere that DON'T block the winning jump

    Args:
        rng: JAX random key
        env_config: Environment configuration
        player: Which player is about to win (1 or 2)
        min_jump_len: Minimum number of men to jump over (contiguous)
        max_jump_len: Maximum number of men to jump over
        add_noise_men: Whether to add random men elsewhere on the board
        max_noise_men: Maximum number of noise men to add

    Returns:
        state: PhutballState where current_player can win in one jump
        winning_action: The action index that wins the game
    """
    rows, cols = env_config.rows, env_config.cols

    # Split RNG for each random choice
    rng, landing_col_rng, jump_len_rng, dir_rng, depth_rng, noise_rng = jax.random.split(rng, 6)

    # P1 wins by reaching row >= rows-2 (bottom endzone)
    # P2 wins by reaching row <= 1 (top endzone)

    # Directions toward each player's goal
    if player == 1:
        # P1 moves down (increasing row): vertical and both diagonals
        directions = [(1, 0), (1, -1), (1, 1)]  # down, down-left, down-right
        endzone_rows = [rows - 2, rows - 1]  # Can land on either endzone row
    else:
        # P2 moves up (decreasing row): vertical and both diagonals
        directions = [(-1, 0), (-1, -1), (-1, 1)]  # up, up-left, up-right
        endzone_rows = [0, 1]  # Can land on either endzone row

    # Choose direction (vertical or diagonal)
    dir_idx = int(jax.random.randint(dir_rng, (), 0, len(directions)))
    dr, dc = directions[dir_idx]

    # Choose which endzone row to land on (sometimes land deeper into endzone)
    depth_choice = int(jax.random.randint(depth_rng, (), 0, len(endzone_rows)))
    landing_row = endzone_rows[depth_choice]

    # Choose landing column (any valid column for the endzone)
    landing_col = int(jax.random.randint(landing_col_rng, (), 0, cols))

    # For diagonal jumps, we need to ensure landing_col is reachable
    # Adjust if diagonal would go out of bounds
    if dc != 0:
        # Calculate max jump length that keeps us in bounds
        if dc > 0:
            max_possible_len = cols - 1 - landing_col  # rightward diagonal
        else:
            max_possible_len = landing_col  # leftward diagonal
        # Also limit by board height
        if player == 1:
            max_by_height = landing_row - 2  # Can't start in endzone (rows 0,1 are P2's)
        else:
            max_by_height = (rows - 1) - landing_row - 2  # Can't start in P1's endzone
        max_jump_len = min(max_jump_len, max_possible_len, max_by_height)
        if max_jump_len < min_jump_len:
            # Fallback to vertical if diagonal not possible
            dc = 0
            max_jump_len = max_by_height

    # Choose jump length (number of men to jump over)
    actual_max = max(min_jump_len, min(max_jump_len, 8))
    jump_len = int(jax.random.randint(jump_len_rng, (), min_jump_len, actual_max + 1))

    # Calculate ball position: ball + (jump_len + 1) * direction = landing
    # So ball = landing - (jump_len + 1) * direction
    ball_row = landing_row - (jump_len + 1) * dr
    ball_col = landing_col - (jump_len + 1) * dc

    # Validate ball position is not in an endzone and is on the board
    if ball_col < 0 or ball_col >= cols:
        # Fallback to vertical
        dc = 0
        ball_col = landing_col
        ball_row = landing_row - (jump_len + 1) * dr

    # Make sure ball is in playable area (not in endzones)
    if player == 1:
        # Ball shouldn't be in P2's endzone (rows 0,1) or P1's endzone (rows-2, rows-1)
        if ball_row <= 1 or ball_row >= rows - 2:
            # Reduce jump length to fit
            if ball_row <= 1:
                ball_row = 2
                jump_len = (landing_row - ball_row) // abs(dr) - 1
            else:
                # This shouldn't happen for P1 moving down, but safety check
                jump_len = max(1, jump_len - 1)
                ball_row = landing_row - (jump_len + 1) * dr
    else:
        # Ball shouldn't be in P2's endzone or P1's endzone
        if ball_row <= 1 or ball_row >= rows - 2:
            if ball_row >= rows - 2:
                ball_row = rows - 3
                jump_len = (ball_row - landing_row) // abs(dr) - 1
            else:
                jump_len = max(1, jump_len - 1)
                ball_row = landing_row - (jump_len + 1) * dr

    # Ensure jump_len is at least 1
    jump_len = max(1, jump_len)

    # Recalculate landing to be consistent
    landing_row = ball_row + (jump_len + 1) * dr
    landing_col = ball_col + (jump_len + 1) * dc

    # Create the board
    board = np.zeros((rows, cols), dtype=np.int32)

    # Set end zones (these are the tile markers, men can still be placed on them)
    board[0, :] = END_HI
    board[1, :] = END_HI
    board[rows - 2, :] = END_LO
    board[rows - 1, :] = END_LO

    # Place ball (not in endzone)
    if 2 <= ball_row < rows - 2:
        board[ball_row, ball_col] = BALL
    else:
        # Safety: place ball in valid area
        ball_row = rows // 2
        board[ball_row, ball_col] = BALL
        # Recalculate everything for safety
        jump_len = 1
        landing_row = ball_row + (jump_len + 1) * dr
        landing_col = ball_col

    # Track positions used by the winning jump path (ball + men + landing)
    jump_path_positions = set()
    jump_path_positions.add((ball_row, ball_col))
    jump_path_positions.add((landing_row, landing_col))

    # Place men to jump over (contiguous line from ball toward landing)
    for i in range(1, jump_len + 1):
        man_row = ball_row + i * dr
        man_col = ball_col + i * dc
        if 0 <= man_row < rows and 0 <= man_col < cols:
            # Place man (even on endzone tiles - they get removed during jump)
            board[man_row, man_col] = MAN
            jump_path_positions.add((man_row, man_col))

    # Add noise men that DON'T block the winning jump path
    if add_noise_men:
        num_noise = int(jax.random.randint(noise_rng, (), 0, max_noise_men + 1))
        noise_rng_keys = jax.random.split(noise_rng, max(1, num_noise * 2))
        noise_placed = 0
        attempts = 0
        max_attempts = num_noise * 10

        while noise_placed < num_noise and attempts < max_attempts:
            nr_rng = noise_rng_keys[attempts % len(noise_rng_keys)]
            nc_rng = jax.random.split(nr_rng)[0]

            # Random position in playable area only (not endzone rows)
            nr = int(jax.random.randint(nr_rng, (), 2, rows - 2))
            nc = int(jax.random.randint(nc_rng, (), 0, cols))

            # Don't place on the jump path or existing pieces
            if (nr, nc) not in jump_path_positions:
                if board[nr, nc] == EMPTY:
                    board[nr, nc] = MAN
                    noise_placed += 1

            attempts += 1

    # Create state
    board_jax = jnp.array(board, dtype=jnp.int32)
    jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)

    state = PhutballState(
        board=board_jax,
        ball_pos=jnp.array([ball_row, ball_col], dtype=jnp.int32),
        current_player=jnp.array(player, dtype=jnp.int32),
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
        num_turns=jnp.array(0, dtype=jnp.int32),
        jump_sequence=jump_sequence,
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )

    # Calculate winning action
    # Jump actions are indexed as: total_positions + (landing_row * cols + landing_col)
    total_positions = rows * cols
    winning_action = total_positions + landing_row * cols + landing_col

    return state, winning_action


def generate_n_move_win_state(
    rng: jax.Array,
    env_config: EnvConfig,
    num_jumps: int = 2,
    player: int = 1,
    min_jump_len: int = 1,
    max_jump_len: int = 3,
    add_noise_men: bool = True,
    max_noise_men: int = 8,
) -> Tuple[PhutballState, List[int]]:
    """
    Generate a state where `player` can win in exactly `num_jumps` jumps (one turn).

    Works backwards from the endzone:
    1. Pick endzone landing position
    2. For each jump (from last to first), calculate the previous position
    3. Place men for each jump segment
    4. First position becomes the ball

    Args:
        rng: JAX random key
        env_config: Environment configuration
        num_jumps: Number of jumps required to win (1, 2, 3, or 4)
        player: Which player is about to win (1 or 2)
        min_jump_len: Minimum men to jump over per jump
        max_jump_len: Maximum men to jump over per jump
        add_noise_men: Whether to add random noise men
        max_noise_men: Maximum noise men to add

    Returns:
        state: PhutballState where current_player can win in num_jumps jumps
        actions: List of jump actions [first_action, second_action, ...]
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    # Ensure jump chain can fit in available space
    # Playable rows = rows - 4 (excluding 2 rows per endzone)
    # Each jump uses (jump_len + 1) rows, so worst case = num_jumps * (max_jump_len + 1)
    playable_rows = rows - 4
    worst_case_rows = num_jumps * (max_jump_len + 1)
    if worst_case_rows > playable_rows - 1:  # -1 for ball position buffer
        # Reduce max_jump_len to fit
        adjusted_max = (playable_rows - 1) // num_jumps - 1
        max_jump_len = max(min_jump_len, adjusted_max)

    # Directions toward goal
    if player == 1:
        directions = [(1, 0), (1, -1), (1, 1)]  # down, down-left, down-right
        endzone_rows = [rows - 2, rows - 1]
    else:
        directions = [(-1, 0), (-1, -1), (-1, 1)]  # up, up-left, up-right
        endzone_rows = [0, 1]

    # We'll build the jump chain backwards: endzone <- pos_n-1 <- ... <- pos_1 <- ball
    # positions[0] = ball, positions[1] = after jump 1, ..., positions[n] = endzone
    positions = []  # Will hold (row, col) for each position
    jump_lengths = []  # Length of each jump
    jump_dirs = []  # Direction of each jump

    # Split RNG for all random choices
    rng_keys = jax.random.split(rng, 3 * num_jumps + 3)
    rng_idx = 0

    # Start with endzone position
    end_col = int(jax.random.randint(rng_keys[rng_idx], (), 0, cols))
    rng_idx += 1
    end_row = endzone_rows[int(jax.random.randint(rng_keys[rng_idx], (), 0, 2))]
    rng_idx += 1

    positions.append((end_row, end_col))

    # Work backwards from endzone to ball
    for jump_idx in range(num_jumps):
        # Current position (where this jump lands)
        curr_row, curr_col = positions[-1]

        # Choose direction for this jump
        dir_idx = int(jax.random.randint(rng_keys[rng_idx], (), 0, len(directions)))
        rng_idx += 1
        dr, dc = directions[dir_idx]

        # Choose jump length
        jump_len = int(jax.random.randint(rng_keys[rng_idx], (), min_jump_len, max_jump_len + 1))
        rng_idx += 1

        # Calculate previous position (where the ball/intermediate was before this jump)
        # prev + (jump_len + 1) * dir = curr
        # prev = curr - (jump_len + 1) * dir
        prev_row = curr_row - (jump_len + 1) * dr
        prev_col = curr_col - (jump_len + 1) * dc

        # Validate column bounds
        if prev_col < 0 or prev_col >= cols:
            # Fallback to vertical
            dc = 0
            prev_col = curr_col
            prev_row = curr_row - (jump_len + 1) * dr

        # For intermediate positions (not the ball), must be in playable area
        # For the ball (last iteration), also must be in playable area
        if prev_row <= 1 or prev_row >= rows - 2:
            # Need to reduce jump length to fit
            if player == 1:
                # Moving down, prev must be above curr
                available_space = curr_row - 2  # rows 0,1 are P2 endzone
            else:
                # Moving up, prev must be below curr
                available_space = (rows - 3) - curr_row  # rows-2, rows-1 are P1 endzone

            max_possible_len = available_space // abs(dr) - 1 if dr != 0 else available_space - 1
            jump_len = max(1, min(jump_len, max(1, max_possible_len)))

            prev_row = curr_row - (jump_len + 1) * dr
            prev_col = curr_col - (jump_len + 1) * dc

            if prev_col < 0 or prev_col >= cols:
                dc = 0
                prev_col = curr_col
                prev_row = curr_row - (jump_len + 1) * dr

        # Final bounds check - if still out of bounds, use fallback
        if prev_row <= 1 or prev_row >= rows - 2:
            # Place at a safe distance from current position using VERTICAL jump
            # Ensure prev is exactly 2 rows away for a minimal valid jump (jump_len=1)
            dc = 0
            prev_col = curr_col
            if player == 1:
                # P1 moves down (toward higher rows), so prev should be above curr
                prev_row = curr_row - 2
                dr = 1  # Ensure correct direction
            else:
                # P2 moves up (toward lower rows), so prev should be below curr
                prev_row = curr_row + 2
                dr = -1  # Ensure correct direction
            jump_len = 1

            # If still out of bounds, this jump chain isn't feasible - clamp to valid
            prev_row = max(2, min(rows - 3, prev_row))

            # Recalculate jump_len based on actual distance
            actual_distance = abs(curr_row - prev_row)
            if actual_distance >= 2:
                jump_len = actual_distance - 1
            else:
                # Can't make a valid jump, set minimal distance
                if player == 1:
                    prev_row = curr_row - 2
                else:
                    prev_row = curr_row + 2
                prev_row = max(2, min(rows - 3, prev_row))
                jump_len = max(1, abs(curr_row - prev_row) - 1)

        positions.append((prev_row, prev_col))
        jump_lengths.append(jump_len)
        jump_dirs.append((dr, dc))

    # Reverse to get ball -> ... -> endzone order
    positions = positions[::-1]  # [ball, pos1, pos2, ..., endzone]
    jump_lengths = jump_lengths[::-1]
    jump_dirs = jump_dirs[::-1]

    ball_row, ball_col = positions[0]

    # === BUILD THE BOARD ===
    board = np.zeros((rows, cols), dtype=np.int32)

    # Set end zones
    board[0, :] = END_HI
    board[1, :] = END_HI
    board[rows - 2, :] = END_LO
    board[rows - 1, :] = END_LO

    # Track all positions used by jump paths
    jump_path_positions = set()
    for pos in positions:
        jump_path_positions.add(pos)

    # Place ball
    if 2 <= ball_row < rows - 2 and 0 <= ball_col < cols:
        board[ball_row, ball_col] = BALL
    else:
        # Safety fallback
        ball_row = rows // 2
        ball_col = cols // 2
        board[ball_row, ball_col] = BALL
        positions[0] = (ball_row, ball_col)

    # Place men for each jump segment
    for jump_idx in range(num_jumps):
        start_row, start_col = positions[jump_idx]
        dr, dc = jump_dirs[jump_idx]
        jump_len = jump_lengths[jump_idx]

        for i in range(1, jump_len + 1):
            mr = start_row + i * dr
            mc = start_col + i * dc
            # Men can be placed in first endzone rows (1 and rows-2) but not deep rows (0 and rows-1)
            if 1 <= mr <= rows - 2 and 0 <= mc < cols:
                board[mr, mc] = MAN
                jump_path_positions.add((mr, mc))

    # Add noise men that don't block any jump path
    if add_noise_men:
        noise_rng = rng_keys[rng_idx] if rng_idx < len(rng_keys) else rng
        num_noise = int(jax.random.randint(noise_rng, (), 0, max_noise_men + 1))
        noise_keys = jax.random.split(noise_rng, max(1, num_noise * 2))
        noise_placed = 0
        attempts = 0

        while noise_placed < num_noise and attempts < num_noise * 10:
            nr_rng = noise_keys[attempts % len(noise_keys)]
            nc_rng = jax.random.split(nr_rng)[0]
            # Random position in playable area (rows 1 to rows-2, not deep endzone rows 0 and rows-1)
            nr = int(jax.random.randint(nr_rng, (), 1, rows - 1))
            nc = int(jax.random.randint(nc_rng, (), 0, cols))

            if (nr, nc) not in jump_path_positions:
                if board[nr, nc] == EMPTY:
                    board[nr, nc] = MAN
                    noise_placed += 1
            attempts += 1

    # Create state
    board_jax = jnp.array(board, dtype=jnp.int32)
    jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)

    state = PhutballState(
        board=board_jax,
        ball_pos=jnp.array([ball_row, ball_col], dtype=jnp.int32),
        current_player=jnp.array(player, dtype=jnp.int32),
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
        num_turns=jnp.array(0, dtype=jnp.int32),
        jump_sequence=jump_sequence,
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )

    # Calculate actions for each jump
    actions = []
    for i in range(1, len(positions)):  # Skip ball position, get landing positions
        land_row, land_col = positions[i]
        action = total_positions + land_row * cols + land_col
        actions.append(action)

    return state, actions


# Wrapper for backwards compatibility
def generate_two_move_win_state(
    rng: jax.Array,
    env_config: EnvConfig,
    player: int = 1,
    min_jump_len: int = 1,
    max_jump_len: int = 4,
    add_noise_men: bool = True,
    max_noise_men: int = 8,
) -> Tuple[PhutballState, int, int]:
    """Generate a 2-jump winning state. Wrapper around generate_n_move_win_state."""
    state, actions = generate_n_move_win_state(
        rng, env_config, num_jumps=2, player=player,
        min_jump_len=min_jump_len, max_jump_len=max_jump_len,
        add_noise_men=add_noise_men, max_noise_men=max_noise_men
    )
    return state, actions[0], actions[1] if len(actions) > 1 else actions[0]


def generate_curriculum_batch(
    rng: jax.Array,
    env_config: EnvConfig,
    batch_size: int,
    jump_distribution: List[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of winning states for curriculum learning.

    Includes a mix of N-jump winning states where N can be 1, 2, 3, or 4.
    Each state has value=1 (current player wins) and policy targeting the first jump.

    Args:
        rng: JAX random key
        env_config: Environment configuration
        batch_size: Number of examples to generate
        jump_distribution: List of 4 floats [p1, p2, p3, p4] for probability of
                          1-jump, 2-jump, 3-jump, 4-jump examples.
                          Default: [0.4, 0.3, 0.2, 0.1]

    Returns:
        states: (batch_size, 6, rows, cols) - network input format
        policy_targets: (batch_size, action_space_size) - one-hot on first action
        value_targets: (batch_size,) - all +1 (current player wins)
    """
    if jump_distribution is None:
        jump_distribution = [0.4, 0.3, 0.2, 0.1]  # Default: more 1-jump, fewer 4-jump

    # Normalize distribution
    total = sum(jump_distribution)
    jump_distribution = [p / total for p in jump_distribution]

    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    states_list = []
    policy_list = []

    # Calculate counts for each jump type
    counts = []
    remaining = batch_size
    for i, prob in enumerate(jump_distribution[:-1]):
        count = int(batch_size * prob)
        counts.append(count)
        remaining -= count
    counts.append(remaining)  # Last category gets the remainder

    # Generate examples for each jump count (1, 2, 3, 4)
    for num_jumps, count in enumerate(counts, start=1):
        for _ in range(count):
            rng, state_rng, player_rng = jax.random.split(rng, 3)
            player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2

            if num_jumps == 1:
                # Use optimized 1-jump generator
                state, first_action = generate_one_move_win_state(
                    state_rng, env_config, player=player
                )
            else:
                # Use N-jump generator for 2, 3, 4 jumps
                state, actions = generate_n_move_win_state(
                    state_rng, env_config, num_jumps=num_jumps, player=player
                )
                first_action = actions[0]

            obs = state_to_network_input(state, env_config)
            states_list.append(np.array(obs))

            policy = np.zeros(action_space_size, dtype=np.float32)
            policy[first_action] = 1.0
            policy_list.append(policy)

    states = np.stack(states_list, axis=0)
    policies = np.stack(policy_list, axis=0)
    values = np.ones(batch_size, dtype=np.float32)  # Current player wins

    return states, policies, values


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Environment
    rows: int = 21
    cols: int = 15
    
    # Network architecture
    num_channels: int = 128
    num_res_blocks: int = 8
    
    # Self-play (batched)
    batch_size_games: int = 64       # Games played in parallel
    max_turns_per_game: int = 2048   # Max turns before game ends
    max_moves_per_game: int = 4096   # Memory cap on moves stored
    temperature: float = 1.0         # Initial sampling temperature (exploration)
    temp_threshold: int = 30         # Moves before temperature drops
    temp_final: float = 0.1          # Temperature after threshold (exploitation)
    num_simulations: int = 50        # MCTS simulations per move (0 = raw network)
    
    # Training
    batch_size_train: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100
    
    # Replay buffer
    buffer_size: int = 500000
    min_buffer_size: int = 1000  # Min examples before training starts
    
    # Iterations
    num_iterations: int = 500
    games_per_iteration: int = 256   # Total games per iteration (multiple batches)
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10  # iterations
    
    # Logging
    log_every: int = 1

    eval_enable: bool = False
    eval_max_prev_checkpoints: int = 5
    eval_games_per_color: int = 5      # 5 as P1 + 5 as P2 per opponent
    eval_num_simulations: int = 128    # MCTS sims per move during eval
    eval_max_moves: int = 2048         # cutoff to avoid marathon eval games
    eval_temperature: float = 0.0

    eval_vs_random_games: int = 100
    eval_vs_random_threshold: Optional[float] = None
    stop_when_beating_random: bool = False

    # Curriculum learning (N-jump winning states)
    curriculum_enabled: bool = True
    curriculum_initial_ratio: float = 0.5    # Initial fraction of batch from curriculum
    curriculum_final_ratio: float = 0.0      # Final fraction (decays to this)
    curriculum_decay_iterations: int = 100   # Iterations to decay from initial to final
    curriculum_min_jump_len: int = 1         # Min men to jump over per jump
    curriculum_max_jump_len: int = 3         # Max men to jump over per jump (contiguous)
    curriculum_add_noise: bool = True        # Add random men to curriculum states
    # Jump distribution: [p1, p2, p3, p4] = probability for 1,2,3,4-jump examples
    curriculum_jump_distribution: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)

    use_wandb: bool = False
    wandb_project: str = "phutball-az"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class AlphaZeroTrainer:
    """Main training class with batched self-play."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Environment config
        self.env_config = EnvConfig(rows=config.rows, cols=config.cols, max_turns=config.max_turns_per_game)
        
        # Create network
        self.network = create_network(
            rows=config.rows,
            cols=config.cols,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        )
        
        # Optimizer
        self.optimizer = create_optimizer(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize network
        self._init_network()
        
        # Create JIT-compiled train step
        self.train_step_fn = make_train_step_fn(self.network, self.optimizer)
        
        # Metrics
        self.iteration = 0
        self.total_games = 0
        self.total_examples = 0
        self.metrics_history = []
        self.last_self_play_stats = None

        self.wandb_run = None
        if self.config.use_wandb:
            if wandb is None:
                print("WARNING: use_wandb=True but wandb is not installed; disabling wandb.")
                self.config.use_wandb = False
            else:
                run_name = (
                    self.config.wandb_run_name
                    or f"phutball_az_{int(time.time())}"
                )
                cfg = asdict(self.config)
                # asdict(config) is all simple types, so fine for wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    config=cfg,
                    mode=self.config.wandb_mode,
                )
        
    def _init_network(self):
        """Initialize network parameters."""
        self.rng, init_rng = jax.random.split(self.rng)
        variables = init_network(init_rng, self.network, num_input_channels=6)
        
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.opt_state = self.optimizer.init(self.params)
    
    def get_network_params(self) -> dict:
        """Get params dict for self-play."""
        return {
            'network_params': self.params,
            'batch_stats': self.batch_stats,
        }
    
    def run_self_play(self) -> int:
        """Run batched self-play games. Returns number of new examples."""
        start_time = time.time()
        
        total_states = []
        total_policies = []
        total_values = []

        stats_totals = {
            "num_games": 0,
            "total_moves": 0,
            "total_placements": 0,
            "total_jumps": 0,
            "num_jump_sequences": 0,
            "sum_jump_sequence_lengths": 0,
            "sum_jump_removed_tiles": 0,
            "adjacency_opportunities": 0,
            "adjacency_conversions": 0
        }

        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        # Run multiple batches to get desired number of games
        num_batches = self.config.games_per_iteration // self.config.batch_size_games
        
        for batch_idx in range(num_batches):
            self.rng, game_rng = jax.random.split(self.rng)
            
            # Play games in parallel with temperature scheduling
            trajectory = play_games_batched(
                params=self.get_network_params(),
                rng=game_rng,
                network=self.network,
                env_config=self.env_config,
                batch_size=self.config.batch_size_games,
                max_turns=self.config.max_turns_per_game,
                max_moves=self.config.max_moves_per_game,
                temperature=self.config.temperature,
                temp_threshold=self.config.temp_threshold,
                temp_final=self.config.temp_final,
                num_simulations=self.config.num_simulations,
            )

            batch_stats = compute_phutball_stats(trajectory, self.env_config)
            for k in stats_totals:
                stats_totals[k] += batch_stats[k]
            
            winners_np = np.array(trajectory.winners)
            p1_wins += int(np.sum(winners_np == 1))
            p2_wins += int(np.sum(winners_np == 2))
            draws += int(np.sum(winners_np == 0))
            
            # Convert to training examples
            states, policies, values = trajectory_to_training_examples(trajectory)
            
            total_states.append(states)
            total_policies.append(policies)
            total_values.append(values)
        
        # Concatenate all examples
        if total_states:
            all_states = np.concatenate(total_states, axis=0)
            all_policies = np.concatenate(total_policies, axis=0)
            all_values = np.concatenate(total_values, axis=0)
            
            self.replay_buffer.add(all_states, all_policies, all_values)
            num_examples = len(all_states)
        else:
            num_examples = 0
        
        elapsed = time.time() - start_time
        games_total = num_batches * self.config.batch_size_games
        self.total_games += games_total
        self.total_examples += num_examples

        # ---- derive averages from stats_totals ----
        if stats_totals["num_games"] > 0:
            ng = stats_totals["num_games"]
            total_moves = stats_totals["total_moves"]
            total_placements = stats_totals["total_placements"]
            total_jumps = stats_totals["total_jumps"]
            num_seq = stats_totals["num_jump_sequences"]
            seq_len_sum = stats_totals["sum_jump_sequence_lengths"]
            removed_sum = stats_totals["sum_jump_removed_tiles"]
            adj_ops = stats_totals["adjacency_opportunities"]
            adj_conv = stats_totals["adjacency_conversions"]

            avg_moves_per_game = total_moves / ng
            avg_placements_per_jump = (
                float(total_placements) / total_jumps if total_jumps > 0 else 0.0
            )
            avg_jump_seq_len = (
                float(seq_len_sum) / num_seq if num_seq > 0 else 0.0
            )
            avg_jump_length = (
                float(removed_sum) / total_jumps if total_jumps > 0 else 0.0
            )
            adj_conv_rate = (
                float(adj_conv) / adj_ops if adj_ops > 0 else 0.0
            )
        else:
            avg_moves_per_game = avg_placements_per_jump = 0.0
            avg_jump_seq_len = avg_jump_length = adj_conv_rate = 0.0

        self.last_self_play_stats = {
            "games_total": games_total,
            "avg_moves_per_game": avg_moves_per_game,
            "avg_placements_per_jump": avg_placements_per_jump,
            "avg_jump_sequence_len": avg_jump_seq_len,
            "avg_jump_length": avg_jump_length,
            "adjacent_conversion_rate": adj_conv_rate,
        }
        
        games_per_sec = games_total / elapsed if elapsed > 0 else 0
        print(
            f"  Self-play: {num_examples} examples from {games_total} games "
            f"({games_per_sec:.1f} games/sec, {elapsed:.1f}s) | "
            f"W1/W2/D={p1_wins}/{p2_wins}/{draws} | "
            f"avg_moves/game={avg_moves_per_game:.1f}, "
            f"avg_jump_seq={avg_jump_seq_len:.2f}, "
            f"avg_jump_len={avg_jump_length:.2f}, "
            f"adj_conv={adj_conv_rate*100:.1f}%"
        )

        return num_examples
    
    def get_curriculum_ratio(self) -> float:
        """Calculate current curriculum ratio based on iteration (linear decay)."""
        if not self.config.curriculum_enabled:
            return 0.0

        if self.iteration >= self.config.curriculum_decay_iterations:
            return self.config.curriculum_final_ratio

        # Linear interpolation from initial to final
        progress = self.iteration / self.config.curriculum_decay_iterations
        ratio = self.config.curriculum_initial_ratio + progress * (
            self.config.curriculum_final_ratio - self.config.curriculum_initial_ratio
        )
        return ratio

    def run_training(self) -> dict:
        """Run training steps. Returns average metrics."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            print(f"  Skipping training: buffer has {len(self.replay_buffer)}/{self.config.min_buffer_size} examples")
            return {}

        start_time = time.time()
        metrics_sum = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0,
            'policy_entropy': 0.0,
            'mcts_entropy': 0.0,
            'kl_divergence': 0.0,
        }

        # Calculate curriculum mix ratio
        curriculum_ratio = self.get_curriculum_ratio()
        curriculum_size = int(self.config.batch_size_train * curriculum_ratio)
        replay_size = self.config.batch_size_train - curriculum_size

        for step_idx in range(self.config.train_steps_per_iteration):
            self.rng, step_rng, curriculum_rng = jax.random.split(self.rng, 3)

            # Sample from replay buffer
            if replay_size > 0:
                replay_batch = self.replay_buffer.sample(replay_size)
                states = replay_batch['states']
                policies = replay_batch['policy_targets']
                values = replay_batch['value_targets']
            else:
                states = np.empty((0, 6, self.config.rows, self.config.cols), dtype=np.float32)
                policies = np.empty((0, 2 * self.config.rows * self.config.cols + 1), dtype=np.float32)
                values = np.empty((0,), dtype=np.float32)

            # Generate curriculum examples
            if curriculum_size > 0:
                curr_states, curr_policies, curr_values = generate_curriculum_batch(
                    curriculum_rng, self.env_config, curriculum_size,
                    jump_distribution=list(self.config.curriculum_jump_distribution)
                )
                # Concatenate with replay batch
                states = np.concatenate([states, curr_states], axis=0)
                policies = np.concatenate([policies, curr_policies], axis=0)
                values = np.concatenate([values, curr_values], axis=0)

            # Create combined batch
            batch = {
                'states': states,
                'policy_targets': policies,
                'value_targets': values,
            }

            # Train step
            self.params, self.batch_stats, self.opt_state, metrics = self.train_step_fn(
                self.params, self.batch_stats, self.opt_state, batch, step_rng
            )

            # Accumulate metrics
            for k, v in metrics.items():
                metrics_sum[k] += float(v)

        elapsed = time.time() - start_time
        steps_per_sec = self.config.train_steps_per_iteration / elapsed

        # Average metrics
        avg_metrics = {k: v / self.config.train_steps_per_iteration for k, v in metrics_sum.items()}

        curriculum_pct = curriculum_ratio * 100
        print(f"  Training: {self.config.train_steps_per_iteration} steps "
              f"({steps_per_sec:.1f} steps/sec) | "
              f"policy_loss: {avg_metrics['policy_loss']:.4f}, "
              f"value_loss: {avg_metrics['value_loss']:.4f} | "
              f"curriculum: {curriculum_pct:.1f}%")

        # Add curriculum ratio to metrics for logging
        avg_metrics['curriculum_ratio'] = curriculum_ratio

        return avg_metrics
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{self.iteration:06d}.pkl")
        
        checkpoint = {
            'params': self.params,
            'batch_stats': self.batch_stats,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_examples': self.total_examples,
            'config': self.config,
            'metrics_history': self.metrics_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint['params']
        self.batch_stats = checkpoint['batch_stats']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.total_examples = checkpoint['total_examples']
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        print(f"Loaded checkpoint from iteration {self.iteration}")
    
    def train(self):
        """Main training loop."""

        # Auto-resume from latest checkpoint if exists
        existing = glob.glob(os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl"))
        if existing:
            latest = max(existing, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            self.load_checkpoint(latest)
            self.iteration += 1
        print(f"Resuming from iteration {self.iteration}")
        print("=" * 60)
        print("AlphaZero Training for Phutball (Batched)")
        print("=" * 60)
        print(f"Board: {self.config.rows}x{self.config.cols}")
        print(f"Network: {self.config.num_channels} channels, {self.config.num_res_blocks} res blocks")
        print(f"Self-play: {self.config.games_per_iteration} games/iter (batch={self.config.batch_size_games})")
        print(f"Training: {self.config.train_steps_per_iteration} steps/iter (batch={self.config.batch_size_train})")
        print(f"Temperature: {self.config.temperature} -> {self.config.temp_final} after {self.config.temp_threshold} moves")
        print(f"Devices: {jax.devices()}")
        print("=" * 60)
        print()
        
        for iteration in range(self.iteration, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()
            
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print("-" * 40)
            
            # Self-play
            num_examples = self.run_self_play()
            
            # Training
            metrics = self.run_training()
            
            # Record metrics
            if metrics:
                self.metrics_history.append({
                    'iteration': iteration,
                    'total_games': self.total_games,
                    'total_examples': self.total_examples,
                    'buffer_size': len(self.replay_buffer),
                    **metrics,
                })
            
            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint()

                if self.config.eval_enable:
                    # First check vs random
                    win_rate, stats = self.evaluate_vs_random_batched()
                    threshold = self.config.eval_vs_random_threshold
                    if threshold is not None:
                        # Threshold is active: gate the expensive comparison
                        if win_rate >= threshold:
                            print(
                                f"  [eval] Win rate {win_rate:.1%} >= {threshold:.0%}, "
                                f"running checkpoint comparison..."
                            )
                            self.evaluate_current_checkpoint_vs_recent()
                        else:
                            print(
                                f"  [eval] Win rate {win_rate:.1%} < {threshold:.0%}, "
                                f"skipping checkpoint comparison"
                            )

                        # Optional early stop if we are configured to do so
                        if self.config.stop_when_beating_random and win_rate >= threshold:
                            print(
                                f"  [train] Reached eval_vs_random_threshold={threshold:.0%}, "
                                f"stopping training early for this board size."
                            )
                            break
                    else:
                        # No threshold configured: don't gate on it at all.
                        # Behave like the old code: always run checkpoint-vs-checkpoint eval.
                        print(
                            "  [eval] eval_vs_random_threshold not set; "
                            "running checkpoint comparison unconditionally."
                        )
                        self.evaluate_current_checkpoint_vs_recent()


            # iteration timing + wandb logging
            iter_time = time.time() - iter_start
            print(f"  Iteration time: {iter_time:.1f}s | Buffer: {len(self.replay_buffer)} examples")
            print()

            if metrics and self.config.use_wandb and self.wandb_run is not None:
                global_step = (iteration + 1) * self.config.train_steps_per_iteration

                log_data = {
                    "iteration": iteration,
                    "total_games": self.total_games,
                    "total_examples": self.total_examples,
                    "buffer_size": len(self.replay_buffer),
                    "selfplay/examples_per_iter": num_examples,
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "train/curriculum_ratio": metrics.get("curriculum_ratio", 0.0),
                    "time/iteration_sec": iter_time,
                }

                # Phutball-specific stats if available
                if self.last_self_play_stats is not None:
                    for k, v in self.last_self_play_stats.items():
                        log_data[f"selfplay/{k}"] = v

                wandb.log(log_data, step=global_step)

        # Final checkpoint
        self.save_checkpoint()
        print("Training complete!")
        print(f"Total games: {self.total_games}")
        print(f"Total examples: {self.total_examples}")

    
    def evaluate_current_checkpoint_vs_recent(self):
        """
        After saving the *current* checkpoint, pit it against up to the
        last K previous checkpoints using the batched evaluator.

        For each opponent:
        - games_per_color with current as P1 (home=P1, away=P2)
        - games_per_color with current as P2 (home=P2, away=P1)

        All 2 * games_per_color games vs that opponent are played in a
        single batched call to play_match_batched.
        """
        if not self.config.eval_enable:
            return

        import glob, os, pickle
        import jax
        import numpy as np

        max_prev = self.config.eval_max_prev_checkpoints
        games_per_color = self.config.eval_games_per_color
        if max_prev <= 0 or games_per_color <= 0:
            return

        pattern = os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl")
        ckpt_paths = sorted(glob.glob(pattern))
        if len(ckpt_paths) < 2:
            return  # need at least current + one opponent

        # Current is the most recently saved
        current_path = ckpt_paths[-1]
        # Up to max_prev most recent previous checkpoints
        prev_paths = ckpt_paths[-(max_prev + 1):-1]
        prev_paths = list(reversed(prev_paths))  # newest first

        # Load current params once
        with open(current_path, "rb") as f:
            current_ckpt = pickle.load(f)
        current_params = {
            "network_params": current_ckpt["params"],
            "batch_stats": current_ckpt["batch_stats"],
        }
        current_name = os.path.basename(current_path)

        # Eval hyperparams from config
        num_simulations = self.config.eval_num_simulations
        max_moves = self.config.eval_max_moves
        temperature = self.config.eval_temperature

        print(f"  [eval] Batched evaluation for {current_name} vs {len(prev_paths)} recent checkpoints...")

        rng = self.rng  # thread rng through

        for opp_path in prev_paths:
            opp_name = os.path.basename(opp_path)
            with open(opp_path, "rb") as f:
                opp_ckpt = pickle.load(f)
            opp_params = {
                "network_params": opp_ckpt["params"],
                "batch_stats": opp_ckpt["batch_stats"],
            }

            rng, match_rng = jax.random.split(rng)

            (
                total_score_home,
                total_games,
                wins_home,
                draws,
                wins_away,
                per_game_turns,
                per_game_jumps_p1,
                per_game_jumps_p2,
                per_game_jumps_total,
                per_game_removed,
                winners,
                home_is_P1,
            ) = play_match_batched(
                home_params=current_params,   # treat A as "home"
                away_params=opp_params,       # treat B as "away"
                rng=match_rng,
                network=self.network,
                env_config=self.env_config,
                games_per_color=games_per_color,
                max_moves=max_moves,
                num_simulations=num_simulations,
                temperature=temperature,
            )

            # ---- convert to numpy / python ----
            total_score_home = float(total_score_home)
            total_games = int(total_games)
            wins_home = int(wins_home)
            wins_away = int(wins_away)
            draws = int(draws)

            per_game_turns = np.asarray(per_game_turns, dtype=np.int32)
            per_game_jumps_p1 = np.asarray(per_game_jumps_p1, dtype=np.int32)
            per_game_jumps_p2 = np.asarray(per_game_jumps_p2, dtype=np.int32)
            per_game_jumps_total = np.asarray(per_game_jumps_total, dtype=np.int32)
            per_game_removed = np.asarray(per_game_removed, dtype=np.int32)
            winners = np.asarray(winners, dtype=np.int32)
            home_is_P1 = np.asarray(home_is_P1, dtype=bool)

            assert total_games == len(per_game_turns)

            # Map P1/P2 jump counts to home/away
            jumps_home = np.where(home_is_P1, per_game_jumps_p1, per_game_jumps_p2)
            jumps_away = per_game_jumps_p1 + per_game_jumps_p2 - jumps_home

            avg_score_home = total_score_home / total_games if total_games > 0 else 0.0
            avg_turns = float(per_game_turns.mean())
            avg_jumps_home = float(jumps_home.mean())
            avg_jumps_away = float(jumps_away.mean())

            # Average jump length across all jumps in this match
            jump_mask = per_game_jumps_total > 0
            if jump_mask.any():
                total_removed = int(per_game_removed[jump_mask].sum())
                total_jumps = int(per_game_jumps_total[jump_mask].sum())
                avg_jump_len = total_removed / total_jumps
            else:
                avg_jump_len = 0.0

            # ---- aggregate line ----
            print(
                f"    [eval] {current_name} (home) vs {opp_name} (away): "
                f"W-D-L = {wins_home}-{draws}-{wins_away} "
                f"({total_score_home:+.1f} / {total_games}, avg {avg_score_home:+.3f})"
            )
            print(
                f"      turns/game={avg_turns:.1f}, "
                f"jumps/game home={avg_jumps_home:.2f}, away={avg_jumps_away:.2f}, "
                f"avg_jump_len={avg_jump_len:.2f}"
            )

            # ---- per-game detailed lines ----
            for g in range(total_games):
                if winners[g] == 0:
                    outcome = "draw"
                else:
                    home_won = (
                        (winners[g] == 1 and home_is_P1[g]) or
                        (winners[g] == 2 and not home_is_P1[g])
                    )
                    outcome = "home win" if home_won else "away win"

                role_str = "home=P1,away=P2" if home_is_P1[g] else "home=P2,away=P1"

                if per_game_jumps_total[g] > 0:
                    game_avg_jump_len = per_game_removed[g] / per_game_jumps_total[g]
                else:
                    game_avg_jump_len = 0.0

                print(
                    f"        game {g:2d}: {role_str}, "
                    f"turns={per_game_turns[g]}, "
                    f"jumps_home={int(jumps_home[g])}, jumps_away={int(jumps_away[g])}, "
                    f"avg_jump_len={game_avg_jump_len:.2f}, "
                    f"result={outcome}"
                )

        # persist RNG so eval randomness keeps moving forward
        self.rng = rng
    

    def evaluate_vs_random_batched(self):
        """
        Evaluate current checkpoint vs random, side-aware.
        Returns (overall_win_rate, stats_dict).
        """
        self.rng, eval_rng = jax.random.split(self.rng)

        (
            p1_wins, p1_draws, p1_losses,
            p2_wins, p2_draws, p2_losses,
            turns,
        ) = play_vs_random_batched(
            checkpoint_params=self.get_network_params(),
            rng=eval_rng,
            network=self.network,
            env_config=self.env_config,
            num_games=self.config.eval_vs_random_games,
            max_moves=self.config.eval_max_moves,
            num_simulations=self.config.eval_num_simulations,
            temperature=self.config.eval_temperature,
        )

        # Make sure these are Python ints
        p1_wins   = int(p1_wins)
        p1_draws  = int(p1_draws)
        p1_losses = int(p1_losses)
        p2_wins   = int(p2_wins)
        p2_draws  = int(p2_draws)
        p2_losses = int(p2_losses)

        # Totals
        total_p1 = p1_wins + p1_draws + p1_losses
        total_p2 = p2_wins + p2_draws + p2_losses
        total    = total_p1 + total_p2

        # Overall (checkpoint POV)
        total_wins   = p1_wins + p2_wins
        total_draws  = p1_draws + p2_draws
        total_losses = p1_losses + p2_losses

        win_rate_overall  = total_wins / total if total > 0 else 0.0
        draw_rate_overall = total_draws / total if total > 0 else 0.0
        loss_rate_overall = total_losses / total if total > 0 else 0.0

        # Per-side win rates (checkpoint as P1 / P2)
        win_rate_p1 = p1_wins / total_p1 if total_p1 > 0 else 0.0
        win_rate_p2 = p2_wins / total_p2 if total_p2 > 0 else 0.0

        # "Score" (win - loss) / total, overall and per-side
        score_overall = (total_wins - total_losses) / total if total > 0 else 0.0
        score_p1 = (p1_wins - p1_losses) / total_p1 if total_p1 > 0 else 0.0
        score_p2 = (p2_wins - p2_losses) / total_p2 if total_p2 > 0 else 0.0
        side_bias = score_p1 - score_p2  # positive => better as P1

        avg_turns = float(jnp.mean(turns))

        print(
            "  [eval] vs random (side-aware):\n"
            f"    as P1: {p1_wins}-{p1_draws}-{p1_losses} "
            f"(win: {win_rate_p1:.1%}, score: {score_p1:.3f})\n"
            f"    as P2: {p2_wins}-{p2_draws}-{p2_losses} "
            f"(win: {win_rate_p2:.1%}, score: {score_p2:.3f})\n"
            f"    combined: win {win_rate_overall:.1%}, draw {draw_rate_overall:.1%}, "
            f"loss {loss_rate_overall:.1%}, score {score_overall:.3f}, "
            f"side_bias {side_bias:.3f}, avg turns {avg_turns:.1f}"
        )

        stats = {
            "total_games": total,
            "overall": {
                "wins": total_wins,
                "draws": total_draws,
                "losses": total_losses,
                "win_rate": win_rate_overall,
                "draw_rate": draw_rate_overall,
                "loss_rate": loss_rate_overall,
                "score": score_overall,
            },
            "p1": {
                "wins": p1_wins,
                "draws": p1_draws,
                "losses": p1_losses,
                "win_rate": win_rate_p1,
                "score": score_p1,
            },
            "p2": {
                "wins": p2_wins,
                "draws": p2_draws,
                "losses": p2_losses,
                "win_rate": win_rate_p2,
                "score": score_p2,
            },
            "side_bias": side_bias,
            "avg_turns": avg_turns,
        }

        return win_rate_overall, stats


    def run_round_robin(
        self,
        games_per_color: int = 5,
        max_checkpoints: int | None = None,
        num_simulations: int | None = None,
        max_moves: int | None = None,
        temperature: float | None = None,
    ):
        import glob, os, pickle, numpy as np
        import jax

        if num_simulations is None:
            num_simulations = self.config.eval_num_simulations
        if max_moves is None:
            max_moves = self.config.eval_max_moves
        if temperature is None:
            temperature = self.config.eval_temperature

        pattern = os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl")
        ckpt_paths = sorted(glob.glob(pattern))
        if max_checkpoints is not None:
            ckpt_paths = ckpt_paths[-max_checkpoints:]

        n = len(ckpt_paths)
        if n < 2:
            print("[round-robin] Need at least 2 checkpoints.")
            return [], None, None, None

        names = [os.path.basename(p) for p in ckpt_paths]
        print(f"[round-robin] Using {n} checkpoints:")
        for name in names:
            print(f"  - {name}")

        # Load params
        params_list = []
        for path in ckpt_paths:
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            params_list.append(
                {
                    "network_params": ckpt["params"],
                    "batch_stats": ckpt["batch_stats"],
                }
            )

        scores = np.zeros((n, n), dtype=float)  # avg score in [-1, 1]
        games_mat = np.zeros((n, n), dtype=int)

        # Optional W/D/L matrices if you care
        W_mat = np.zeros((n, n), dtype=int)
        D_mat = np.zeros((n, n), dtype=int)
        L_mat = np.zeros((n, n), dtype=int)

        total_pairs = n * (n - 1) // 2
        pair_idx = 0

        rng = self.rng

        for i in range(n):
            for j in range(i + 1, n):
                pair_idx += 1
                print(f"[round-robin] Pair {pair_idx}/{total_pairs}: {names[i]} (home) vs {names[j]} (away)")

                home_params = params_list[i]
                away_params = params_list[j]

                rng, match_rng = jax.random.split(rng)

                (total_score_home,
                total_games,
                home_wins,
                draws,
                away_wins,
                per_game_turns,
                per_game_jumps_p1,
                per_game_jumps_p2,
                per_game_jumps_total,
                per_game_removed,
                winners,
                home_is_P1) = play_match_batched(
                    home_params=home_params,
                    away_params=away_params,
                    rng=match_rng,
                    network=self.network,
                    env_config=self.env_config,
                    games_per_color=games_per_color,
                    max_moves=max_moves,
                    num_simulations=num_simulations,
                    temperature=temperature,
                )

                # --- Convert to numpy / python ---
                total_score_home = float(total_score_home)
                total_games = int(total_games)
                home_wins = int(home_wins)
                away_wins = int(away_wins)
                draws = int(draws)

                per_game_turns = np.array(per_game_turns, dtype=np.int32)
                per_game_jumps_p1 = np.array(per_game_jumps_p1, dtype=np.int32)
                per_game_jumps_p2 = np.array(per_game_jumps_p2, dtype=np.int32)
                per_game_jumps_total = np.array(per_game_jumps_total, dtype=np.int32)
                per_game_removed = np.array(per_game_removed, dtype=np.int32)
                winners = np.array(winners, dtype=np.int32)
                home_is_P1 = np.array(home_is_P1, dtype=bool)

                assert total_games == len(per_game_turns)

                # Map jumps to "home" and "away" for each game
                jumps_home = np.where(home_is_P1, per_game_jumps_p1, per_game_jumps_p2)
                jumps_away = per_game_jumps_p1 + per_game_jumps_p2 - jumps_home

                # Aggregate stats
                avg_score_home = total_score_home / total_games if total_games > 0 else 0.0
                avg_turns = float(per_game_turns.mean())

                total_jumps_home = jumps_home.sum()
                total_jumps_away = jumps_away.sum()
                avg_jumps_home = total_jumps_home / total_games
                avg_jumps_away = total_jumps_away / total_games

                jump_mask = per_game_jumps_total > 0
                if jump_mask.any():
                    total_removed = per_game_removed[jump_mask].sum()
                    total_jumps = per_game_jumps_total[jump_mask].sum()
                    avg_jump_len = total_removed / total_jumps
                else:
                    avg_jump_len = 0.0

                # Store results from HOME's POV
                scores[i, j] = avg_score_home
                scores[j, i] = -avg_score_home
                games_mat[i, j] = games_mat[j, i] = total_games

                W_mat[i, j] = home_wins
                L_mat[i, j] = away_wins
                D_mat[i, j] = draws
                W_mat[j, i] = away_wins
                L_mat[j, i] = home_wins
                D_mat[j, i] = draws

                print(
                    f"  [round-robin] {names[i]} (home) vs {names[j]} (away): "
                    f"home W-D-L = {home_wins}-{draws}-{away_wins} "
                    f"(score={total_score_home:+.1f} over {total_games} games, "
                    f"avg {avg_score_home:+.3f})"
                )
                print(
                    f"    turns/game={avg_turns:.1f}, "
                    f"jumps/game home={avg_jumps_home:.2f}, away={avg_jumps_away:.2f}, "
                    f"avg_jump_len={avg_jump_len:.2f}"
                )

                # --- Per-game detailed report ---
                for g in range(total_games):
                    if winners[g] == 0:
                        outcome = "draw"
                    else:
                        # Did home win this game?
                        home_won = (
                            (winners[g] == 1 and home_is_P1[g]) or
                            (winners[g] == 2 and not home_is_P1[g])
                        )
                        outcome = "home win" if home_won else "away win"

                    # How does P1/P2 map to home/away?
                    role_str = "P1=home,P2=away" if home_is_P1[g] else "P1=away,P2=home"

                    if per_game_jumps_total[g] > 0:
                        game_avg_jump_len = per_game_removed[g] / per_game_jumps_total[g]
                    else:
                        game_avg_jump_len = 0.0

                    print(
                        f"      game {g:2d}: {role_str}, "
                        f"turns={per_game_turns[g]}, "
                        f"jumps_home={int(jumps_home[g])}, jumps_away={int(jumps_away[g])}, "
                        f"avg_jump_len={game_avg_jump_len:.2f}, "
                        f"result={outcome}"
                    )

        self.rng = rng


        # --- Simple Elo-ish rating based on average scores ---
        ratings = np.zeros(n, dtype=float)
        K = 16.0

        def expected_score(r_i, r_j):
            return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))

        for i in range(n):
            for j in range(n):
                if i == j or games_mat[i, j] == 0:
                    continue
                # Convert avg score in [-1,1] to [0,1]
                s_01 = 0.5 * (scores[i, j] + 1.0)
                e_01 = expected_score(ratings[i], ratings[j])
                ratings[i] += K * (s_01 - e_01)

        return names, scores, games_mat, ratings, (W_mat, D_mat, L_mat)

def compute_elo_from_results(
    names: List[str],
    W: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    base_elo: float = 1500.0,
    K: float = 32.0,
    iters: int = 50,
) -> np.ndarray:
    """
    Simple multi-player Elo solver from a full W/D/L matrix.

    W[i,j] = wins of i vs j
    D[i,j] = draws
    L[i,j] = losses of i vs j

    Returns:
        ratings: np.ndarray of shape (N,)
    """
    n = len(names)
    ratings = np.full(n, base_elo, dtype=np.float64)

    for _ in range(iters):
        for i in range(n):
            for j in range(i + 1, n):
                # Aggregate results for this pair
                wins_ij  = float(W[i, j])
                draws_ij = float(D[i, j])
                loss_ij  = float(L[i, j])
                N_ij = wins_ij + draws_ij + loss_ij
                if N_ij == 0:
                    continue

                # Actual total score for i vs j (win=1, draw=0.5, loss=0)
                S_i = wins_ij + 0.5 * draws_ij

                # Expected score per game for i vs j
                diff = (ratings[j] - ratings[i]) / 400.0
                p_i = 1.0 / (1.0 + 10.0**diff)
                E_i = N_ij * p_i

                # Batch Elo update (equivalent to N_ij separate games)
                delta = K * (S_i - E_i) / max(1.0, N_ij)
                ratings[i] += delta
                ratings[j] -= delta

    return ratings


def evaluate_vs_random(trainer: AlphaZeroTrainer, num_games: int = 20) -> float:
    """Evaluate trained model against random play."""
    from mcts import MCTSConfig, select_action
    
    wins = 0
    
    for game_idx in range(num_games):
        state = reset(trainer.env_config)
        trainer.rng, game_rng = jax.random.split(trainer.rng)
        
        # Alternate who goes first
        trained_player = 1 if game_idx % 2 == 0 else 2
        
        move = 0
        while not state.terminated and move < 500:
            current_player = int(state.current_player)
            
            if current_player == trained_player:
                # Trained model's turn - use MCTS
                game_rng, action_rng = jax.random.split(game_rng)
                action, _, _ = select_action(
                    trainer.get_network_params(),
                    action_rng,
                    state,
                    trainer.network,
                    trainer.env_config,
                    MCTSConfig(num_simulations=50, max_num_considered_actions=16),
                    temperature=0.1,
                )
            else:
                # Random player's turn
                game_rng, action_rng = jax.random.split(game_rng)
                legal_mask = get_legal_actions(state, trainer.env_config)
                legal_actions = jnp.where(legal_mask == 1)[0]
                action = int(jax.random.choice(action_rng, legal_actions))
            
            state = step(state, jnp.array(action, dtype=jnp.int32), trainer.env_config)
            move += 1
        
        winner = int(state.winner)
        if winner == trained_player:
            wins += 1
    
    win_rate = wins / num_games
    print(f"Evaluation vs Random: {wins}/{num_games} wins ({win_rate*100:.1f}%)")
    return win_rate


def get_buffer_size(rows, cols, games_per_iter=256, max_moves=512, target_staleness_iters=8):
    examples_per_iter = games_per_iter * max_moves
    return examples_per_iter * target_staleness_iters


def make_train_config(
    rows: int,
    cols: int,
    checkpoint_dir: str,
    num_iterations: int = 256,
    checkpoint_every: int = 20,
    use_wandb: bool = False,
    wandb_run_name: Optional[str] = None,
    eval_vs_random_threshold: Optional[float] = 0.90,
    stop_when_beating_random: bool = True,
    num_simulations: int = 50,
    temp_threshold: int = 30,
    temp_final: float = 0.1,
    # Curriculum learning
    curriculum_enabled: bool = True,
    curriculum_initial_ratio: float = 0.5,
    curriculum_final_ratio: float = 0.0,
    curriculum_decay_iterations: int = 100,
    curriculum_jump_distribution: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> TrainConfig:
    """
    Convenience factory for TrainConfig so we don't duplicate hyperparams
    in every notebook/script.
    """

    buffer_size = get_buffer_size(rows, cols)


    return TrainConfig(
        # Environment
        rows=rows,
        cols=cols,

        # Network
        num_channels=64,
        num_res_blocks=8,

        # Self-play
        batch_size_games=128,
        games_per_iteration=256,
        max_turns_per_game=512,
        max_moves_per_game=512,
        temperature=1.0,
        temp_threshold=temp_threshold,
        temp_final=temp_final,
        num_simulations=num_simulations,

        # Training
        batch_size_train=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        train_steps_per_iteration=100,

        # Replay buffer
        buffer_size=buffer_size,
        min_buffer_size=1_000,

        # Iterations / checkpoints
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        log_every=1,

        # Eval vs random
        eval_enable=True,
        eval_vs_random_games=100,
        eval_vs_random_threshold=eval_vs_random_threshold,
        stop_when_beating_random=stop_when_beating_random,
        eval_max_prev_checkpoints=5,
        eval_games_per_color=5,
        eval_num_simulations=128,
        eval_max_moves=512,
        eval_temperature=0.0,

        # Curriculum learning
        curriculum_enabled=curriculum_enabled,
        curriculum_initial_ratio=curriculum_initial_ratio,
        curriculum_final_ratio=curriculum_final_ratio,
        curriculum_decay_iterations=curriculum_decay_iterations,
        curriculum_jump_distribution=curriculum_jump_distribution,

        # Wandb
        use_wandb=use_wandb,
        wandb_project="phutball-az",
        wandb_run_name=wandb_run_name,
        wandb_mode="online" if use_wandb else "disabled",
    )




# ============================================================================
# Quick test
# ============================================================================

def test_training_loop():
    """Quick test of training loop with tiny config."""
    config = TrainConfig(
        # Small board for testing
        rows=9,
        cols=9,
        # Tiny network
        num_channels=16,
        num_res_blocks=2,
        # Batched self-play
        batch_size_games=8,
        max_turns_per_game=30,   # 30 turns max
        max_moves_per_game=200,  # Memory cap
        games_per_iteration=16,  # 2 batches of 8
        temperature=1.0,
        temp_threshold=15,       # Drop temp after 15 moves
        temp_final=0.1,
        # Minimal training
        train_steps_per_iteration=5,
        batch_size_train=8,
        buffer_size=500,
        min_buffer_size=10,
        # Short run
        num_iterations=3,
        checkpoint_every=100,
    )
    
    trainer = AlphaZeroTrainer(config)
    
    # Run training
    trainer.train()
    
    print("\n✓ Training loop test passed!")
    print(f"  Total games: {trainer.total_games}")
    print(f"  Total examples: {trainer.total_examples}")
    print(f"  Final buffer size: {len(trainer.replay_buffer)}")


if __name__ == "__main__":
    print("Testing batched training loop...\n")
    test_training_loop()