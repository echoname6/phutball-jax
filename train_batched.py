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
    create_optimizer, make_train_step_fn, predict,
    # Chimera imports
    ChimeraNetwork, create_chimera_network, init_chimera_network,
    expand_chimera_network, make_chimera_train_step_fn, predict_chimera,
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

    # Directions toward goal (for final jump into endzone)
    if player == 1:
        goal_directions = [(1, 0), (1, -1), (1, 1)]  # down, down-left, down-right
        endzone_rows = [rows - 2, rows - 1]
    else:
        goal_directions = [(-1, 0), (-1, -1), (-1, 1)]  # up, up-left, up-right
        endzone_rows = [0, 1]

    # All directions for intermediate jumps (including horizontal)
    # Validation below will ensure paths don't overlap
    all_directions = [
        (1, 0), (-1, 0),   # vertical
        (0, 1), (0, -1),   # horizontal
        (1, 1), (1, -1),   # diagonal down
        (-1, 1), (-1, -1), # diagonal up
    ]

    # Try multiple random configurations if needed
    max_attempts = 10
    for attempt in range(max_attempts):
        # Split RNG for this attempt
        attempt_rng, rng = jax.random.split(rng)
        rng_keys = jax.random.split(attempt_rng, 3 * num_jumps + 3)
        rng_idx = 0

        # Reset state for this attempt
        positions = []
        jump_lengths = []
        jump_dirs = []
        used_men = set()
        generation_failed = False

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
            # First jump (jump_idx=0) is the final jump INTO endzone - must be goal-directed
            # Later jumps (jump_idx>0) are intermediate - can include horizontal
            if jump_idx == 0:
                directions = goal_directions
            else:
                directions = all_directions

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
                if dr != 0:
                    # Was diagonal - fallback to vertical, but avoid pure-backward
                    dc = 0
                    prev_col = curr_col
                    # If dr would be backward (away from goal), flip to forward
                    if (player == 1 and dr < 0) or (player == 2 and dr > 0):
                        dr = 1 if player == 1 else -1  # forward toward goal
                    prev_row = curr_row - (jump_len + 1) * dr
                else:
                    # Was horizontal (dr=0) - flip direction
                    dc = -dc
                    prev_col = curr_col - (jump_len + 1) * dc
                    if prev_col < 0 or prev_col >= cols:
                        # Still out of bounds, switch to goal-directed vertical
                        dr = 1 if player == 1 else -1
                        dc = 0
                        prev_row = curr_row - (jump_len + 1) * dr
                        prev_col = curr_col

            # For intermediate positions (not the ball), must be in playable area
            # For the ball (last iteration), also must be in playable area
            # Note: for horizontal jumps (dr=0), row doesn't change so check column instead
            if dr != 0 and (prev_row <= 1 or prev_row >= rows - 2):
                # Need to reduce jump length to fit (vertical/diagonal case)
                if player == 1:
                    # Moving down, prev must be above curr
                    available_space = curr_row - 2  # rows 0,1 are P2 endzone
                else:
                    # Moving up, prev must be below curr
                    available_space = (rows - 3) - curr_row  # rows-2, rows-1 are P1 endzone

                max_possible_len = available_space // abs(dr) - 1
                jump_len = max(1, min(jump_len, max(1, max_possible_len)))

                prev_row = curr_row - (jump_len + 1) * dr
                prev_col = curr_col - (jump_len + 1) * dc

                if prev_col < 0 or prev_col >= cols:
                    dc = 0
                    prev_col = curr_col
                    prev_row = curr_row - (jump_len + 1) * dr
            elif dr == 0 and (prev_col < 0 or prev_col >= cols):
                # Horizontal jump went out of column bounds - reduce length or flip direction
                if dc > 0:
                    available_space = cols - 1 - curr_col
                else:
                    available_space = curr_col
                if available_space >= 2:
                    jump_len = min(jump_len, available_space - 1)
                    prev_col = curr_col - (jump_len + 1) * dc
                else:
                    # Flip direction
                    dc = -dc
                    if dc > 0:
                        available_space = cols - 1 - curr_col
                    else:
                        available_space = curr_col
                    jump_len = min(jump_len, max(1, available_space - 1))
                    prev_col = curr_col - (jump_len + 1) * dc
                prev_row = curr_row  # stays same for horizontal

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

            # Calculate men positions for this jump and check for overlap
            men_positions = set()
            for i in range(1, jump_len + 1):
                mr = prev_row + i * dr
                mc = prev_col + i * dc
                if 0 <= mr < rows and 0 <= mc < cols:
                    men_positions.add((mr, mc))

            # Track landing positions that must stay empty for ball to land
            landing_positions_so_far = set(positions)

            # For the last jump, check if prev (ball position) conflicts with used_men
            ball_conflict = jump_idx == num_jumps - 1 and (prev_row, prev_col) in used_men

            # Check overlap with used_men AND with landing positions
            overlap = (men_positions & used_men) | (men_positions & landing_positions_so_far)
            if (overlap or ball_conflict) and jump_idx > 0:
                resolved = False
                # Try all directions with different jump lengths
                # Prioritize goal-directed vertical, then horizontal, then diagonal
                test_directions = [
                    (1 if player == 1 else -1, 0),   # goal-directed vertical
                    (0, 1), (0, -1),                  # horizontal
                    (1 if player == 1 else -1, 1),   # goal-directed diagonal right
                    (1 if player == 1 else -1, -1),  # goal-directed diagonal left
                    (-1 if player == 1 else 1, 0),   # backward vertical
                    (-1 if player == 1 else 1, 1),   # backward diagonal right
                    (-1 if player == 1 else 1, -1),  # backward diagonal left
                ]
                for test_dr, test_dc in test_directions:
                    if resolved:
                        break
                    for test_len in range(1, max_jump_len + 1):
                        if test_dr != 0:
                            test_prev_row = curr_row - (test_len + 1) * test_dr
                            test_prev_col = curr_col - (test_len + 1) * test_dc
                        else:
                            # Horizontal jump
                            test_prev_row = curr_row
                            test_prev_col = curr_col - (test_len + 1) * test_dc

                        # Check bounds
                        if test_prev_row <= 1 or test_prev_row >= rows - 2:
                            continue
                        if test_prev_col < 0 or test_prev_col >= cols:
                            continue

                        test_men = set()
                        for i in range(1, test_len + 1):
                            mr = test_prev_row + i * test_dr
                            mc = test_prev_col + i * test_dc
                            if 0 <= mr < rows and 0 <= mc < cols:
                                test_men.add((mr, mc))

                        # For the last jump, the prev position becomes the ball
                        # So also check that prev is not in used_men
                        if jump_idx == num_jumps - 1:
                            prev_candidate = (test_prev_row, test_prev_col)
                            if prev_candidate in used_men:
                                continue  # This would put ball on a men position

                        if test_men and not (test_men & used_men) and not (test_men & landing_positions_so_far):
                            prev_row, prev_col = test_prev_row, test_prev_col
                            dr, dc = test_dr, test_dc
                            jump_len = test_len
                            men_positions = test_men
                            resolved = True
                            break

                # If we couldn't resolve, mark generation as failed and try a new random config
                if not resolved:
                    generation_failed = True
                    break  # Exit jump loop to try a new random configuration

            # Add men positions to used set
            used_men.update(men_positions)

            positions.append((prev_row, prev_col))
            jump_lengths.append(jump_len)
            jump_dirs.append((dr, dc))

            # After the last jump is planned, check if ball position conflicts with any men
            # (This is the last iteration, so positions[-1] will be the ball position)
            if jump_idx == num_jumps - 1:
                ball_candidate = (prev_row, prev_col)
                if ball_candidate in used_men:
                    # Ball position conflicts with a man needed by an earlier jump
                    # Try shifting the ball position by changing the last jump's direction/length
                    resolved_ball = False
                    for alt_dr, alt_dc in all_directions:
                        if resolved_ball:
                            break
                        for alt_len in range(1, max_jump_len + 1):
                            if alt_dr != 0:
                                alt_prev_row = curr_row - (alt_len + 1) * alt_dr
                                alt_prev_col = curr_col - (alt_len + 1) * alt_dc
                            else:
                                alt_prev_row = curr_row
                                alt_prev_col = curr_col - (alt_len + 1) * alt_dc

                            # Check bounds
                            if alt_prev_row <= 1 or alt_prev_row >= rows - 2:
                                continue
                            if alt_prev_col < 0 or alt_prev_col >= cols:
                                continue

                            # Compute men for this alternative
                            alt_men = set()
                            for i in range(1, alt_len + 1):
                                mr = alt_prev_row + i * alt_dr
                                mc = alt_prev_col + i * alt_dc
                                if 0 <= mr < rows and 0 <= mc < cols:
                                    alt_men.add((mr, mc))

                            alt_ball = (alt_prev_row, alt_prev_col)
                            # Check: ball not in used_men, alt_men don't overlap used_men or landing positions
                            other_used_men = used_men - men_positions
                            other_landings = landing_positions_so_far - {(prev_row, prev_col)}
                            if (alt_ball not in other_used_men and alt_men and
                                not (alt_men & other_used_men) and not (alt_men & other_landings)):
                                # Found a valid alternative
                                positions[-1] = alt_ball
                                jump_lengths[-1] = alt_len
                                jump_dirs[-1] = (alt_dr, alt_dc)
                                # Update used_men: remove old men, add new
                                used_men -= men_positions
                                used_men.update(alt_men)
                                resolved_ball = True
                                break

        # Check if this attempt succeeded (generated all required jumps)
        if not generation_failed and len(jump_dirs) == num_jumps:
            # Post-validation: check that no men positions overlap with any landing positions
            # This catches cases where a later jump (in backwards order) places men at
            # a landing position that wasn't known when the earlier jump was planned
            all_landing_positions = set(positions)  # positions includes all landings
            all_men_positions = used_men.copy()
            conflict = all_men_positions & all_landing_positions
            if conflict:
                # Men were placed at landing positions - retry with different config
                generation_failed = True
            else:
                break  # Success! Exit attempt loop
        # Otherwise, try another random configuration
        # (generation_failed is already True, or we didn't get enough jumps)

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
    # Note: len(jump_dirs) may be less than num_jumps if we abandoned a jump
    actual_num_jumps = len(jump_dirs)

    # Landing positions that must stay empty (all positions except the ball)
    landing_positions = set(positions[1:])

    for jump_idx in range(actual_num_jumps):
        start_row, start_col = positions[jump_idx]
        dr, dc = jump_dirs[jump_idx]
        jump_len = jump_lengths[jump_idx]

        for i in range(1, jump_len + 1):
            mr = start_row + i * dr
            mc = start_col + i * dc
            # Men can be placed in first endzone rows (1 and rows-2) but not deep rows (0 and rows-1)
            if 1 <= mr <= rows - 2 and 0 <= mc < cols:
                # Safety check: never overwrite the ball or a landing position with a man
                if (mr, mc) != (ball_row, ball_col) and (mr, mc) not in landing_positions:
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


def _stack_states(state_list: List[PhutballState]) -> PhutballState:
    """Stack a list of PhutballState objects into a batched PhutballState."""
    return PhutballState(
        board=jnp.stack([s.board for s in state_list], axis=0),
        ball_pos=jnp.stack([s.ball_pos for s in state_list], axis=0),
        current_player=jnp.stack([s.current_player for s in state_list], axis=0),
        is_jumping=jnp.stack([s.is_jumping for s in state_list], axis=0),
        terminated=jnp.stack([s.terminated for s in state_list], axis=0),
        winner=jnp.stack([s.winner for s in state_list], axis=0),
        num_turns=jnp.stack([s.num_turns for s in state_list], axis=0),
        jump_sequence=jnp.stack([s.jump_sequence for s in state_list], axis=0),
        jump_sequence_length=jnp.stack([s.jump_sequence_length for s in state_list], axis=0),
    )


def generate_curriculum_batch(
    rng: jax.Array,
    env_config: EnvConfig,
    batch_size: int,
    jump_distribution: List[float] = None,
    return_stats: bool = False,
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
        return_stats: If True, return statistics dict as 4th element

    Returns:
        states: (batch_size, 6, rows, cols) - network input format
        policy_targets: (batch_size, action_space_size) - one-hot on first action
        value_targets: (batch_size,) - all +1 (current player wins)
        stats: (optional) dict with counts by jump type if return_stats=True
    """
    if jump_distribution is None:
        jump_distribution = [0.4, 0.3, 0.2, 0.1]  # Default: more 1-jump, fewer 4-jump

    # Normalize distribution
    total = sum(jump_distribution)
    jump_distribution = [p / total for p in jump_distribution]

    rows, cols = env_config.rows, env_config.cols
    action_space_size = 2 * rows * cols + 1

    # Collect states and actions
    state_list = []
    action_list = []

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

            state_list.append(state)
            action_list.append(first_action)

    # Batch convert states to network inputs using vmap
    batched_states = _stack_states(state_list)
    batched_to_input = jax.vmap(lambda s: state_to_network_input(s, env_config))
    states = np.array(batched_to_input(batched_states))

    # Create policy targets
    policies = np.zeros((batch_size, action_space_size), dtype=np.float32)
    action_indices = np.array(action_list, dtype=np.int32)
    policies[np.arange(batch_size), action_indices] = 1.0

    values = np.ones(batch_size, dtype=np.float32)  # Current player wins

    if return_stats:
        stats = {
            'curriculum_1jump': counts[0],
            'curriculum_2jump': counts[1],
            'curriculum_3jump': counts[2],
            'curriculum_4jump': counts[3],
            'curriculum_total': batch_size,
        }
        return states, policies, values, stats

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

    # LR decay on loss plateau
    lr_decay_enabled: bool = False
    lr_decay_factor: float = 0.5       # Multiply LR by this when loss stalls
    lr_decay_patience: int = 10        # Iterations without improvement before decay
    lr_min: float = 1e-5               # Don't decay below this

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

    # Random opponent mixing: adds diversity to break passive self-play equilibrium
    random_opponent_enabled: bool = False
    random_opponent_initial_ratio: float = 0.33   # Initial fraction of games vs random
    random_opponent_final_ratio: float = 0.003    # Final fraction (decays to this)
    random_opponent_decay_iterations: int = 100   # Iterations to decay from initial to final

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
        
        # Replay buffer (with flip + rotation augmentation)
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size,
            cols=config.cols,
            augment_flip=True,
        )

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

        # LR decay on plateau tracking
        self.current_lr = config.learning_rate
        self.best_loss = float('inf')
        self.iters_without_improvement = 0

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

    def _maybe_decay_lr(self, total_loss: float):
        """Decay learning rate if loss has stalled."""
        if not self.config.lr_decay_enabled:
            return

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.iters_without_improvement = 0
        else:
            self.iters_without_improvement += 1

        if self.iters_without_improvement >= self.config.lr_decay_patience:
            new_lr = max(self.current_lr * self.config.lr_decay_factor, self.config.lr_min)
            if new_lr < self.current_lr:
                print(f"  [LR decay] Loss stalled for {self.config.lr_decay_patience} iters, "
                      f"reducing LR: {self.current_lr:.2e} -> {new_lr:.2e}")
                self.current_lr = new_lr
                # Recreate optimizer with new LR
                self.optimizer = create_optimizer(
                    learning_rate=self.current_lr,
                    weight_decay=self.config.weight_decay,
                )
                self.opt_state = self.optimizer.init(self.params)
                self.train_step_fn = make_train_step_fn(self.network, self.optimizer)
                self.iters_without_improvement = 0
                self.best_loss = total_loss  # Reset best loss after decay

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
        vs_random_games = 0
        self_play_games = 0

        # Calculate current random opponent ratio (decays over iterations)
        if self.config.random_opponent_enabled:
            decay_progress = min(1.0, self.iteration / max(1, self.config.random_opponent_decay_iterations))
            random_opp_ratio = (
                self.config.random_opponent_initial_ratio
                + (self.config.random_opponent_final_ratio - self.config.random_opponent_initial_ratio)
                * decay_progress
            )
        else:
            random_opp_ratio = 0.0

        # Run multiple batches to get desired number of games
        num_batches = self.config.games_per_iteration // self.config.batch_size_games

        for batch_idx in range(num_batches):
            self.rng, game_rng = jax.random.split(self.rng)

            # Play games with mixed self-play and random opponents within each batch
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
                random_opponent_ratio=random_opp_ratio,
            )

            # Track game counts (same logic as play_games_batched)
            num_vs_random_this_batch = int(self.config.batch_size_games * random_opp_ratio)
            if random_opp_ratio > 0 and num_vs_random_this_batch == 0:
                num_vs_random_this_batch = 1
            vs_random_games += num_vs_random_this_batch
            self_play_games += self.config.batch_size_games - num_vs_random_this_batch

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
            "self_play_games": self_play_games,
            "vs_random_games": vs_random_games,
            "random_opp_ratio": random_opp_ratio,
            "avg_moves_per_game": avg_moves_per_game,
            "avg_placements_per_jump": avg_placements_per_jump,
            "avg_jump_sequence_len": avg_jump_seq_len,
            "avg_jump_length": avg_jump_length,
            "adjacent_conversion_rate": adj_conv_rate,
        }

        games_per_sec = games_total / elapsed if elapsed > 0 else 0
        vs_rand_str = f", vs_rand={vs_random_games}" if vs_random_games > 0 else ""
        print(
            f"  Self-play: {num_examples} examples from {games_total} games "
            f"({games_per_sec:.1f} games/sec, {elapsed:.1f}s){vs_rand_str} | "
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
            'value_pred_mean': 0.0,
            'value_pred_std': 0.0,
        }

        # Curriculum statistics tracking
        curriculum_stats_sum = {
            'curriculum_1jump': 0,
            'curriculum_2jump': 0,
            'curriculum_3jump': 0,
            'curriculum_4jump': 0,
            'curriculum_total': 0,
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
                curr_states, curr_policies, curr_values, curr_stats = generate_curriculum_batch(
                    curriculum_rng, self.env_config, curriculum_size,
                    jump_distribution=list(self.config.curriculum_jump_distribution),
                    return_stats=True
                )
                # Accumulate curriculum stats
                for k, v in curr_stats.items():
                    curriculum_stats_sum[k] += v
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

        # Add curriculum ratio and stats to metrics for logging
        avg_metrics['curriculum_ratio'] = curriculum_ratio
        avg_metrics.update(curriculum_stats_sum)

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
                # Check for LR decay on loss plateau
                total_loss = metrics.get('total_loss', metrics.get('policy_loss', 0) + metrics.get('value_loss', 0))
                self._maybe_decay_lr(total_loss)

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
                    "train/policy_entropy": metrics["policy_entropy"],
                    "train/mcts_entropy": metrics["mcts_entropy"],
                    "train/kl_divergence": metrics["kl_divergence"],
                    "train/value_pred_mean": metrics["value_pred_mean"],
                    "train/value_pred_std": metrics["value_pred_std"],
                    "train/curriculum_ratio": metrics.get("curriculum_ratio", 0.0),
                    "train/curriculum_1jump": metrics.get("curriculum_1jump", 0),
                    "train/curriculum_2jump": metrics.get("curriculum_2jump", 0),
                    "train/curriculum_3jump": metrics.get("curriculum_3jump", 0),
                    "train/curriculum_4jump": metrics.get("curriculum_4jump", 0),
                    "train/curriculum_total": metrics.get("curriculum_total", 0),
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
    wandb_project: str = "phutball-az",
    eval_vs_random_threshold: Optional[float] = 0.90,
    eval_vs_random_games: int = 100,
    stop_when_beating_random: bool = True,
    num_simulations: int = 50,
    temp_threshold: int = 30,
    temp_final: float = 0.1,
    # Network
    num_channels: int = 64,
    num_res_blocks: int = 8,
    # Batch sizes
    batch_size_games: int = 128,
    batch_size_train: int = 256,
    # Training params
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    train_steps_per_iteration: int = 100,
    buffer_size: Optional[int] = None,
    min_buffer_size: int = 1000,
    max_moves_per_game: Optional[int] = None,
    # LR decay on plateau
    lr_decay_enabled: bool = False,
    lr_decay_factor: float = 0.5,
    lr_decay_patience: int = 10,
    lr_min: float = 1e-5,
    # Curriculum learning
    curriculum_enabled: bool = True,
    curriculum_initial_ratio: float = 0.5,
    curriculum_final_ratio: float = 0.0,
    curriculum_decay_iterations: int = 100,
    curriculum_jump_distribution: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
    # Random opponent mixing
    random_opponent_enabled: bool = False,
    random_opponent_initial_ratio: float = 0.33,
    random_opponent_final_ratio: float = 0.003,
    random_opponent_decay_iterations: int = 100,
) -> TrainConfig:
    """
    Convenience factory for TrainConfig so we don't duplicate hyperparams
    in every notebook/script.
    """

    if buffer_size is None:
        buffer_size = get_buffer_size(rows, cols)

    if max_moves_per_game is None:
        max_moves_per_game = rows * cols * 2

    return TrainConfig(
        # Environment
        rows=rows,
        cols=cols,

        # Network
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,

        # Self-play
        batch_size_games=batch_size_games,
        games_per_iteration=batch_size_games * 2,
        max_turns_per_game=512,
        max_moves_per_game=max_moves_per_game,
        temperature=1.0,
        temp_threshold=temp_threshold,
        temp_final=temp_final,
        num_simulations=num_simulations,

        # Training
        batch_size_train=batch_size_train,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_steps_per_iteration=train_steps_per_iteration,

        # LR decay on plateau
        lr_decay_enabled=lr_decay_enabled,
        lr_decay_factor=lr_decay_factor,
        lr_decay_patience=lr_decay_patience,
        lr_min=lr_min,

        # Replay buffer
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,

        # Iterations / checkpoints
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        log_every=1,

        # Eval vs random
        eval_enable=True,
        eval_vs_random_games=eval_vs_random_games,
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

        # Random opponent mixing
        random_opponent_enabled=random_opponent_enabled,
        random_opponent_initial_ratio=random_opponent_initial_ratio,
        random_opponent_final_ratio=random_opponent_final_ratio,
        random_opponent_decay_iterations=random_opponent_decay_iterations,

        # Wandb
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode="online" if use_wandb else "disabled",
    )


# ============================================================================
# Chimera Trainer: Shared backbone, separate policy heads per board size
# ============================================================================

@dataclass
class ChimeraConfig:
    """Configuration for ChimeraTrainer (multi-board training)."""
    # Board sizes: tuple of (rows, cols)
    board_sizes: Tuple[Tuple[int, int], ...] = ((11, 9), (15, 11), (21, 15))

    # Network architecture (shared backbone)
    num_channels: int = 64
    num_res_blocks: int = 6

    # Self-play
    batch_size_games: int = 64
    max_turns_per_game: int = 2048
    max_moves_per_game: int = 4096
    temperature: float = 1.0
    temp_threshold: int = 30
    temp_final: float = 0.1
    num_simulations: int = 50

    # Training
    batch_size_train: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100

    # LR decay on loss plateau
    lr_decay_enabled: bool = False
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 10
    lr_min: float = 1e-5

    # How to mix board sizes during training
    # 'uniform': equal samples from each size
    # 'weighted': more samples from larger boards (proportional to board area)
    # 'round_robin': cycle through sizes each step
    board_mix_strategy: str = 'weighted'

    # Replay buffer (per board size)
    buffer_size: int = 200000
    min_buffer_size: int = 500

    # Iterations
    num_iterations: int = 500
    games_per_iteration: int = 256

    # Checkpointing
    checkpoint_dir: str = "checkpoints_chimera"
    checkpoint_every: int = 10
    log_every: int = 1

    # Curriculum learning (applied per board size)
    curriculum_enabled: bool = True
    curriculum_initial_ratio: float = 0.5
    curriculum_final_ratio: float = 0.0
    curriculum_decay_iterations: int = 100
    curriculum_jump_distribution: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)

    use_wandb: bool = False
    wandb_project: str = "phutball-az-chimera"
    wandb_run_name: Optional[str] = None

    # MCTS simulation curriculum: double sims when avg win rate vs random hits threshold
    sim_curriculum_enabled: bool = False
    sim_curriculum_initial: int = 16       # Starting sims
    sim_curriculum_target: int = 32        # Target sims after threshold
    sim_curriculum_threshold: float = 0.90  # Avg win rate to trigger doubling
    sim_curriculum_eval_every: int = 10    # Evaluate vs random every N iterations
    sim_curriculum_eval_games: int = 50    # Games per board size for evaluation

    # Eval vs random (independent of sim curriculum)
    eval_vs_random_every: int = 0          # 0 = disabled, 1 = every iter
    eval_vs_random_games: int = 50         # Games per board size

    # League (opponent sampling from past checkpoints)
    league_enabled: bool = False
    league_pool_size: int = 10             # Max past checkpoints to keep
    league_opponent_ratio: float = 0.5     # Fraction of games vs past opponent
    league_save_every: int = 5             # Save to pool every N iterations


class ChimeraTrainer:
    """
    Multi-board trainer with shared backbone + value head, separate policy heads.

    Key features:
    - Single backbone learns features useful across all board sizes
    - Value head uses global pooling (size-agnostic)
    - Separate policy head per board size
    - Can add new board sizes post-hoc via expand_chimera_network()
    """

    def __init__(self, config: ChimeraConfig):
        self.config = config

        # Create EnvConfig for each board size
        self.env_configs = {
            f"{r}x{c}": EnvConfig(rows=r, cols=c, max_turns=config.max_turns_per_game)
            for r, c in config.board_sizes
        }

        # Create ChimeraNetwork
        self.network = create_chimera_network(
            board_sizes=config.board_sizes,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        )

        # Optimizer
        self.optimizer = create_optimizer(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Separate replay buffer per board size (with flip + rotation augmentation)
        self.replay_buffers = {
            f"{r}x{c}": ReplayBuffer(max_size=config.buffer_size, cols=c, augment_flip=True)
            for r, c in config.board_sizes
        }

        # Initialize
        self.rng = jax.random.PRNGKey(42)
        self._init_network()

        # Create JIT-compiled train steps for each board size
        self.train_step_fns = {
            board_key: make_chimera_train_step_fn(self.network, self.optimizer, board_key)
            for board_key in self.env_configs.keys()
        }

        # Metrics
        self.iteration = 0
        self.total_games = {bk: 0 for bk in self.env_configs}
        self.total_examples = {bk: 0 for bk in self.env_configs}
        self.metrics_history = []

        # LR decay on plateau tracking
        self.current_lr = config.learning_rate
        self.best_loss = float('inf')
        self.iters_without_improvement = 0

        # Sim curriculum tracking
        if config.sim_curriculum_enabled:
            self.current_sims = config.sim_curriculum_initial
            self.sims_doubled = False
        else:
            self.current_sims = config.num_simulations
            self.sims_doubled = False

        # League (past opponent pool)
        self.league_pool = []  # List of (iteration, params) tuples

        # Wandb
        self.wandb_run = None
        if self.config.use_wandb:
            if wandb is None:
                print("WARNING: wandb not installed, disabling")
                self.config.use_wandb = False
            else:
                run_name = config.wandb_run_name or f"chimera_{int(time.time())}"
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=run_name,
                    config=asdict(config),
                )

    def _init_network(self):
        """Initialize network parameters."""
        self.rng, init_rng = jax.random.split(self.rng)
        variables = init_chimera_network(init_rng, self.network, num_input_channels=6)

        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.opt_state = self.optimizer.init(self.params)

    def get_network_params(self) -> dict:
        """Get params dict for self-play."""
        return {
            'network_params': self.params,
            'batch_stats': self.batch_stats,
        }

    def _maybe_decay_lr(self, total_loss: float):
        """Decay learning rate if loss has stalled."""
        if not self.config.lr_decay_enabled:
            return

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.iters_without_improvement = 0
        else:
            self.iters_without_improvement += 1

        if self.iters_without_improvement >= self.config.lr_decay_patience:
            new_lr = max(self.current_lr * self.config.lr_decay_factor, self.config.lr_min)
            if new_lr < self.current_lr:
                print(f"  [LR decay] Loss stalled for {self.config.lr_decay_patience} iters, "
                      f"reducing LR: {self.current_lr:.2e} -> {new_lr:.2e}")
                self.current_lr = new_lr
                # Recreate optimizer with new LR
                self.optimizer = create_optimizer(
                    learning_rate=self.current_lr,
                    weight_decay=self.config.weight_decay,
                )
                self.opt_state = self.optimizer.init(self.params)
                # Recreate train step functions
                self.train_step_fns = {
                    board_key: make_chimera_train_step_fn(self.network, self.optimizer, board_key)
                    for board_key in self.env_configs.keys()
                }
                self.iters_without_improvement = 0
                self.best_loss = total_loss

    def run_self_play_for_board(self, board_key: str) -> dict:
        """Run self-play for a specific board size. Returns stats dict."""
        env_config = self.env_configs[board_key]
        rows, cols = env_config.rows, env_config.cols

        class ChimeraWrapper:
            def __init__(wrapper_self, chimera, board_key):
                wrapper_self.chimera = chimera
                wrapper_self.board_key = board_key
                wrapper_self.rows = rows
                wrapper_self.cols = cols

            def apply(wrapper_self, variables, x, train=True, mutable=None):
                if mutable:
                    return wrapper_self.chimera.apply(
                        variables, x, wrapper_self.board_key, train=train, mutable=mutable
                    )
                return wrapper_self.chimera.apply(
                    variables, x, wrapper_self.board_key, train=train
                )

        wrapper = ChimeraWrapper(self.network, board_key)

        total_states = []
        total_policies = []
        total_values = []

        # Stats tracking
        stats_totals = {
            "num_games": 0, "total_moves": 0, "total_placements": 0,
            "total_jumps": 0, "num_jump_sequences": 0, "sum_jump_sequence_lengths": 0,
            "sum_jump_removed_tiles": 0, "adjacency_opportunities": 0, "adjacency_conversions": 0,
        }
        p1_wins, p2_wins, draws = 0, 0, 0

        num_batches = self.config.games_per_iteration // self.config.batch_size_games
        start_time = time.time()

        # Get league opponent if enabled
        league_opponent = None
        league_ratio = 0.0
        if self.config.league_enabled and len(self.league_pool) > 0:
            league_opponent = self.get_league_opponent_params()
            league_ratio = self.config.league_opponent_ratio

        for _ in range(num_batches):
            self.rng, game_rng = jax.random.split(self.rng)

            trajectory = play_games_batched(
                params=self.get_network_params(),
                rng=game_rng,
                network=wrapper,
                env_config=env_config,
                batch_size=self.config.batch_size_games,
                max_turns=self.config.max_turns_per_game,
                max_moves=self.config.max_moves_per_game,
                temperature=self.config.temperature,
                temp_threshold=self.config.temp_threshold,
                temp_final=self.config.temp_final,
                num_simulations=self.current_sims,
                opponent_params=league_opponent,
                opponent_ratio=league_ratio,
            )

            # Compute stats
            batch_stats = compute_phutball_stats(trajectory, env_config)
            for k in stats_totals:
                stats_totals[k] += batch_stats[k]

            winners_np = np.array(trajectory.winners)
            p1_wins += int(np.sum(winners_np == 1))
            p2_wins += int(np.sum(winners_np == 2))
            draws += int(np.sum(winners_np == 0))

            states, policies, values = trajectory_to_training_examples(trajectory)
            total_states.append(states)
            total_policies.append(policies)
            total_values.append(values)

        elapsed = time.time() - start_time

        if total_states:
            all_states = np.concatenate(total_states, axis=0)
            all_policies = np.concatenate(total_policies, axis=0)
            all_values = np.concatenate(total_values, axis=0)
            self.replay_buffers[board_key].add(all_states, all_policies, all_values)
            num_examples = len(all_states)
        else:
            num_examples = 0

        games_total = num_batches * self.config.batch_size_games
        self.total_games[board_key] += games_total
        self.total_examples[board_key] += num_examples

        # Compute averages
        ng = stats_totals["num_games"] if stats_totals["num_games"] > 0 else 1
        total_jumps = stats_totals["total_jumps"]
        num_seq = stats_totals["num_jump_sequences"]
        avg_moves = stats_totals["total_moves"] / ng
        avg_jump_seq = stats_totals["sum_jump_sequence_lengths"] / num_seq if num_seq > 0 else 0
        avg_jump_len = stats_totals["sum_jump_removed_tiles"] / total_jumps if total_jumps > 0 else 0
        adj_ops = stats_totals["adjacency_opportunities"]
        adj_conv = stats_totals["adjacency_conversions"] / adj_ops if adj_ops > 0 else 0

        return {
            "examples": num_examples,
            "games": games_total,
            "elapsed": elapsed,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "avg_moves": avg_moves,
            "avg_jump_seq": avg_jump_seq,
            "avg_jump_len": avg_jump_len,
            "adj_conv": adj_conv,
            "buffer_size": len(self.replay_buffers[board_key]),
        }

    def run_self_play(self) -> dict:
        """Run self-play for all board sizes. Returns stats per board."""
        start_time = time.time()
        stats_per_board = {}

        for board_key in self.env_configs:
            stats = self.run_self_play_for_board(board_key)
            stats_per_board[board_key] = stats

        elapsed = time.time() - start_time
        total_examples = sum(s["examples"] for s in stats_per_board.values())
        total_games = sum(s["games"] for s in stats_per_board.values())
        games_per_sec = total_games / elapsed if elapsed > 0 else 0

        print(f"  Self-play: {total_examples} examples from {total_games} games ({games_per_sec:.2f} games/sec, {elapsed:.1f}s)")
        for bk, s in stats_per_board.items():
            print(f"    {bk}: {s['examples']} ex | W1/W2/D={s['p1_wins']}/{s['p2_wins']}/{s['draws']} | "
                  f"moves={s['avg_moves']:.1f}, jump_seq={s['avg_jump_seq']:.2f}, "
                  f"jump_len={s['avg_jump_len']:.2f}, adj={s['adj_conv']*100:.1f}% | buf={s['buffer_size']}")

        return stats_per_board

    def get_curriculum_ratio(self) -> float:
        """Calculate current curriculum ratio."""
        if not self.config.curriculum_enabled:
            return 0.0
        if self.iteration >= self.config.curriculum_decay_iterations:
            return self.config.curriculum_final_ratio
        progress = self.iteration / self.config.curriculum_decay_iterations
        return self.config.curriculum_initial_ratio + progress * (
            self.config.curriculum_final_ratio - self.config.curriculum_initial_ratio
        )

    def maybe_save_to_league(self):
        """Save current params to league pool if enabled."""
        if not self.config.league_enabled:
            return
        if (self.iteration + 1) % self.config.league_save_every != 0:
            return

        # Deep copy params to avoid issues with later updates
        import copy
        params_copy = {
            'network_params': copy.deepcopy(self.params),
            'batch_stats': copy.deepcopy(self.batch_stats),
        }
        self.league_pool.append((self.iteration, params_copy))

        # Trim pool if over capacity
        if len(self.league_pool) > self.config.league_pool_size:
            self.league_pool = self.league_pool[-self.config.league_pool_size:]

        print(f"  [League] Saved checkpoint to pool (size={len(self.league_pool)})")

    def get_league_opponent_params(self) -> dict:
        """Get random params from league pool, or None if pool is empty."""
        if not self.league_pool:
            return None
        idx = np.random.randint(len(self.league_pool))
        iter_num, params = self.league_pool[idx]
        return params

    def evaluate_vs_random_for_board(self, board_key: str) -> float:
        """Evaluate current model vs random for a specific board size. Returns win rate."""
        env_config = self.env_configs[board_key]
        rows, cols = env_config.rows, env_config.cols

        # Create wrapper for this board size (same as in run_self_play_for_board)
        class ChimeraWrapper:
            def __init__(wrapper_self, chimera, board_key):
                wrapper_self.chimera = chimera
                wrapper_self.board_key = board_key
                wrapper_self.rows = rows
                wrapper_self.cols = cols

            def apply(wrapper_self, variables, x, train=True, mutable=None):
                if mutable:
                    return wrapper_self.chimera.apply(
                        variables, x, wrapper_self.board_key, train=train, mutable=mutable
                    )
                return wrapper_self.chimera.apply(
                    variables, x, wrapper_self.board_key, train=train
                )

        wrapper = ChimeraWrapper(self.network, board_key)

        self.rng, eval_rng = jax.random.split(self.rng)
        (
            p1_wins, p1_draws, p1_losses,
            p2_wins, p2_draws, p2_losses,
            turns,
        ) = play_vs_random_batched(
            checkpoint_params=self.get_network_params(),
            rng=eval_rng,
            network=wrapper,
            env_config=env_config,
            num_games=self.config.sim_curriculum_eval_games,
            max_moves=rows * cols * 2,
            num_simulations=self.current_sims,
            temperature=0.1,
        )

        total = int(p1_wins + p1_draws + p1_losses + p2_wins + p2_draws + p2_losses)
        total_wins = int(p1_wins + p2_wins)
        win_rate = total_wins / total if total > 0 else 0.0

        print(f"    {board_key}: {total_wins}/{total} wins ({win_rate:.1%})")
        return win_rate

    def maybe_double_sims(self):
        """Check if avg win rate across board sizes triggers sim doubling."""
        if not self.config.sim_curriculum_enabled or self.sims_doubled:
            return

        if self.iteration % self.config.sim_curriculum_eval_every != 0:
            return

        print(f"  [Sim curriculum] Evaluating vs random (current sims: {self.current_sims})...")

        win_rates = []
        for board_key in self.env_configs:
            win_rate = self.evaluate_vs_random_for_board(board_key)
            win_rates.append(win_rate)

        avg_win_rate = sum(win_rates) / len(win_rates)
        print(f"  [Sim curriculum] Avg win rate: {avg_win_rate:.1%}")

        if avg_win_rate >= self.config.sim_curriculum_threshold:
            old_sims = self.current_sims
            self.current_sims = self.config.sim_curriculum_target
            self.sims_doubled = True
            print(f"  [Sim curriculum] THRESHOLD REACHED! Doubling sims: {old_sims} -> {self.current_sims}")

            if self.config.use_wandb and self.wandb_run:
                wandb.log({
                    "sim_curriculum/sims_doubled": 1,
                    "sim_curriculum/new_sims": self.current_sims,
                    "sim_curriculum/trigger_win_rate": avg_win_rate,
                })

    def run_eval_vs_random(self) -> dict:
        """Evaluate current model vs random for all board sizes. Returns win rates dict."""
        if self.config.eval_vs_random_every <= 0:
            return {}
        if (self.iteration + 1) % self.config.eval_vs_random_every != 0:
            return {}

        print(f"  [Eval] vs random ({self.config.eval_vs_random_games} games/board)...")
        win_rates = {}
        for board_key in self.env_configs:
            env_config = self.env_configs[board_key]
            rows, cols = env_config.rows, env_config.cols

            class ChimeraWrapper:
                def __init__(wrapper_self, chimera, board_key):
                    wrapper_self.chimera = chimera
                    wrapper_self.board_key = board_key
                    wrapper_self.rows = rows
                    wrapper_self.cols = cols

                def apply(wrapper_self, variables, x, train=True, mutable=None):
                    if mutable:
                        return wrapper_self.chimera.apply(
                            variables, x, wrapper_self.board_key, train=train, mutable=mutable
                        )
                    return wrapper_self.chimera.apply(
                        variables, x, wrapper_self.board_key, train=train
                    )

            wrapper = ChimeraWrapper(self.network, board_key)
            self.rng, eval_rng = jax.random.split(self.rng)

            (p1_wins, p1_draws, p1_losses,
             p2_wins, p2_draws, p2_losses, turns) = play_vs_random_batched(
                checkpoint_params=self.get_network_params(),
                rng=eval_rng,
                network=wrapper,
                env_config=env_config,
                num_games=self.config.eval_vs_random_games,
                max_moves=rows * cols * 2,
                num_simulations=self.current_sims,
                temperature=0.1,
            )

            total = int(p1_wins + p1_draws + p1_losses + p2_wins + p2_draws + p2_losses)
            total_wins = int(p1_wins + p2_wins)
            win_rate = total_wins / total if total > 0 else 0.0
            win_rates[board_key] = win_rate
            print(f"    {board_key}: {total_wins}/{total} wins ({win_rate:.1%})")

        avg_win_rate = sum(win_rates.values()) / len(win_rates) if win_rates else 0.0
        print(f"  [Eval] Avg win rate: {avg_win_rate:.1%}")
        win_rates['avg'] = avg_win_rate
        return win_rates

    def run_training(self) -> dict:
        """Run training steps across all board sizes."""
        # Check all buffers have minimum size
        for board_key, buffer in self.replay_buffers.items():
            if len(buffer) < self.config.min_buffer_size:
                print(f"  Skipping training: {board_key} buffer too small "
                      f"({len(buffer)}/{self.config.min_buffer_size})")
                return {}

        start_time = time.time()
        metrics_sum = {
            'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0,
            'policy_entropy': 0.0, 'mcts_entropy': 0.0, 'kl_divergence': 0.0,
        }

        # Curriculum statistics tracking
        curriculum_stats_sum = {
            'curriculum_1jump': 0,
            'curriculum_2jump': 0,
            'curriculum_3jump': 0,
            'curriculum_4jump': 0,
            'curriculum_total': 0,
        }

        curriculum_ratio = self.get_curriculum_ratio()
        board_keys = list(self.env_configs.keys())
        curriculum_size_per_step = int(self.config.batch_size_train * curriculum_ratio)

        # Compute board weights for weighted sampling
        if self.config.board_mix_strategy == 'weighted':
            # Weight by board area (larger boards get more training)
            areas = [r * c for r, c in self.config.board_sizes]
            total_area = sum(areas)
            board_weights = np.array([a / total_area for a in areas])
            # Cumulative for sampling
            board_cum_weights = np.cumsum(board_weights)

        # PRE-GENERATE all curriculum examples for this iteration (once per board size)
        # This avoids the slow Python loop inside generate_curriculum_batch being called every step
        curriculum_cache = {}
        curriculum_idx = {}  # Track how many we've used per board
        if curriculum_size_per_step > 0:
            # Generate enough for all steps (worst case: all steps use same board)
            total_curriculum_per_board = curriculum_size_per_step * self.config.train_steps_per_iteration
            for board_key in board_keys:
                self.rng, curr_rng = jax.random.split(self.rng)
                env_config = self.env_configs[board_key]
                curr_states, curr_policies, curr_values, curr_stats = generate_curriculum_batch(
                    curr_rng, env_config, total_curriculum_per_board,
                    jump_distribution=list(self.config.curriculum_jump_distribution),
                    return_stats=True
                )
                curriculum_cache[board_key] = {
                    'states': curr_states,
                    'policies': curr_policies,
                    'values': curr_values,
                }
                curriculum_idx[board_key] = 0
                # Accumulate stats (will be for all pre-generated, but that's fine)
                for k, v in curr_stats.items():
                    curriculum_stats_sum[k] += v

        for step_idx in range(self.config.train_steps_per_iteration):
            self.rng, step_rng, board_rng = jax.random.split(self.rng, 3)

            # Select board size based on strategy
            if self.config.board_mix_strategy == 'weighted':
                # Sample proportional to board area
                rand_val = float(jax.random.uniform(board_rng))
                board_idx = int(np.searchsorted(board_cum_weights, rand_val))
                board_idx = min(board_idx, len(board_keys) - 1)
                board_key = board_keys[board_idx]
            elif self.config.board_mix_strategy == 'round_robin':
                board_key = board_keys[step_idx % len(board_keys)]
            else:  # uniform
                board_key = board_keys[step_idx % len(board_keys)]

            env_config = self.env_configs[board_key]
            rows, cols = env_config.rows, env_config.cols
            action_space_size = 2 * rows * cols + 1

            # Sample from replay buffer
            replay_size = self.config.batch_size_train - curriculum_size_per_step

            if replay_size > 0:
                replay_batch = self.replay_buffers[board_key].sample(replay_size)
                states = replay_batch['states']
                policies = replay_batch['policy_targets']
                values = replay_batch['value_targets']
            else:
                states = np.empty((0, 6, rows, cols), dtype=np.float32)
                policies = np.empty((0, action_space_size), dtype=np.float32)
                values = np.empty((0,), dtype=np.float32)

            # Slice from pre-generated curriculum cache
            if curriculum_size_per_step > 0:
                idx = curriculum_idx[board_key]
                curr_cache = curriculum_cache[board_key]
                curr_states = curr_cache['states'][idx:idx + curriculum_size_per_step]
                curr_policies = curr_cache['policies'][idx:idx + curriculum_size_per_step]
                curr_values = curr_cache['values'][idx:idx + curriculum_size_per_step]
                curriculum_idx[board_key] = idx + curriculum_size_per_step

                states = np.concatenate([states, curr_states], axis=0)
                policies = np.concatenate([policies, curr_policies], axis=0)
                values = np.concatenate([values, curr_values], axis=0)

            batch = {
                'states': states,
                'policy_targets': policies,
                'value_targets': values,
            }

            # Train step for this board size
            train_step_fn = self.train_step_fns[board_key]
            self.params, self.batch_stats, self.opt_state, metrics = train_step_fn(
                self.params, self.batch_stats, self.opt_state, batch, step_rng
            )

            for k, v in metrics.items():
                metrics_sum[k] += float(v)

        elapsed = time.time() - start_time
        avg_metrics = {k: v / self.config.train_steps_per_iteration for k, v in metrics_sum.items()}
        avg_metrics['curriculum_ratio'] = curriculum_ratio
        avg_metrics.update(curriculum_stats_sum)

        print(f"  Training: {self.config.train_steps_per_iteration} steps ({elapsed:.1f}s) | "
              f"policy_loss: {avg_metrics['policy_loss']:.4f}, "
              f"value_loss: {avg_metrics['value_loss']:.4f}")

        return avg_metrics

    def save_checkpoint(self, path: Optional[str] = None):
        """Save checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, f"chimera_{self.iteration:06d}.pkl")

        checkpoint = {
            'params': self.params,
            'batch_stats': self.batch_stats,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_examples': self.total_examples,
            'config': self.config,
            'board_sizes': self.config.board_sizes,
            'metrics_history': self.metrics_history,
            # Curriculum state
            'current_sims': self.current_sims,
            'sims_doubled': self.sims_doubled,
            'current_lr': self.current_lr,
            'best_loss': self.best_loss,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']
        self.batch_stats = checkpoint['batch_stats']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.total_examples = checkpoint['total_examples']
        self.metrics_history = checkpoint.get('metrics_history', [])

        # Restore curriculum state
        self.current_sims = checkpoint.get('current_sims', self.config.num_simulations)
        self.sims_doubled = checkpoint.get('sims_doubled', False)
        self.current_lr = checkpoint.get('current_lr', self.config.learning_rate)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Loaded checkpoint from iteration {self.iteration} (sims={self.current_sims}, lr={self.current_lr:.2e})")

    def add_board_size(self, new_rows: int, new_cols: int):
        """Add a new board size to the chimera (post-hoc expansion)."""
        board_key = f"{new_rows}x{new_cols}"
        if board_key in self.env_configs:
            print(f"Board size {board_key} already exists")
            return

        old_board_sizes = self.config.board_sizes
        new_board_sizes = old_board_sizes + ((new_rows, new_cols),)

        # Create new expanded network
        new_network = create_chimera_network(
            board_sizes=new_board_sizes,
            num_channels=self.config.num_channels,
            num_res_blocks=self.config.num_res_blocks,
        )

        # Expand parameters
        self.rng, expand_rng = jax.random.split(self.rng)
        old_variables = {'params': self.params, 'batch_stats': self.batch_stats}
        new_variables = expand_chimera_network(
            old_variables, old_board_sizes, new_network, expand_rng
        )

        # Update state
        self.network = new_network
        self.params = new_variables['params']
        self.batch_stats = new_variables['batch_stats']
        self.config.board_sizes = new_board_sizes

        # Add new env config and buffer
        self.env_configs[board_key] = EnvConfig(
            rows=new_rows, cols=new_cols, max_turns=self.config.max_turns_per_game
        )
        self.replay_buffers[board_key] = ReplayBuffer(
            max_size=self.config.buffer_size, cols=new_cols, augment_flip=True
        )
        self.total_games[board_key] = 0
        self.total_examples[board_key] = 0

        # Add new train step fn
        self.train_step_fns[board_key] = make_chimera_train_step_fn(
            self.network, self.optimizer, board_key
        )

        print(f"Added board size {board_key}. Now training on: {list(self.env_configs.keys())}")

    def train(self):
        """Main training loop."""
        # Auto-resume
        existing = glob.glob(os.path.join(self.config.checkpoint_dir, "chimera_*.pkl"))
        if existing:
            latest = max(existing, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            self.load_checkpoint(latest)
            self.iteration += 1

        print("=" * 60)
        print("Chimera Training for Phutball")
        print("=" * 60)
        print(f"Board sizes: {list(self.env_configs.keys())}")
        print(f"Shared backbone: {self.config.num_channels}ch, {self.config.num_res_blocks} blocks")
        print(f"MCTS sims: {self.current_sims}" +
              (f" (curriculum: {self.config.sim_curriculum_initial} -> {self.config.sim_curriculum_target})"
               if self.config.sim_curriculum_enabled else ""))
        print(f"Training: {self.config.train_steps_per_iteration} steps x {self.config.batch_size_train} batch = {self.config.train_steps_per_iteration * self.config.batch_size_train:,} samples/iter")
        print(f"Devices: {jax.devices()}")
        print("=" * 60)

        for iteration in range(self.iteration, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\nIteration {iteration + 1}/{self.config.num_iterations}")
            print("-" * 40)

            # Self-play for all board sizes
            stats_per_board = self.run_self_play()

            # Training
            metrics = self.run_training()

            if metrics:
                self.metrics_history.append({
                    'iteration': iteration,
                    'total_games': dict(self.total_games),
                    'total_examples': dict(self.total_examples),
                    **metrics,
                })
                # Check for LR decay on loss plateau
                total_loss = metrics.get('total_loss', metrics.get('policy_loss', 0) + metrics.get('value_loss', 0))
                self._maybe_decay_lr(total_loss)

            # Check for sim curriculum (double sims when avg win rate hits threshold)
            self.maybe_double_sims()

            # Save to league pool if enabled
            self.maybe_save_to_league()

            # Eval vs random
            eval_win_rates = self.run_eval_vs_random()

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint()

            iter_time = time.time() - iter_start
            print(f"  Iteration time: {iter_time:.1f}s")

            if self.config.use_wandb and self.wandb_run and metrics:
                log_data = {
                    "iteration": iteration,
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/curriculum_ratio": metrics.get("curriculum_ratio", 0),
                    "train/curriculum_1jump": metrics.get("curriculum_1jump", 0),
                    "train/curriculum_2jump": metrics.get("curriculum_2jump", 0),
                    "train/curriculum_3jump": metrics.get("curriculum_3jump", 0),
                    "train/curriculum_4jump": metrics.get("curriculum_4jump", 0),
                    "train/curriculum_total": metrics.get("curriculum_total", 0),
                    "train/current_sims": self.current_sims,
                    "train/learning_rate": self.current_lr,
                }
                for bk, stats in stats_per_board.items():
                    log_data[f"selfplay/{bk}/examples"] = stats["examples"]
                    log_data[f"selfplay/{bk}/buffer_size"] = stats["buffer_size"]
                    log_data[f"selfplay/{bk}/p1_wins"] = stats["p1_wins"]
                    log_data[f"selfplay/{bk}/p2_wins"] = stats["p2_wins"]
                    log_data[f"selfplay/{bk}/draws"] = stats["draws"]
                    log_data[f"selfplay/{bk}/avg_moves"] = stats["avg_moves"]
                    log_data[f"selfplay/{bk}/avg_jump_seq"] = stats["avg_jump_seq"]
                    log_data[f"selfplay/{bk}/avg_jump_len"] = stats["avg_jump_len"]
                    log_data[f"selfplay/{bk}/adj_conv"] = stats["adj_conv"]
                for bk, wr in eval_win_rates.items():
                    log_data[f"eval/{bk}/win_rate"] = wr
                wandb.log(log_data)

        self.save_checkpoint()
        print("\nChimera training complete!")


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