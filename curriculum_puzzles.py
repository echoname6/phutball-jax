"""
Procedural curriculum puzzles for Phutball.

Generators construct PhutballState board states with a known winning action
(or sequence). Used for AlphaZero curriculum pretraining and for offline
LLM evaluation. Kept independent of the AlphaZero training loop so the dev
server, test harness, and any future evaluator can share one source of truth.

Public API:
    generate_one_move_win_state
    generate_n_move_win_state
    generate_two_move_win_state
    generate_curriculum_batch
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple

from phutball_env_jax import (
    EnvConfig, PhutballState, state_to_network_input,
    EMPTY, BALL, MAN, END_HI, END_LO, MAX_JUMP_SEQUENCE_LENGTH,
)


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

    # P1 wins by reaching row <= 1 (top endzone, END_HI)
    # P2 wins by reaching row >= rows-2 (bottom endzone, END_LO)

    # Directions toward each player's goal
    if player == 2:
        # P2 moves down (increasing row): vertical and both diagonals
        directions = [(1, 0), (1, -1), (1, 1)]  # down, down-left, down-right
        endzone_rows = [rows - 2, rows - 1]  # Can land on either endzone row
    else:
        # P1 moves up (decreasing row): vertical and both diagonals
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
        if player == 2:
            max_by_height = landing_row - 2  # Can't start in endzone (rows 0,1 are P1's)
        else:
            max_by_height = (rows - 1) - landing_row - 2  # Can't start in P2's endzone
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
    if player == 2:
        # Ball shouldn't be in P1's endzone (rows 0,1) or P2's endzone (rows-2, rows-1)
        if ball_row <= 1 or ball_row >= rows - 2:
            # Reduce jump length to fit
            if ball_row <= 1:
                ball_row = 2
                jump_len = (landing_row - ball_row) // abs(dr) - 1
            else:
                # This shouldn't happen for P1 moving up, but safety check
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
    if player == 2:
        goal_directions = [(1, 0), (1, -1), (1, 1)]  # P2 attacks bottom: down/down-left/down-right
        endzone_rows = [rows - 2, rows - 1]
    else:
        goal_directions = [(-1, 0), (-1, -1), (-1, 1)]  # P1 attacks top: up/up-left/up-right
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
                    if (player == 2 and dr < 0) or (player == 1 and dr > 0):
                        dr = 1 if player == 2 else -1  # forward toward goal
                    prev_row = curr_row - (jump_len + 1) * dr
                else:
                    # Was horizontal (dr=0) - flip direction
                    dc = -dc
                    prev_col = curr_col - (jump_len + 1) * dc
                    if prev_col < 0 or prev_col >= cols:
                        # Still out of bounds, switch to goal-directed vertical
                        dr = 1 if player == 2 else -1
                        dc = 0
                        prev_row = curr_row - (jump_len + 1) * dr
                        prev_col = curr_col

            # For intermediate positions (not the ball), must be in playable area
            # For the ball (last iteration), also must be in playable area
            # Note: for horizontal jumps (dr=0), row doesn't change so check column instead
            if dr != 0 and (prev_row <= 1 or prev_row >= rows - 2):
                # Need to reduce jump length to fit (vertical/diagonal case)
                if player == 2:
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
                if player == 2:
                    # P1 moves up (toward lower rows), so prev should be below curr
                    prev_row = curr_row - 2
                    dr = 1  # Ensure correct direction
                else:
                    # P2 moves down (toward higher rows), so prev should be above curr
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
                    if player == 2:
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
                    (1 if player == 2 else -1, 0),   # goal-directed vertical
                    (0, 1), (0, -1),                  # horizontal
                    (1 if player == 2 else -1, 1),   # goal-directed diagonal right
                    (1 if player == 2 else -1, -1),  # goal-directed diagonal left
                    (-1 if player == 2 else 1, 0),   # backward vertical
                    (-1 if player == 2 else 1, 1),   # backward diagonal right
                    (-1 if player == 2 else 1, -1),  # backward diagonal left
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
