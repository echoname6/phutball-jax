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
    generate_threat_recognition_state
    generate_passthrough_recognition_state
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple

# Support both import styles: as a package member (`phutball_jax.curriculum_puzzles`,
# used by server.py) or as a bare top-level module (used when test scripts inside
# phutball_jax/ are invoked directly).
try:
    from .phutball_env_jax import (
        EnvConfig, PhutballState, state_to_network_input,
        EMPTY, BALL, MAN, END_HI, END_LO, MAX_JUMP_SEQUENCE_LENGTH,
        step as env_step, get_legal_actions as env_get_legal_actions,
    )
except ImportError:
    from phutball_env_jax import (
        EnvConfig, PhutballState, state_to_network_input,
        EMPTY, BALL, MAN, END_HI, END_LO, MAX_JUMP_SEQUENCE_LENGTH,
        step as env_step, get_legal_actions as env_get_legal_actions,
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


# ============================================================================
# Threat Recognition Puzzles
# ============================================================================

def _normalize_dir(dr: int, dc: int):
    """Reduce a vector to its unit direction (sign-preserving)."""
    from math import gcd
    if dr == 0 and dc == 0:
        return (0, 0)
    g = gcd(abs(dr), abs(dc)) or 1
    return (dr // g, dc // g)


def _decode_jump_landings(actions, rows: int, cols: int):
    total = rows * cols
    out = []
    for a in actions:
        a = int(a)
        if total <= a < 2 * total:
            j = a - total
            out.append((j // cols, j % cols))
    return out


def _state_key_for_visit(state: PhutballState) -> bytes:
    """Compact key for memoizing search states. Captures everything that
    affects future legal moves: board contents, ball position, jump-sequence
    history, and active jump count."""
    return (
        np.asarray(state.board, dtype=np.int8).tobytes()
        + np.asarray(state.ball_pos, dtype=np.int8).tobytes()
        + np.asarray(state.jump_sequence, dtype=np.int8).tobytes()
        + bytes([int(state.jump_sequence_length)])
    )


def _opponent_has_winning_jump(
    state: PhutballState,
    env_config: EnvConfig,
    opp_player: int,
    max_depth: int = 14,
) -> bool:
    """
    Return True iff the opponent (`opp_player`) has any sequence of jumps
    from `state` that lands the ball in their endzone, scoring a win on
    this turn.

    Placement actions are skipped (placements end the turn without scoring,
    so they cannot constitute a winning move on a single turn).

    Sparse boards (noise-free puzzles) make the DFS cheap in practice —
    most positions have ≤2 legal jump directions and stones get consumed
    along each branch.
    """
    rows, cols = env_config.rows, env_config.cols
    total = rows * cols

    # Reset turn-state so the opponent is fresh-on-move.
    starting_state = state._replace(
        current_player=jnp.array(opp_player, dtype=jnp.int32),
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
        jump_sequence=jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32),
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )

    visited: set = set()

    def dfs(s: PhutballState, depth: int) -> bool:
        if depth > max_depth:
            return False
        legal_np = np.asarray(env_get_legal_actions(s, env_config))
        # Only consider jump actions; placements end the turn without winning.
        for a in range(total, 2 * total):
            if not bool(legal_np[a]):
                continue
            new_s = env_step(s, jnp.array(a, dtype=jnp.int32), env_config)
            if bool(new_s.terminated):
                if int(new_s.winner) == opp_player:
                    return True
                continue
            key = _state_key_for_visit(new_s)
            if key in visited:
                continue
            visited.add(key)
            if dfs(new_s, depth + 1):
                return True
        return False

    return dfs(starting_state, 0)


def generate_threat_recognition_state(
    rng: jax.Array,
    env_config: EnvConfig,
    num_jumps: int = 2,
    threat_player: int = 1,
    min_jump_len: int = 1,
    max_jump_len: int = 3,
    max_attempts: int = 64,
):
    """
    Generate a state where `threat_player` has an n-jump winning threat AND
    at least one direction-change pivot is a TRUE disruption — i.e., placing
    a defender stone at that pivot leaves the opponent with no winning jump
    sequence on their next turn.

    Per-attempt pipeline:
      1. Generate a candidate canonical winning sequence for `threat_player`
         (no noise, per methodology).
      2. Compute direction-change pivots (intermediate landings where the
         chain pivots between two unit directions). Skip same-direction chains.
      3. Sanity-check the threat: opponent must in fact have a winning jump
         from the unmodified state (`_opponent_has_winning_jump` returns True).
      4. For each candidate pivot, simulate placing a defender stone at the
         pivot and re-run the opponent's winning-jump search. A pivot is a
         TRUE disruption iff the search returns False — no opponent recovery
         exists, canonical or otherwise.
      5. Emit the puzzle iff at least one true-disruption pivot was found.
         Returned `canonical_pivots` includes only the verified true
         disruptions; any of them is a valid response.

    Same-direction chains, mid-board threats with diagonal alternatives,
    and other geometries where the opponent can recover from the canonical
    path are filtered automatically by step 4.
    """
    if num_jumps < 2:
        raise ValueError("Threat puzzles require num_jumps >= 2 (no pivot exists for n=1).")

    rows, cols = env_config.rows, env_config.cols

    for _ in range(max_attempts):
        rng, sub_rng = jax.random.split(rng)
        state, actions = generate_n_move_win_state(
            sub_rng,
            env_config,
            num_jumps=num_jumps,
            player=threat_player,
            min_jump_len=min_jump_len,
            max_jump_len=max_jump_len,
            add_noise_men=False,
            max_noise_men=0,
        )

        landings = _decode_jump_landings(actions, rows, cols)
        if len(landings) != num_jumps:
            continue

        ball_pos = (int(state.ball_pos[0]), int(state.ball_pos[1]))
        positions = [ball_pos] + landings
        dirs = []
        bad = False
        for i in range(len(positions) - 1):
            dr = positions[i+1][0] - positions[i][0]
            dc = positions[i+1][1] - positions[i][1]
            unit = _normalize_dir(dr, dc)
            if unit == (0, 0):
                bad = True
                break
            dirs.append(unit)
        if bad:
            continue

        pivot_candidates = [landings[i] for i in range(len(dirs) - 1) if dirs[i] != dirs[i+1]]
        if not pivot_candidates:
            continue

        # Sanity: opponent must actually have a winning jump from the
        # unmodified state. (If not, the candidate is malformed — skip.)
        if not _opponent_has_winning_jump(state, env_config, threat_player):
            continue

        # Test each pivot: place a defender stone, check opponent recovery.
        valid_pivots = []
        for (pr, pc) in pivot_candidates:
            modified_board = state.board.at[pr, pc].set(MAN)
            modified_state = state._replace(board=modified_board)
            if not _opponent_has_winning_jump(modified_state, env_config, threat_player):
                valid_pivots.append((pr, pc))

        if not valid_pivots:
            # Every candidate pivot leaves the opponent with an alternative
            # winning sequence — discard and retry.
            continue

        responder = 3 - threat_player
        new_state = state._replace(
            current_player=jnp.array(responder, dtype=jnp.int32),
        )
        return new_state, valid_pivots, landings, [int(a) for a in actions]

    raise RuntimeError(
        f"Failed to generate a threat puzzle (n={num_jumps}, player={threat_player}) "
        f"with a true-disruption pivot after {max_attempts} attempts. "
        f"Consider raising max_attempts or relaxing constraints."
    )


# ============================================================================
# Passthrough Recognition Puzzles
# ============================================================================

def generate_passthrough_recognition_state(
    rng: jax.Array,
    env_config: EnvConfig,
    player: int = 1,
    num_jumps: int = 2,
    max_attempts: int = 64,
):
    """
    Generate a state where `player`'s only canonical winning move requires
    passing through their own (defensive) endzone mid-sequence.

    Construction (forward-chained, no noise):
      - Ball starts in playable territory, with at least 4 rows of clearance
        from the player's own endzone (so the test isn't trivial).
      - Jump 1 is a vertical jump from the start cell into the player's own
        endzone (rows rows-2..rows-1 for P1, rows 0..1 for P2).
      - Jump 2 is a diagonal jump from the endzone landing back out, ending
        at a row strictly closer to the opponent's endzone than the start
        (and explicitly NOT in the player's own endzone — that would be a
        self-loss).
      - Stones are placed exactly along the canonical path. With no noise,
        the canonical sequence is the only legal jump sequence available, so
        no opponent-recovery search is needed.

    Currently supports num_jumps=2 only. Higher depths require chaining
    additional setup jumps; left as TODO since n=2 is the minimal test of
    the "endzone use is legal mid-sequence" concept.

    Returns:
        state:                PhutballState ready for `player` to play
        canonical_landings:   [(r1,c1), (r2,c2)] — the two-jump solution
        canonical_actions:    list of action indices for the env
        meta:                 dict carrying own_endzone_rows + start_pos for
                              the validator
    """
    if num_jumps < 2 or num_jumps > 8:
        raise ValueError("Passthrough puzzles support num_jumps in [2, 8].")

    rows, cols = env_config.rows, env_config.cols

    if player == 1:
        own_ez_rows = [rows - 2, rows - 1]  # P1 defends bottom
        forward_dr = 1                       # toward own endzone (south)
        opp_dr = -1                          # toward opp endzone (north)
    else:
        own_ez_rows = [0, 1]
        forward_dr = -1
        opp_dr = 1

    def append_leg(cur_row, cur_col, dr_step, length, dc_step,
                   start_pos, all_stones, allow_endzone_landing):
        """Try to extend the chain by one jump. Returns (ok, new_row, new_col,
        list_of_stones_added). The leg consumes (length-1) stones along the
        axis (dr_step, dc_step) and lands at length cells away."""
        next_row = cur_row + dr_step * length
        next_col = cur_col + dc_step * length
        if not (0 <= next_row < rows and 0 <= next_col < cols):
            return False, None, None, None
        if not allow_endzone_landing and next_row in own_ez_rows:
            return False, None, None, None
        new_stones = []
        for k in range(1, length):
            sr, sc = cur_row + dr_step * k, cur_col + dc_step * k
            if not (0 <= sr < rows and 0 <= sc < cols):
                return False, None, None, None
            if (sr, sc) == start_pos:
                return False, None, None, None
            if (sr, sc) in all_stones:
                return False, None, None, None
            new_stones.append((sr, sc))
        return True, next_row, next_col, new_stones

    for _ in range(max_attempts):
        # Generous split allocation: split point + start + jumps_in legs +
        # jumps_out legs (each consumes direction + length keys).
        rng, *subs = jax.random.split(rng, 16 + 4 * num_jumps)
        sub_idx = 0
        def take():
            nonlocal sub_idx
            v = subs[sub_idx]; sub_idx += 1; return v

        # 1. Pre-compute the feasible (jumps_in, jumps_out) splits and sample
        # uniformly among them. Without this filter, naive uniform sampling
        # over [1..n-1] for jumps_in skews the OUTPUT distribution toward
        # easy splits — harder splits keep failing construction and the
        # outer retry happens to land on an easy split.
        feasible_splits = []
        for ji in range(1, num_jumps):
            jo = num_jumps - ji
            ji_min = 2 * ji
            ji_max = min(6 * ji, 6 * jo - 1, rows - 4)
            if ji_min <= ji_max:
                feasible_splits.append((ji, jo))
        if not feasible_splits:
            continue
        sel = int(jax.random.randint(take(), (), 0, len(feasible_splits)))
        jumps_in, jumps_out = feasible_splits[sel]

        # 2. Pick the BALL'S START POSITION. The ball needs enough room
        # for jumps_in legs to reach the endzone, AND for jumps_out legs
        # to bring the ball back past start_row from the endzone landing.
        # Each leg is 2..6 cells. Sum of out-leg lengths must exceed in_dist
        # (final_row < start_row), so in_dist ≤ 6*jumps_out - 1.
        min_in_dist = 2 * jumps_in
        max_in_dist = min(6 * jumps_in, 6 * jumps_out - 1, rows - 4)
        if min_in_dist > max_in_dist:
            continue  # this (jumps_in, jumps_out) split is infeasible
        in_dist = int(jax.random.randint(take(), (), min_in_dist, max_in_dist + 1))
        if player == 1:
            start_row = (rows - 2) - in_dist
        else:
            start_row = 1 + in_dist
        if not (2 <= start_row < rows - 2):
            continue
        start_col = int(jax.random.randint(take(), (), 0, cols))
        start_pos = (start_row, start_col)

        # 3. Forward-chain the IN-LEG: jumps_in legs from start_pos toward own
        # endzone. Each leg is forward (dr=forward_dr) plus 0/+1/-1 column
        # offset. Total row delta must equal in_dist; the LAST leg must land
        # in an own-endzone row.
        all_stones = set()
        landings_in = []
        cur_row, cur_col = start_row, start_col
        ok = True

        # Distribute in_dist across jumps_in legs (each ∈ [2, 6]).
        # Pick lengths sequentially from the range [2, max_for_this_leg].
        remaining = in_dist
        leg_lengths = []
        for li in range(jumps_in):
            legs_left = jumps_in - li
            # Each remaining leg needs at least 2 cells; this leg's max is
            # bounded so future legs can each have ≥ 2.
            min_l = 2
            max_l = min(6, remaining - 2 * (legs_left - 1))
            if max_l < min_l:
                ok = False; break
            l = int(jax.random.randint(take(), (), min_l, max_l + 1))
            leg_lengths.append(l)
            remaining -= l
        if not ok or remaining != 0:
            continue

        # The last leg must land in own endzone. Force its direction to be
        # vertical (dc=0) to ensure the row math lines up; column constraints
        # already passed.
        for li, length in enumerate(leg_lengths):
            is_last_in_leg = (li == jumps_in - 1)
            if is_last_in_leg:
                dc_step = 0  # vertical into endzone
            else:
                ddir = int(jax.random.randint(take(), (), 0, 3))  # 0=vert, 1=col+, 2=col-
                dc_step = 0 if ddir == 0 else (1 if ddir == 1 else -1)
            ok2, nr, nc, new_stones = append_leg(
                cur_row, cur_col, forward_dr, length, dc_step,
                start_pos, all_stones, allow_endzone_landing=is_last_in_leg,
            )
            if not ok2:
                ok = False; break
            # Last leg must actually land in endzone
            if is_last_in_leg and nr not in own_ez_rows:
                ok = False; break
            for s in new_stones:
                all_stones.add(s)
            landings_in.append((nr, nc))
            cur_row, cur_col = nr, nc
        if not ok:
            continue

        # 4. Forward-chain the OUT-LEG: jumps_out legs from endzone landing
        # back toward opp endzone. First leg must be diagonal (dc != 0) so its
        # line doesn't reuse cells already consumed in the in-leg's same column.
        # Subsequent legs can be vert/diag.
        # Total out_dist must exceed in_dist so final_row strictly < start_row.
        # Sample a target out_dist first, then partition across legs (mirrors
        # the in-leg construction).
        min_out_dist = in_dist + 1
        max_out_dist = min(6 * jumps_out, in_dist + 1 + 6 * (jumps_out - 1) + 5)
        if min_out_dist > max_out_dist:
            continue
        out_dist = int(jax.random.randint(take(), (), min_out_dist, max_out_dist + 1))
        out_remaining = out_dist
        out_lengths = []
        for oi in range(jumps_out):
            legs_left = jumps_out - oi
            min_l = 2
            max_l = min(6, out_remaining - 2 * (legs_left - 1))
            if max_l < min_l:
                ok = False; break
            l = int(jax.random.randint(take(), (), min_l, max_l + 1))
            out_lengths.append(l)
            out_remaining -= l
        if not ok or out_remaining != 0:
            continue

        landings_out = []
        for oj, length in enumerate(out_lengths):
            if oj == 0:
                dc_step = 1 if int(jax.random.randint(take(), (), 0, 2)) == 0 else -1
            else:
                ddir = int(jax.random.randint(take(), (), 0, 3))
                dc_step = 0 if ddir == 0 else (1 if ddir == 1 else -1)
            ok2, nr, nc, new_stones = append_leg(
                cur_row, cur_col, opp_dr, length, dc_step,
                start_pos, all_stones, allow_endzone_landing=False,
            )
            if not ok2:
                ok = False; break
            for s in new_stones:
                all_stones.add(s)
            landings_out.append((nr, nc))
            cur_row, cur_col = nr, nc
        if not ok:
            continue

        # 5. Final landing must be strictly closer to opp endzone than the
        # start (otherwise the whole exercise was a wash).
        final_row, _ = landings_out[-1]
        if player == 1 and final_row >= start_row:
            continue
        if player == 2 and final_row <= start_row:
            continue

        landings = landings_in + landings_out

        # Build the board
        board = np.zeros((rows, cols), dtype=np.int32)
        board[0, :] = END_HI
        board[1, :] = END_HI
        board[rows - 2, :] = END_LO
        board[rows - 1, :] = END_LO
        board[start_row, start_col] = BALL
        for (r, c) in all_stones:
            board[r, c] = MAN

        jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)
        state = PhutballState(
            board=jnp.array(board, dtype=jnp.int32),
            ball_pos=jnp.array([start_row, start_col], dtype=jnp.int32),
            current_player=jnp.array(player, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            terminated=jnp.array(False, dtype=jnp.bool_),
            winner=jnp.array(0, dtype=jnp.int32),
            num_turns=jnp.array(0, dtype=jnp.int32),
            jump_sequence=jump_sequence,
            jump_sequence_length=jnp.array(0, dtype=jnp.int32),
        )

        canonical_actions = [rows * cols + r * cols + c for (r, c) in landings]
        meta = {
            'own_endzone_rows': own_ez_rows,
            'start_pos': [start_row, start_col],
        }
        return state, landings, canonical_actions, meta

    raise RuntimeError(
        f"Failed to generate a passthrough puzzle (player={player}, n={num_jumps}) "
        f"after {max_attempts} attempts."
    )


# ============================================================================
# Denial Recognition Puzzles
# ============================================================================

def generate_denial_recognition_state(
    rng: jax.Array,
    env_config: EnvConfig,
    player: int = 1,
    num_jumps: int = 5,
    max_attempts: int = 64,
):
    """
    Generate a denial puzzle. Ball is at a mid-board cell from which an
    "obvious" advance toward the opponent endzone is available, but the
    correct play is to first execute a multi-jump LOOP that consumes
    opponent-aligned setup material in the player's own defensive
    territory, RETURNS to the start cell, and then ADVANCES toward the
    opponent endzone — deliberately stopping 2-4 rows short of scoring
    this turn.

    The total turn:
        chain = [loop_1, ..., loop_K, advance_1, ..., advance_M]
                                ↑
                  ball returns to start (chain[K-1] == start_pos)

    Loop patterns (one is sampled uniformly per attempt, with a horizontal
    mirror flip for variety):
        - 'diamond':  4-jump square (K=4)
        - 'triangle': 3-jump right triangle (K=3)
    The loop's stones live in the player's defensive territory (rows ≥
    rows-7 for P1, rows ≤ 6 for P2) — i.e. the cells that represent the
    opponent's "setup material" the puzzle is testing whether the LLM
    eliminates before advancing.

    Advance phase: M = num_jumps - K vertical legs from start_pos toward
    the opponent endzone. Final landing must be 2-4 rows from the
    opponent endzone and NOT inside any endzone.

    Returns:
        state, all_landings, canonical_actions, meta where meta has:
            pattern, loop_landings, advance_landings, loop_stones, K, M,
            start_pos, opp_endzone_rows, own_endzone_rows.
    """
    if num_jumps < 4 or num_jumps > 12:
        raise ValueError("Denial puzzles support num_jumps in [4, 12].")

    rows, cols = env_config.rows, env_config.cols

    if player == 1:
        forward_dr = 1   # toward own (defensive) endzone
        opp_dr = -1      # toward opponent endzone (advance direction)
        own_ez_rows = [rows - 2, rows - 1]
        opp_ez_rows = [0, 1]
        defensive_min_row = rows - 7    # rows ≥ 14 are "deep enough" to count as defensive
        # final_row constraint: net forward progress past start, not in either
        # endzone. The advance doesn't need to press up to row 2-4 — short
        # advances are fine. Just must advance some.
        final_row_min, final_row_max = 2, None  # max derived from start_row
    else:
        forward_dr = -1
        opp_dr = 1
        own_ez_rows = [0, 1]
        opp_ez_rows = [rows - 2, rows - 1]
        defensive_max_row = 6
        final_row_min, final_row_max = None, rows - 3

    for _ in range(max_attempts):
        rng, *subs = jax.random.split(rng, 24)
        idx = [0]
        def take():
            v = subs[idx[0]]; idx[0] += 1; return v

        # 1. Loop pattern. Each pattern is a closed polygon traversal that
        # returns the ball to start_pos. Patterns are sampled uniformly.
        # Hook adds asymmetry so the loop doesn't read as a perfect square
        # — closer to the organic shapes that emerge in actual play.
        # Each pattern is a closed loop returning to start, with all sides
        # along Phutball's 8-direction grid (axis or 45° diagonal). The loop
        # must dip into the player's defensive territory so its stones
        # represent opponent setup material.
        feasible = []
        if num_jumps - 3 >= 1:
            feasible.append(('triangle', 3))
        if num_jumps - 4 >= 1:
            feasible.append(('rect', 4))
            feasible.append(('tilted', 4))         # diamond rotated 45° pointing south
            feasible.append(('parallelogram', 4))  # H + diag + H + diag
        if not feasible:
            continue
        pattern, K_base = feasible[int(jax.random.randint(take(), (), 0, len(feasible)))]
        # K_extras (loop fragmentation count) is sampled AFTER leg specs are
        # built — see step 6. We need to know each leg's length to cap K_extras
        # at the number of legs that can actually accept a stone-gap.

        # 2. Sample horizontal mirror (+1 = loop extends right, -1 = left)
        mirror = 1 if int(jax.random.randint(take(), (), 0, 2)) == 0 else -1

        # 3. Pick start row. Mid-board placement; advance just needs to be
        # NET-progress (any amount past start), so we don't constrain by M.
        # Loop dips into defensive territory (≥ rows-7 for P1) so the loop
        # height bounds are checked when we sample width/height below.
        if player == 1:
            sr_min = max(7, 4)
            sr_max = min(11, rows - 9)
        else:
            sr_min = max(rows - 12, 9)
            sr_max = min(rows - 8, rows - 5)
        if sr_min > sr_max:
            continue
        start_row = int(jax.random.randint(take(), (), sr_min, sr_max + 1))

        # 4. Pick start col. Loop side is bounded by what fits horizontally.
        sc_min = 4 if mirror < 0 else 1
        sc_max = (cols - 5) if mirror > 0 else (cols - 2)
        if sc_min > sc_max:
            continue
        start_col = int(jax.random.randint(take(), (), sc_min, sc_max + 1))

        # 5. Sample loop dimensions. Width (horizontal extent) and height
        # (vertical extent into defensive territory) are sampled
        # independently for rect/hook so the cycle isn't necessarily square.
        # Triangle stays isoceles because Phutball diagonal jumps must travel
        # equal rows and cols per step.
        if player == 1:
            min_height = max(3, defensive_min_row - start_row)
            max_height_row = rows - 2 - start_row - 1
        else:
            min_height = max(3, start_row - defensive_max_row)
            max_height_row = start_row - 2 - 1
        if mirror > 0:
            max_width_col = cols - 1 - start_col
        else:
            max_width_col = start_col
        max_height = min(6, max_height_row)
        max_width = min(6, max_width_col)

        if pattern == 'triangle':
            min_dim = max(min_height, 3)
            max_dim = min(max_height, max_width)
            if min_dim > max_dim:
                continue
            width = height = int(jax.random.randint(take(), (), min_dim, max_dim + 1))
        elif pattern == 'tilted':
            # Rotated diamond pointing south: 4 diagonal legs of equal length s.
            # Total north-south span is 2s; need ≥ defensive penetration so
            # 2s ≥ defensive_min_row - start_row (P1) etc. Side dim is s.
            need_south = (defensive_min_row - start_row) if player == 1 else (start_row - defensive_max_row)
            min_dim = max(2, (need_south + 1) // 2)
            # Also need sc ± s on board (loop extends both ways from start_col)
            max_dim = min(max_height // 2, max_width, start_col, cols - 1 - start_col)
            if min_dim > max_dim:
                continue
            width = height = int(jax.random.randint(take(), (), min_dim, max_dim + 1))
        elif pattern == 'parallelogram':
            # Horizontal width + diagonal height: total south-extent = height,
            # total east-extent = width + height (the diagonal adds width).
            if min_height > max_height or 3 > max_width:
                continue
            height = int(jax.random.randint(take(), (), min_height, max_height + 1))
            # Diagonal extends past width by `height` cells; total horizontal
            # extent of the loop is width + height. Constrain accordingly.
            if mirror > 0:
                max_w_par = cols - 1 - start_col - height
            else:
                max_w_par = start_col - height
            if max_w_par < 3:
                continue
            width = int(jax.random.randint(take(), (), 3, max_w_par + 1))
        else:  # rect
            if min_height > max_height or 3 > max_width:
                continue
            height = int(jax.random.randint(take(), (), min_height, max_height + 1))
            width = int(jax.random.randint(take(), (), 3, max_width + 1))

        start_pos = (start_row, start_col)

        # 6. Build leg specs as (start_cell, end_cell, dr_step, dc_step, length).
        # We then fragment each leg with stone-gaps (up to one gap per leg)
        # to add intermediate landings. The final loop_landings + loop_stones
        # are computed from the leg specs after gap insertion.
        leg_specs = []
        if pattern == 'rect':
            c1 = (start_row, start_col + mirror * width)
            c2 = (start_row + forward_dr * height, start_col + mirror * width)
            c3 = (start_row + forward_dr * height, start_col)
            leg_specs = [
                (start_pos, c1, 0, mirror, width),
                (c1, c2, forward_dr, 0, height),
                (c2, c3, 0, -mirror, width),
                (c3, start_pos, opp_dr, 0, height),
            ]
        elif pattern == 'triangle':
            c1 = (start_row, start_col + mirror * width)
            c2 = (start_row + forward_dr * height, start_col)
            # Hypotenuse from c1 to c2 is diagonal (forward_dr, -mirror).
            leg_specs = [
                (start_pos, c1, 0, mirror, width),
                (c1, c2, forward_dr, -mirror, height),
                (c2, start_pos, opp_dr, 0, height),
            ]
        elif pattern == 'tilted':
            s = width
            c1 = (start_row + forward_dr * s, start_col + mirror * s)         # diag forward+mirror
            c2 = (start_row + 2 * forward_dr * s, start_col)                  # diag forward+(-mirror)
            c3 = (start_row + forward_dr * s, start_col - mirror * s)         # diag opp+(-mirror)
            leg_specs = [
                (start_pos, c1, forward_dr, mirror, s),
                (c1, c2, forward_dr, -mirror, s),
                (c2, c3, opp_dr, -mirror, s),
                (c3, start_pos, opp_dr, mirror, s),
            ]
        else:  # parallelogram
            c1 = (start_row, start_col + mirror * width)                                       # E width
            c2 = (start_row + forward_dr * height, start_col + mirror * (width + height))      # diag SE
            c3 = (start_row + forward_dr * height, start_col + mirror * height)                # W width
            leg_specs = [
                (start_pos, c1, 0, mirror, width),
                (c1, c2, forward_dr, mirror, height),
                (c2, c3, 0, -mirror, width),
                (c3, start_pos, opp_dr, -mirror, height),
            ]

        # Sample K_extras NOW that we know each leg's length. Fragmentable
        # legs are those with length ≥ 4 (need ≥1 stone before AND after the
        # gap, plus the gap cell itself). The cap is the smaller of:
        #   - fragmentable leg count (geometric ceiling)
        #   - num_jumps - K_base - 1 (need M ≥ 1)
        # Sampled uniformly so high K_loop is just as likely as low when both
        # are feasible — fixes the prior bias toward shallow loops at high n.
        fragmentable = [i for i, leg in enumerate(leg_specs) if leg[4] >= 4]
        max_extras = min(len(fragmentable), num_jumps - K_base - 1)
        if max_extras < 0:
            continue
        if max_extras == 0:
            K_extras = 0
        else:
            K_extras = int(jax.random.randint(take(), (), 0, max_extras + 1))
        K = K_base + K_extras
        M = num_jumps - K
        if M < 1:
            continue
        if K_extras > 0:
            perm = jax.random.permutation(take(), jnp.array(fragmentable))
            chosen = set(int(perm[i]) for i in range(K_extras))
        else:
            chosen = set()

        loop_landings = []
        loop_stones = []
        ok = True
        for i, (start_cell, end_cell, dr_step, dc_step, length) in enumerate(leg_specs):
            sr, sc = start_cell
            if i in chosen:
                # Sample gap position g ∈ [2, length-2]. The g range deliberately
                # excludes g=1 and g=length-1 so the cell immediately adjacent to
                # each corner is always a stone, never the gap — every leg starts
                # with the ball jumping over at least one stone before reaching
                # the gap landing, and ends with at least one stone before
                # reaching the next corner. Corners themselves are open
                # intersections (empty landings).
                g = int(jax.random.randint(take(), (), 2, length - 1))
                # First sub-leg stones (cells 1..g-1)
                for k in range(1, g):
                    cell = (sr + dr_step * k, sc + dc_step * k)
                    loop_stones.append(cell)
                # Intermediate landing (the gap cell)
                loop_landings.append((sr + dr_step * g, sc + dc_step * g))
                # Second sub-leg stones (cells g+1..length-1)
                for k in range(g + 1, length):
                    cell = (sr + dr_step * k, sc + dc_step * k)
                    loop_stones.append(cell)
                # End-of-leg landing
                loop_landings.append(end_cell)
            else:
                for k in range(1, length):
                    cell = (sr + dr_step * k, sc + dc_step * k)
                    loop_stones.append(cell)
                loop_landings.append(end_cell)
        if not ok:
            continue


        # 7. Validate landings: on board, not in any endzone (we don't want
        #    the loop to traverse an endzone — that's a different puzzle).
        ok = True
        for (r, c) in loop_landings:
            if not (0 <= r < rows and 0 <= c < cols):
                ok = False; break
            if r in own_ez_rows or r in opp_ez_rows:
                ok = False; break
        if not ok:
            continue

        # 8. Validate stones: on board, no collision, none on start.
        all_stones = set()
        for s in loop_stones:
            r, c = s
            if not (0 <= r < rows and 0 <= c < cols):
                ok = False; break
            if (r, c) == start_pos:
                ok = False; break
            if s in all_stones:
                ok = False; break
            all_stones.add(s)
        if not ok:
            continue

        # 9. Build advance: M vertical legs from start toward opp endzone.
        # final_row: any row strictly past start (toward opp), not in either
        # endzone. Advance just needs net progress; a single short hop counts
        # as much as a deep one for puzzle correctness.
        adv_total_min, adv_total_max = 2 * M, 6 * M
        if player == 1:
            # P1: advance north, final_row < start_row, ≥ 2 (out of opp ez).
            fr_min = max(final_row_min, start_row - adv_total_max)
            fr_max = min(start_row - 1, start_row - adv_total_min)
        else:
            fr_min = max(start_row + 1, start_row + adv_total_min)
            fr_max = min(final_row_max, start_row + adv_total_max)
        if fr_min > fr_max:
            continue
        final_row = int(jax.random.randint(take(), (), fr_min, fr_max + 1))
        adv_dist = abs(final_row - start_row)

        # Partition adv_dist across M legs (each in [2, 6]).
        remaining = adv_dist
        adv_lengths = []
        for ai in range(M):
            legs_left = M - ai
            min_l = 2
            max_l = min(6, remaining - 2 * (legs_left - 1))
            if max_l < min_l:
                ok = False; break
            l = int(jax.random.randint(take(), (), min_l, max_l + 1))
            adv_lengths.append(l)
            remaining -= l
        if not ok or remaining != 0:
            continue

        # 10. Build advance landings + stones (all vertical for v1).
        advance_landings = []
        cur_row, cur_col = start_row, start_col
        for length in adv_lengths:
            nr = cur_row + opp_dr * length
            nc = cur_col
            if not (0 <= nr < rows and 0 <= nc < cols):
                ok = False; break
            if nr in own_ez_rows:
                ok = False; break
            for k in range(1, length):
                sr, sc = cur_row + opp_dr * k, cur_col
                if (sr, sc) == start_pos:
                    ok = False; break
                if (sr, sc) in all_stones:
                    ok = False; break
                all_stones.add((sr, sc))
            if not ok: break
            advance_landings.append((nr, nc))
            cur_row, cur_col = nr, nc
        if not ok:
            continue

        # Final landing constraints: not in either endzone, must be net
        # progress past start (toward opp).
        final_pos = advance_landings[-1]
        if final_pos[0] in opp_ez_rows or final_pos[0] in own_ez_rows:
            continue
        if player == 1 and final_pos[0] >= start_row:
            continue
        if player == 2 and final_pos[0] <= start_row:
            continue

        # 11. Build the state.
        board = np.zeros((rows, cols), dtype=np.int32)
        board[0, :] = END_HI
        board[1, :] = END_HI
        board[rows - 2, :] = END_LO
        board[rows - 1, :] = END_LO
        board[start_row, start_col] = BALL
        for (r, c) in all_stones:
            board[r, c] = MAN

        jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)
        state = PhutballState(
            board=jnp.array(board, dtype=jnp.int32),
            ball_pos=jnp.array([start_row, start_col], dtype=jnp.int32),
            current_player=jnp.array(player, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            terminated=jnp.array(False, dtype=jnp.bool_),
            winner=jnp.array(0, dtype=jnp.int32),
            num_turns=jnp.array(0, dtype=jnp.int32),
            jump_sequence=jump_sequence,
            jump_sequence_length=jnp.array(0, dtype=jnp.int32),
        )

        all_landings = loop_landings + advance_landings
        canonical_actions = [rows * cols + r * cols + c for (r, c) in all_landings]

        meta = {
            'pattern': pattern,
            'mirror': mirror,
            'width': width,
            'height': height,
            'K': K,
            'M': M,
            'loop_landings': [list(p) for p in loop_landings],
            'advance_landings': [list(p) for p in advance_landings],
            'loop_stones': [list(s) for s in loop_stones],
            'start_pos': [start_row, start_col],
            'opp_endzone_rows': list(opp_ez_rows),
            'own_endzone_rows': list(own_ez_rows),
        }
        return state, all_landings, canonical_actions, meta

    raise RuntimeError(
        f"Failed to generate a denial puzzle (player={player}, n={num_jumps}) "
        f"after {max_attempts} attempts."
    )
