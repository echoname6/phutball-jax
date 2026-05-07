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
    if num_jumps != 2:
        raise NotImplementedError(
            "Passthrough puzzles currently support only num_jumps=2. "
            "Higher depths require chaining setup jumps (TODO)."
        )

    rows, cols = env_config.rows, env_config.cols

    if player == 1:
        own_ez_rows = [rows - 2, rows - 1]  # P1 defends bottom
        forward_dr = 1                       # ball moves south to reach own endzone
    else:
        own_ez_rows = [0, 1]
        forward_dr = -1

    for _ in range(max_attempts):
        rng, *subs = jax.random.split(rng, 6)

        # 1. Pick start row: at least 4 rows from own endzone, but close enough
        # that one jump (max 9 cells = 8 stones consumed) can land in it.
        dist_to_endzone = int(jax.random.randint(subs[0], (), 4, 10))  # 4..9
        if player == 1:
            start_row = (rows - 2) - dist_to_endzone
        else:
            start_row = 1 + dist_to_endzone
        if not (2 <= start_row < rows - 2):
            continue

        start_col = int(jax.random.randint(subs[1], (), 0, cols))

        # 2. Jump 1: vertical into own endzone. Pick which endzone row.
        landing1_row = own_ez_rows[int(jax.random.randint(subs[2], (), 0, 2))]
        delta1 = abs(landing1_row - start_row)
        if delta1 - 1 < 1 or delta1 - 1 > 8:
            continue  # need at least 1 stone, at most 8

        jump1_stones = [(start_row + k * forward_dr, start_col) for k in range(1, delta1)]

        # 3. Jump 2: diagonal back out. NW or NE for P1; SW or SE for P2.
        diag_col_dir = 1 if int(jax.random.randint(subs[3], (), 0, 2)) == 0 else -1

        # delta2 must be > delta1 so target_row crosses past start_row toward opp endzone.
        # Bounded by max jump length and by board edge in the backward-row direction.
        if player == 1:
            min_delta2 = delta1 + 1
            max_delta2_by_row = landing1_row              # so target_row >= 0
        else:
            min_delta2 = delta1 + 1
            max_delta2_by_row = (rows - 1) - landing1_row  # so target_row <= rows-1
        max_delta2_by_chain = 9  # max 8 stones + 1 step
        max_delta2 = min(max_delta2_by_row, max_delta2_by_chain)
        if min_delta2 > max_delta2:
            continue

        delta2 = int(jax.random.randint(subs[4], (), min_delta2, max_delta2 + 1))
        target_row = landing1_row - delta2 * forward_dr  # ↑ for P1, ↓ for P2
        target_col = start_col + delta2 * diag_col_dir
        if not (0 <= target_col < cols):
            continue
        if target_row in own_ez_rows:
            continue  # final landing must NOT be in own endzone (self-loss)
        if target_row < 0 or target_row >= rows:
            continue

        jump2_stones = [
            (landing1_row - k * forward_dr, start_col + k * diag_col_dir)
            for k in range(1, delta2)
        ]

        # Check no stone collides with start cell or overlaps another stone or
        # falls off the board.
        all_stones = set()
        ok = True
        for s in jump1_stones + jump2_stones:
            r, c = s
            if not (0 <= r < rows and 0 <= c < cols):
                ok = False; break
            if (r, c) == (start_row, start_col):
                ok = False; break
            if s in all_stones:
                ok = False; break
            all_stones.add(s)
        if not ok:
            continue

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

        landings = [(landing1_row, start_col), (target_row, target_col)]
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
