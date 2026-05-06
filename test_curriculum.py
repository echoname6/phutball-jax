"""
Test suite for curriculum learning: 1-move winning state generation.

Tests various board sizes and validates that generated states have valid winning jumps.
Uses colorful terminal output inspired by the UI theme.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List

from phutball_env_jax import (
    EnvConfig, PhutballState, reset, step, get_legal_actions,
    EMPTY, BALL, MAN, END_HI, END_LO,
)
from curriculum_puzzles import (
    generate_one_move_win_state,
    generate_two_move_win_state,
    generate_n_move_win_state,
    generate_curriculum_batch,
)


# ============================================================================
# ANSI Color Codes (inspired by ui/theme.py)
# ============================================================================

class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Theme-inspired colors
    PHUTBALL = BRIGHT_WHITE + BOLD  # White ball
    MAN_PIECE = BRIGHT_BLACK + BOLD  # Black men (dark on terminal)
    ENDZONE_P1 = RED  # Player 1 endzone (bottom, rows-2, rows-1)
    ENDZONE_P2 = BLUE  # Player 2 endzone (top, rows 0, 1)
    WINNING_PATH = BRIGHT_CYAN  # Highlight winning jump path
    GRID = DIM  # Grid lines
    HEADER = BRIGHT_MAGENTA + BOLD
    SUCCESS = BRIGHT_GREEN
    ERROR = BRIGHT_RED
    INFO = BRIGHT_YELLOW


def render_board_colored(
    state: PhutballState,
    env_config: EnvConfig,
    winning_action: int = None,
    title: str = None,
) -> str:
    """
    Render the board with ANSI colors for terminal display.

    Highlights:
    - Phutball (ball) in bright white
    - Men in dark/black
    - P1 endzone (bottom) in red
    - P2 endzone (top) in blue
    - Winning jump landing position in cyan
    """
    rows, cols = env_config.rows, env_config.cols
    board = np.array(state.board)
    ball_pos = (int(state.ball_pos[0]), int(state.ball_pos[1]))

    # Calculate landing position from winning action
    landing_pos = None
    if winning_action is not None:
        total_positions = rows * cols
        if winning_action >= total_positions:
            landing_idx = winning_action - total_positions
            landing_pos = (landing_idx // cols, landing_idx % cols)

    lines = []

    # Title
    if title:
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}  {title}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")

    # Column headers
    col_header = "    "
    for c in range(cols):
        col_header += f"{Colors.DIM}{c % 10} {Colors.RESET}"
    lines.append(col_header)

    # Board rows
    for r in range(rows):
        row_str = f"{Colors.DIM}{r:2d}{Colors.RESET}  "

        for c in range(cols):
            tile = board[r, c]

            # Determine cell background/style
            is_p2_endzone = r <= 1
            is_p1_endzone = r >= rows - 2
            is_landing = landing_pos and (r, c) == landing_pos
            is_ball = (r, c) == ball_pos

            # Choose character and color
            if tile == BALL or is_ball:
                char = "●"
                color = Colors.PHUTBALL
            elif tile == MAN:
                char = "○"
                color = Colors.WHITE + Colors.BOLD
            elif is_landing:
                char = "◎"
                color = Colors.WINNING_PATH
            elif is_p2_endzone:
                char = "░"
                color = Colors.ENDZONE_P2
            elif is_p1_endzone:
                char = "░"
                color = Colors.ENDZONE_P1
            else:
                char = "·"
                color = Colors.DIM

            row_str += f"{color}{char}{Colors.RESET} "

        # Row label on right side too
        row_str += f" {Colors.DIM}{r:2d}{Colors.RESET}"
        lines.append(row_str)

    # State info
    player = int(state.current_player)
    player_color = Colors.ENDZONE_P1 if player == 1 else Colors.ENDZONE_P2
    lines.append("")
    lines.append(f"  {Colors.INFO}Player to move: {player_color}P{player}{Colors.RESET}")
    lines.append(f"  {Colors.INFO}Ball position: ({ball_pos[0]}, {ball_pos[1]}){Colors.RESET}")
    if landing_pos:
        lines.append(f"  {Colors.WINNING_PATH}Winning jump lands at: ({landing_pos[0]}, {landing_pos[1]}){Colors.RESET}")

    return "\n".join(lines)


def render_board_two_jump(
    state: PhutballState,
    env_config: EnvConfig,
    first_action: int,
    second_action: int,
    title: str = None,
) -> str:
    """
    Render a 2-jump winning state with both landing positions highlighted.
    """
    rows, cols = env_config.rows, env_config.cols
    board = np.array(state.board)
    ball_pos = (int(state.ball_pos[0]), int(state.ball_pos[1]))
    total_positions = rows * cols

    # Calculate positions
    mid_pos = None
    end_pos = None
    if first_action >= total_positions:
        mid_idx = first_action - total_positions
        mid_pos = (mid_idx // cols, mid_idx % cols)
    if second_action >= total_positions:
        end_idx = second_action - total_positions
        end_pos = (end_idx // cols, end_idx % cols)

    lines = []

    if title:
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}  {title}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")

    # Column headers
    col_header = "    "
    for c in range(cols):
        col_header += f"{Colors.DIM}{c % 10} {Colors.RESET}"
    lines.append(col_header)

    for r in range(rows):
        row_str = f"{Colors.DIM}{r:2d}{Colors.RESET}  "

        for c in range(cols):
            tile = board[r, c]
            is_p2_endzone = r <= 1
            is_p1_endzone = r >= rows - 2
            is_mid = mid_pos and (r, c) == mid_pos
            is_end = end_pos and (r, c) == end_pos
            is_ball = (r, c) == ball_pos

            if tile == BALL or is_ball:
                char = "●"
                color = Colors.PHUTBALL
            elif tile == MAN:
                char = "○"
                color = Colors.WHITE + Colors.BOLD
            elif is_mid:
                char = "①"  # First jump lands here
                color = Colors.BRIGHT_YELLOW
            elif is_end:
                char = "②"  # Second jump lands here (wins)
                color = Colors.WINNING_PATH
            elif is_p2_endzone:
                char = "░"
                color = Colors.ENDZONE_P2
            elif is_p1_endzone:
                char = "░"
                color = Colors.ENDZONE_P1
            else:
                char = "·"
                color = Colors.DIM

            row_str += f"{color}{char}{Colors.RESET} "

        row_str += f" {Colors.DIM}{r:2d}{Colors.RESET}"
        lines.append(row_str)

    player = int(state.current_player)
    player_color = Colors.ENDZONE_P1 if player == 1 else Colors.ENDZONE_P2
    lines.append("")
    lines.append(f"  {Colors.INFO}Player to move: {player_color}P{player}{Colors.RESET}")
    lines.append(f"  {Colors.INFO}Ball position: ({ball_pos[0]}, {ball_pos[1]}){Colors.RESET}")
    if mid_pos:
        lines.append(f"  {Colors.BRIGHT_YELLOW}Jump 1 lands at: ({mid_pos[0]}, {mid_pos[1]}){Colors.RESET}")
    if end_pos:
        lines.append(f"  {Colors.WINNING_PATH}Jump 2 lands at: ({end_pos[0]}, {end_pos[1]}) - WINS{Colors.RESET}")

    return "\n".join(lines)


def validate_two_jump_win(
    state: PhutballState,
    env_config: EnvConfig,
    first_action: int,
    second_action: int,
) -> Tuple[bool, str]:
    """
    Validate that executing first_action then second_action wins the game.
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    # Check both are jump actions
    if first_action < total_positions or first_action >= 2 * total_positions:
        return False, "First action is not a jump"
    if second_action < total_positions or second_action >= 2 * total_positions:
        return False, "Second action is not a jump"

    # Check first action is legal
    legal1 = get_legal_actions(state, env_config)
    if not legal1[first_action]:
        return False, "First action is not legal"

    # Execute first jump
    state_after_1 = step(state, jnp.array(first_action, dtype=jnp.int32), env_config)

    # Should not terminate yet
    if state_after_1.terminated:
        return False, "Game terminated after first jump (should need 2)"

    # Check second action is legal from new state
    legal2 = get_legal_actions(state_after_1, env_config)
    if not legal2[second_action]:
        return False, f"Second action not legal after first jump"

    # Execute second jump
    state_after_2 = step(state_after_1, jnp.array(second_action, dtype=jnp.int32), env_config)

    if not state_after_2.terminated:
        return False, "Game did not terminate after second jump"

    expected_winner = int(state.current_player)
    actual_winner = int(state_after_2.winner)

    if actual_winner != expected_winner:
        return False, f"Wrong winner: expected P{expected_winner}, got P{actual_winner}"

    return True, f"Valid! P{expected_winner} wins in 2 jumps"


def validate_winning_action(
    state: PhutballState,
    env_config: EnvConfig,
    winning_action: int,
) -> Tuple[bool, str]:
    """
    Validate that the winning action is legal and actually wins the game.

    Returns:
        (is_valid, message)
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    # Check action is a jump
    if winning_action < total_positions:
        return False, "Action is a placement, not a jump"
    if winning_action >= 2 * total_positions:
        return False, "Action is halt, not a jump"

    # Check action is legal
    legal_mask = get_legal_actions(state, env_config)
    if not legal_mask[winning_action]:
        return False, "Action is not legal"

    # Execute the action and check for win
    new_state = step(state, jnp.array(winning_action, dtype=jnp.int32), env_config)

    if not new_state.terminated:
        return False, "Action does not terminate the game"

    expected_winner = int(state.current_player)
    actual_winner = int(new_state.winner)

    if actual_winner != expected_winner:
        return False, f"Wrong winner: expected P{expected_winner}, got P{actual_winner}"

    return True, f"Valid! P{expected_winner} wins"


def test_single_board_size(rows: int, cols: int, num_samples: int = 10) -> Tuple[int, int]:
    """
    Test curriculum generation for a specific board size.

    Returns:
        (num_passed, num_failed)
    """
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(42)

    passed = 0
    failed = 0

    print(f"\n{Colors.HEADER}Testing {rows}x{cols} board ({num_samples} samples){Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2

        try:
            state, winning_action = generate_one_move_win_state(
                state_rng, env_config, player=player
            )

            is_valid, message = validate_winning_action(state, env_config, winning_action)

            if is_valid:
                passed += 1
                status = f"{Colors.SUCCESS}✓ PASS{Colors.RESET}"
            else:
                failed += 1
                status = f"{Colors.ERROR}✗ FAIL{Colors.RESET}"
                print(f"  Sample {i+1}: {status} - {message}")
                # Show the failed board
                print(render_board_colored(state, env_config, winning_action,
                                          f"Failed P{player} sample"))
        except Exception as e:
            failed += 1
            print(f"  Sample {i+1}: {Colors.ERROR}✗ ERROR{Colors.RESET} - {e}")

    # Summary
    pct = 100 * passed / num_samples if num_samples > 0 else 0
    color = Colors.SUCCESS if failed == 0 else Colors.ERROR
    print(f"  {color}Results: {passed}/{num_samples} passed ({pct:.1f}%){Colors.RESET}")

    return passed, failed


def test_curriculum_batch(rows: int, cols: int, batch_size: int = 16) -> Tuple[int, int]:
    """Test the batch generation function."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(123)

    print(f"\n{Colors.HEADER}Testing batch generation ({batch_size} samples){Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

    states, policies, values = generate_curriculum_batch(rng, env_config, batch_size)

    # Check shapes
    expected_obs_shape = (batch_size, 6, rows, cols)
    expected_policy_shape = (batch_size, 2 * rows * cols + 1)
    expected_value_shape = (batch_size,)

    passed = 0
    failed = 0

    if states.shape == expected_obs_shape:
        print(f"  {Colors.SUCCESS}✓{Colors.RESET} States shape: {states.shape}")
        passed += 1
    else:
        print(f"  {Colors.ERROR}✗{Colors.RESET} States shape: {states.shape} (expected {expected_obs_shape})")
        failed += 1

    if policies.shape == expected_policy_shape:
        print(f"  {Colors.SUCCESS}✓{Colors.RESET} Policies shape: {policies.shape}")
        passed += 1
    else:
        print(f"  {Colors.ERROR}✗{Colors.RESET} Policies shape: {policies.shape} (expected {expected_policy_shape})")
        failed += 1

    if values.shape == expected_value_shape:
        print(f"  {Colors.SUCCESS}✓{Colors.RESET} Values shape: {values.shape}")
        passed += 1
    else:
        print(f"  {Colors.ERROR}✗{Colors.RESET} Values shape: {values.shape} (expected {expected_value_shape})")
        failed += 1

    # Check all values are +1 (current player wins)
    if np.allclose(values, 1.0):
        print(f"  {Colors.SUCCESS}✓{Colors.RESET} All values are +1")
        passed += 1
    else:
        print(f"  {Colors.ERROR}✗{Colors.RESET} Values not all +1: {values}")
        failed += 1

    # Check policies are one-hot and point to jumps
    total_positions = rows * cols
    for i in range(batch_size):
        policy = policies[i]
        action = np.argmax(policy)
        if policy.sum() != 1.0:
            print(f"  {Colors.ERROR}✗{Colors.RESET} Policy {i} not one-hot")
            failed += 1
        elif action < total_positions or action >= 2 * total_positions:
            print(f"  {Colors.ERROR}✗{Colors.RESET} Policy {i} points to non-jump action {action}")
            failed += 1
        else:
            passed += 1

    return passed, failed


def test_two_jump_board_size(rows: int, cols: int, num_samples: int = 10) -> Tuple[int, int]:
    """Test 2-jump curriculum generation for a specific board size."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(123)

    passed = 0
    failed = 0

    print(f"\n{Colors.HEADER}Testing 2-JUMP {rows}x{cols} board ({num_samples} samples){Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2

        try:
            state, first_action, second_action = generate_two_move_win_state(
                state_rng, env_config, player=player
            )

            is_valid, message = validate_two_jump_win(
                state, env_config, first_action, second_action
            )

            if is_valid:
                passed += 1
            else:
                failed += 1
                print(f"  Sample {i+1}: {Colors.ERROR}✗ FAIL{Colors.RESET} - {message}")
                print(render_board_two_jump(state, env_config, first_action, second_action,
                                           f"Failed P{player} 2-jump sample"))
        except Exception as e:
            failed += 1
            print(f"  Sample {i+1}: {Colors.ERROR}✗ ERROR{Colors.RESET} - {e}")

    pct = 100 * passed / num_samples if num_samples > 0 else 0
    color = Colors.SUCCESS if failed == 0 else Colors.ERROR
    print(f"  {color}Results: {passed}/{num_samples} passed ({pct:.1f}%){Colors.RESET}")

    return passed, failed


def show_two_jump_samples(rows: int, cols: int, num_samples: int = 4):
    """Display sample 2-jump curriculum states."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(777)

    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.HEADER}  Sample 2-JUMP Curriculum States ({rows}x{cols} board){Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if i % 2 == 0 else 2

        state, first_action, second_action = generate_two_move_win_state(
            state_rng, env_config, player=player
        )

        is_valid, message = validate_two_jump_win(
            state, env_config, first_action, second_action
        )
        status = f"{Colors.SUCCESS}✓ {message}{Colors.RESET}" if is_valid else f"{Colors.ERROR}✗ {message}{Colors.RESET}"

        title = f"2-Jump Sample {i+1}: P{player} to win | {status}"
        print(render_board_two_jump(state, env_config, first_action, second_action, title))
        print()


def render_board_n_jump(
    state: PhutballState,
    env_config: EnvConfig,
    actions: List[int],
    title: str = None,
) -> str:
    """
    Render an N-jump winning state with all landing positions highlighted.
    Uses circled numbers: ①②③④ for jump sequence.
    """
    rows, cols = env_config.rows, env_config.cols
    board = np.array(state.board)
    ball_pos = (int(state.ball_pos[0]), int(state.ball_pos[1]))
    total_positions = rows * cols

    # Circled number characters for jump sequence
    jump_chars = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"]
    jump_colors = [
        Colors.BRIGHT_YELLOW,  # Jump 1
        Colors.BRIGHT_GREEN,   # Jump 2
        Colors.BRIGHT_BLUE,    # Jump 3
        Colors.WINNING_PATH,   # Jump 4 (final - cyan)
    ]

    # Calculate landing positions for each action
    landing_positions = []
    for action in actions:
        if action >= total_positions and action < 2 * total_positions:
            idx = action - total_positions
            landing_positions.append((idx // cols, idx % cols))
        else:
            landing_positions.append(None)

    lines = []

    if title:
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}  {title}{Colors.RESET}")
        lines.append(f"{Colors.HEADER}{'═' * (cols * 2 + 6)}{Colors.RESET}")

    # Column headers
    col_header = "    "
    for c in range(cols):
        col_header += f"{Colors.DIM}{c % 10} {Colors.RESET}"
    lines.append(col_header)

    for r in range(rows):
        row_str = f"{Colors.DIM}{r:2d}{Colors.RESET}  "

        for c in range(cols):
            tile = board[r, c]
            is_p2_endzone = r <= 1
            is_p1_endzone = r >= rows - 2
            is_ball = (r, c) == ball_pos

            # Check if this is a landing position
            jump_idx = None
            for idx, pos in enumerate(landing_positions):
                if pos and (r, c) == pos:
                    jump_idx = idx
                    break

            if tile == BALL or is_ball:
                char = "●"
                color = Colors.PHUTBALL
            elif tile == MAN:
                char = "○"
                color = Colors.WHITE + Colors.BOLD
            elif jump_idx is not None:
                char = jump_chars[min(jump_idx, len(jump_chars) - 1)]
                color = jump_colors[min(jump_idx, len(jump_colors) - 1)]
            elif is_p2_endzone:
                char = "░"
                color = Colors.ENDZONE_P2
            elif is_p1_endzone:
                char = "░"
                color = Colors.ENDZONE_P1
            else:
                char = "·"
                color = Colors.DIM

            row_str += f"{color}{char}{Colors.RESET} "

        row_str += f" {Colors.DIM}{r:2d}{Colors.RESET}"
        lines.append(row_str)

    player = int(state.current_player)
    player_color = Colors.ENDZONE_P1 if player == 1 else Colors.ENDZONE_P2
    lines.append("")
    lines.append(f"  {Colors.INFO}Player to move: {player_color}P{player}{Colors.RESET}")
    lines.append(f"  {Colors.INFO}Ball position: ({ball_pos[0]}, {ball_pos[1]}){Colors.RESET}")

    for idx, pos in enumerate(landing_positions):
        if pos:
            color = jump_colors[min(idx, len(jump_colors) - 1)]
            is_final = idx == len(actions) - 1
            suffix = " - WINS" if is_final else ""
            lines.append(f"  {color}Jump {idx+1} lands at: ({pos[0]}, {pos[1]}){suffix}{Colors.RESET}")

    return "\n".join(lines)


def validate_n_jump_win(
    state: PhutballState,
    env_config: EnvConfig,
    actions: List[int],
) -> Tuple[bool, str]:
    """
    Validate that executing the sequence of actions wins the game.
    """
    rows, cols = env_config.rows, env_config.cols
    total_positions = rows * cols

    if len(actions) == 0:
        return False, "No actions provided"

    # Check all actions are jumps
    for i, action in enumerate(actions):
        if action < total_positions or action >= 2 * total_positions:
            return False, f"Action {i+1} is not a jump"

    current_state = state
    for i, action in enumerate(actions):
        # Check action is legal
        legal = get_legal_actions(current_state, env_config)
        if not legal[action]:
            return False, f"Action {i+1} is not legal"

        # Execute jump
        current_state = step(current_state, jnp.array(action, dtype=jnp.int32), env_config)

        # Only the last jump should terminate
        if i < len(actions) - 1:
            if current_state.terminated:
                return False, f"Game terminated after jump {i+1} (should need {len(actions)})"
        else:
            if not current_state.terminated:
                return False, f"Game did not terminate after final jump ({len(actions)})"

    expected_winner = int(state.current_player)
    actual_winner = int(current_state.winner)

    if actual_winner != expected_winner:
        return False, f"Wrong winner: expected P{expected_winner}, got P{actual_winner}"

    return True, f"Valid! P{expected_winner} wins in {len(actions)} jumps"


def test_n_jump_board_size(rows: int, cols: int, num_jumps: int, num_samples: int = 10) -> Tuple[int, int]:
    """Test N-jump curriculum generation for a specific board size."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(42 + num_jumps * 100)

    passed = 0
    failed = 0

    print(f"\n{Colors.HEADER}Testing {num_jumps}-JUMP {rows}x{cols} board ({num_samples} samples){Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if int(jax.random.randint(player_rng, (), 0, 2)) == 0 else 2

        try:
            state, actions = generate_n_move_win_state(
                state_rng, env_config, num_jumps=num_jumps, player=player
            )

            is_valid, message = validate_n_jump_win(state, env_config, actions)

            if is_valid:
                passed += 1
            else:
                failed += 1
                print(f"  Sample {i+1}: {Colors.ERROR}✗ FAIL{Colors.RESET} - {message}")
                print(render_board_n_jump(state, env_config, actions,
                                         f"Failed P{player} {num_jumps}-jump sample"))
        except Exception as e:
            failed += 1
            print(f"  Sample {i+1}: {Colors.ERROR}✗ ERROR{Colors.RESET} - {e}")

    pct = 100 * passed / num_samples if num_samples > 0 else 0
    color = Colors.SUCCESS if failed == 0 else Colors.ERROR
    print(f"  {color}Results: {passed}/{num_samples} passed ({pct:.1f}%){Colors.RESET}")

    return passed, failed


def show_n_jump_samples(rows: int, cols: int, num_jumps: int, num_samples: int = 2):
    """Display sample N-jump curriculum states."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(777 + num_jumps * 100)

    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.HEADER}  Sample {num_jumps}-JUMP Curriculum States ({rows}x{cols} board){Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if i % 2 == 0 else 2

        state, actions = generate_n_move_win_state(
            state_rng, env_config, num_jumps=num_jumps, player=player
        )

        is_valid, message = validate_n_jump_win(state, env_config, actions)
        status = f"{Colors.SUCCESS}✓ {message}{Colors.RESET}" if is_valid else f"{Colors.ERROR}✗ {message}{Colors.RESET}"

        title = f"{num_jumps}-Jump Sample {i+1}: P{player} to win | {status}"
        print(render_board_n_jump(state, env_config, actions, title))
        print()


def show_sample_states(rows: int, cols: int, num_samples: int = 4):
    """Display a few sample curriculum states for visual inspection."""
    env_config = EnvConfig(rows=rows, cols=cols)
    rng = jax.random.PRNGKey(999)

    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.HEADER}  Sample Curriculum States ({rows}x{cols} board){Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")

    for i in range(num_samples):
        rng, state_rng, player_rng = jax.random.split(rng, 3)
        player = 1 if i % 2 == 0 else 2

        state, winning_action = generate_one_move_win_state(
            state_rng, env_config, player=player,
            min_jump_len=1, max_jump_len=6,
        )

        is_valid, message = validate_winning_action(state, env_config, winning_action)
        status = f"{Colors.SUCCESS}✓ {message}{Colors.RESET}" if is_valid else f"{Colors.ERROR}✗ {message}{Colors.RESET}"

        title = f"Sample {i+1}: P{player} to win | {status}"
        print(render_board_colored(state, env_config, winning_action, title))
        print()


def run_all_tests():
    """Run comprehensive test suite."""
    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.HEADER}  CURRICULUM LEARNING TEST SUITE{Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")

    total_passed = 0
    total_failed = 0

    # Test various board sizes for 1-jump
    board_sizes = [
        (7, 7),    # Tiny
        (9, 9),    # Small
        (11, 11),  # Medium square
        (15, 11),  # Medium rectangular
        (21, 15),  # Standard
        (31, 21),  # Large
    ]

    print(f"\n{Colors.HEADER}=== 1-JUMP TESTS ==={Colors.RESET}")
    for rows, cols in board_sizes:
        p, f = test_single_board_size(rows, cols, num_samples=20)
        total_passed += p
        total_failed += f

    # Test N-jump generation using the generic generator
    n_jump_sizes = [
        (15, 11),  # Medium rectangular
        (21, 15),  # Standard
    ]

    for num_jumps in [2, 3, 4]:
        print(f"\n{Colors.HEADER}=== {num_jumps}-JUMP TESTS ==={Colors.RESET}")
        for rows, cols in n_jump_sizes:
            p, f = test_n_jump_board_size(rows, cols, num_jumps=num_jumps, num_samples=20)
            total_passed += p
            total_failed += f

    # Test batch generation (includes 1,2,3,4-jump with configured distribution)
    print(f"\n{Colors.HEADER}=== BATCH GENERATION TESTS ==={Colors.RESET}")
    p, f = test_curriculum_batch(21, 15, batch_size=32)
    total_passed += p
    total_failed += f

    # Final summary
    print(f"\n{Colors.HEADER}{'═' * 60}{Colors.RESET}")
    if total_failed == 0:
        print(f"{Colors.SUCCESS}  ALL TESTS PASSED: {total_passed} checks{Colors.RESET}")
    else:
        print(f"{Colors.ERROR}  TESTS COMPLETE: {total_passed} passed, {total_failed} failed{Colors.RESET}")
    print(f"{Colors.HEADER}{'═' * 60}{Colors.RESET}")

    # Show sample visualizations
    show_sample_states(15, 11, num_samples=2)
    for num_jumps in [2, 3, 4]:
        show_n_jump_samples(21, 15, num_jumps=num_jumps, num_samples=1)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
