"""
JAX-native Phutball Environment

This is a functional, JIT-compilable implementation of Phutball.
All operations are pure functions that take state in and return new state out.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Any

# Type alias for JAX arrays (chex.Array equivalent)
Array = Any  # Would be jax.Array but keeping it simple

# ============================================================================
# Constants (matching your PyTorch env)
# ============================================================================

EMPTY = 0
BALL = -1
MAN = 1
END_HI = 2    # Row 0 - beyond goal for P2
END_LO = -2   # Row -1 - beyond goal for P1

# ============================================================================
# State Definition
# ============================================================================

class PhutballState(NamedTuple):
    """
    Complete game state as a JAX-compatible NamedTuple.
    All fields are JAX arrays for JIT compatibility.
    """
    board: Array           # (rows, cols) int32 - the game board
    ball_pos: Array        # (2,) int32 - [row, col] of ball
    current_player: Array  # () int32 - 1 or 2
    is_jumping: Array      # () bool - are we mid-jump-sequence?
    terminated: Array      # () bool - is game over?
    winner: Array          # () int32 - 0=none, 1=P1 wins, 2=P2 wins
    num_turns: Array       # () int32 - turn counter

# ============================================================================
# Environment Configuration
# ============================================================================

class EnvConfig(NamedTuple):
    """Static environment configuration (not part of state)."""
    rows: int = 21
    cols: int = 15
    max_turns: int = 7200

# ============================================================================
# Core Functions
# ============================================================================

def make_initial_board(config: EnvConfig) -> Array:
    """Create the initial board with end zones and ball in center."""
    rows, cols = config.rows, config.cols
    
    # Start with empty board
    board = jnp.zeros((rows, cols), dtype=jnp.int32)
    
    # Set end zones
    board = board.at[0, :].set(END_HI)           # Row 0: beyond goal (P2's target side)
    board = board.at[rows - 1, :].set(END_LO)    # Last row: beyond goal (P1's target side)
    
    # Place ball in center
    center_row = rows // 2
    center_col = cols // 2
    board = board.at[center_row, center_col].set(BALL)
    
    return board


def reset(config: EnvConfig) -> PhutballState:
    """Initialize a fresh game state."""
    board = make_initial_board(config)
    center_row = config.rows // 2
    center_col = config.cols // 2
    
    return PhutballState(
        board=board,
        ball_pos=jnp.array([center_row, center_col], dtype=jnp.int32),
        current_player=jnp.array(1, dtype=jnp.int32),
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
        num_turns=jnp.array(0, dtype=jnp.int32),
    )


def reset_jit(config: EnvConfig) -> PhutballState:
    """JIT-compiled reset. Config must be static."""
    return jax.jit(reset, static_argnums=(0,))(config)


# ============================================================================
# Action Space Helpers
# ============================================================================

def get_action_space_size(config: EnvConfig) -> int:
    """Total number of actions: placements + jumps + halt."""
    total_positions = config.rows * config.cols
    return 2 * total_positions + 1


def action_to_position(action: Array, config: EnvConfig) -> Array:
    """
    Convert action index to (row, col) position.
    
    Actions 0 to rows*cols-1: placement at that position
    Actions rows*cols to 2*rows*cols-1: jump to that position
    Action 2*rows*cols: halt
    """
    total_positions = config.rows * config.cols
    pos_index = lax.cond(
        action < total_positions,
        lambda: action,
        lambda: action - total_positions
    )
    row = pos_index // config.cols
    col = pos_index % config.cols
    return jnp.array([row, col], dtype=jnp.int32)


def is_placement_action(action: Array, config: EnvConfig) -> Array:
    """Check if action is a placement."""
    total_positions = config.rows * config.cols
    return action < total_positions


def is_jump_action(action: Array, config: EnvConfig) -> Array:
    """Check if action is a jump."""
    total_positions = config.rows * config.cols
    return (action >= total_positions) & (action < 2 * total_positions)


def is_halt_action(action: Array, config: EnvConfig) -> Array:
    """Check if action is halt."""
    total_positions = config.rows * config.cols
    return action == 2 * total_positions


# ============================================================================
# Board Queries
# ============================================================================

def get_tile(board: Array, row: Array, col: Array) -> Array:
    """Get tile value at position. Returns 0 if out of bounds."""
    rows, cols = board.shape
    in_bounds = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return lax.cond(
        in_bounds,
        lambda: board[row, col],
        lambda: jnp.array(0, dtype=jnp.int32)
    )


def is_empty(board: Array, row: Array, col: Array) -> Array:
    """Check if position is empty (can place a man there)."""
    tile = get_tile(board, row, col)
    return tile == EMPTY


def is_man(board: Array, row: Array, col: Array) -> Array:
    """Check if position has a man."""
    tile = get_tile(board, row, col)
    return tile == MAN


# ============================================================================
# Win Condition Check
# ============================================================================

def check_winner(ball_pos: Array, config: EnvConfig) -> Array:
    """
    Check if the ball has reached a goal line.
    
    Player 1 wins by getting ball to row >= rows-2 (row 19 for 21-row board)
    Player 2 wins by getting ball to row <= 1
    
    Returns: 0 (no winner), 1 (P1 wins), 2 (P2 wins)
    """
    ball_row = ball_pos[0]
    
    # P2 wins if ball reaches row 1 or beyond (row 0)
    p2_wins = ball_row <= 1
    
    # P1 wins if ball reaches row rows-2 or beyond (row rows-1)
    p1_wins = ball_row >= config.rows - 2
    
    winner = lax.cond(
        p2_wins,
        lambda: jnp.array(2, dtype=jnp.int32),
        lambda: lax.cond(
            p1_wins,
            lambda: jnp.array(1, dtype=jnp.int32),
            lambda: jnp.array(0, dtype=jnp.int32)
        )
    )
    return winner


# ============================================================================
# Legal Move Calculation
# ============================================================================

def get_legal_placements(state: PhutballState, config: EnvConfig) -> Array:
    """
    Get mask of legal placement positions.
    
    Men can be placed anywhere except:
    - Row 0 (END_HI) and row rows-1 (END_LO) - the "beyond" rows
    - Where the ball is
    - Where a man already is
    
    Returns: (rows * cols,) bool array
    """
    board = state.board
    rows, cols = config.rows, config.cols
    
    # Create mask for each position
    # Legal if: not end zone AND not ball AND not man
    row_indices = jnp.arange(rows)[:, None]  # (rows, 1)
    
    # End zones are row 0 and row rows-1
    not_end_zone = (row_indices > 0) & (row_indices < rows - 1)  # (rows, 1) broadcasts to (rows, cols)
    
    # Check tile contents
    not_ball = board != BALL
    not_man = board != MAN
    
    # Combine conditions
    legal_2d = not_end_zone & not_ball & not_man  # (rows, cols)
    
    # Flatten to 1D
    return legal_2d.flatten().astype(jnp.int8)


def get_legal_jumps(state: PhutballState, config: EnvConfig) -> Array:
    """
    Get mask of legal jump landing positions (vectorized version).
    
    A jump is legal if:
    - There's at least one man between ball and landing position
    - All tiles between ball and landing are men (continuous line of men)
    - Landing position is empty, END_HI, or END_LO (not ball, not man)
    - Jump is in one of 8 directions (horizontal, vertical, diagonal)
    
    Returns: (rows * cols,) int8 array
    """
    board = state.board
    ball_row, ball_col = state.ball_pos[0], state.ball_pos[1]
    rows, cols = config.rows, config.cols
    max_dist = max(rows, cols)
    
    # 8 directions: (dr, dc)
    directions = jnp.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ], dtype=jnp.int32)
    
    # For each direction and distance, check if that landing spot is valid
    # distances: 2 to max_dist (need at least 1 man to jump over)
    distances = jnp.arange(2, max_dist + 1, dtype=jnp.int32)
    
    def check_one_direction_distance(dir_dist):
        """Check if jumping (dir, dist) is valid."""
        direction_idx = dir_dist[0].astype(jnp.int32)
        dist = dir_dist[1].astype(jnp.int32)
        dr = directions[direction_idx, 0]
        dc = directions[direction_idx, 1]
        
        land_row = ball_row + dist * dr
        land_col = ball_col + dist * dc
        
        # Check bounds
        in_bounds = (land_row >= 0) & (land_row < rows) & (land_col >= 0) & (land_col < cols)
        
        # Check landing tile - must not be ball or man
        # Use jnp.where with safe indexing
        safe_land_row = jnp.clip(land_row, 0, rows - 1)
        safe_land_col = jnp.clip(land_col, 0, cols - 1)
        land_tile = board[safe_land_row, safe_land_col]
        landing_ok = (land_tile != BALL) & (land_tile != MAN)
        
        # Check all intermediate tiles are men
        # For distance d, we need to check positions 1, 2, ..., d-1
        def check_intermediate(i):
            i = i.astype(jnp.int32)
            check_row = ball_row + i * dr
            check_col = ball_col + i * dc
            safe_row = jnp.clip(check_row, 0, rows - 1)
            safe_col = jnp.clip(check_col, 0, cols - 1)
            is_in_range = (i >= 1) & (i < dist)
            tile = board[safe_row, safe_col]
            return jnp.where(is_in_range, tile == MAN, True)
        
        intermediate_checks = jax.vmap(check_intermediate)(jnp.arange(max_dist, dtype=jnp.int32))
        all_intermediate_men = jnp.all(intermediate_checks)
        
        # Valid jump if: in bounds, landing ok, all intermediate are men
        valid = in_bounds & landing_ok & all_intermediate_men
        
        # Return (row*cols + col, valid) for this position
        pos_idx = land_row * cols + land_col
        safe_pos_idx = jnp.where(in_bounds, pos_idx, jnp.int32(0))
        
        return jnp.array([safe_pos_idx, valid.astype(jnp.int32)], dtype=jnp.int32)
    
    # Create all (direction, distance) pairs
    dir_indices = jnp.arange(8, dtype=jnp.int32)
    dist_indices = jnp.arange(len(distances), dtype=jnp.int32)
    
    # Cartesian product: (8 * (max_dist-1), 2) array of [dir_idx, dist]
    dir_grid, dist_grid = jnp.meshgrid(dir_indices, dist_indices, indexing='ij')
    dir_dist_pairs = jnp.stack([
        dir_grid.flatten(),
        distances[dist_grid.flatten()]
    ], axis=1).astype(jnp.int32)
    
    # Check all pairs in parallel
    results = jax.vmap(check_one_direction_distance)(dir_dist_pairs)
    # results: (N, 2) where each row is [pos_idx, valid]
    
    positions = results[:, 0]
    valid_flags = results[:, 1]
    
    # Create output mask using scatter
    legal_mask = jnp.zeros(rows * cols, dtype=jnp.int8)
    
    # For each valid position, set mask to 1
    # Use jnp.where to conditionally update
    def update_mask(i, mask):
        i = i.astype(jnp.int32)
        pos = positions[i]
        is_valid = valid_flags[i]
        safe_pos = jnp.clip(pos, 0, rows * cols - 1).astype(jnp.int32)
        return jnp.where(
            (is_valid == 1) & (pos >= 0) & (pos < rows * cols),
            mask.at[safe_pos].set(jnp.int8(1)),
            mask
        )
    
    num_pairs = 8 * (max_dist - 1)  # Number of direction-distance pairs
    legal_mask = lax.fori_loop(0, num_pairs, update_mask, legal_mask)
    
    return legal_mask


def get_legal_actions(state: PhutballState, config: EnvConfig) -> Array:
    """
    Get full legal action mask.
    
    Actions: [placements (rows*cols) | jumps (rows*cols) | halt (1)]
    
    During jump sequence: only jumps and halt are legal
    Otherwise: placements and jumps are legal, halt is not
    """
    rows, cols = config.rows, config.cols
    total_positions = rows * cols
    
    placement_mask = get_legal_placements(state, config)
    jump_mask = get_legal_jumps(state, config)
    
    # During jump sequence, placements are not allowed
    placement_mask = lax.cond(
        state.is_jumping,
        lambda: jnp.zeros(total_positions, dtype=jnp.int8),
        lambda: placement_mask
    )
    
    # Halt is only legal during jump sequence
    halt_legal = state.is_jumping.astype(jnp.int8)
    
    # Combine: [placements | jumps | halt]
    return jnp.concatenate([placement_mask, jump_mask, jnp.array([halt_legal], dtype=jnp.int8)])


# ============================================================================
# Step Function - Execute Actions
# ============================================================================

def execute_placement(state: PhutballState, position: Array, config: EnvConfig) -> PhutballState:
    """Place a man at the given position and end turn."""
    row, col = position[0], position[1]
    new_board = state.board.at[row, col].set(MAN)
    
    # Switch player and increment turn
    new_player = 3 - state.current_player  # 1 -> 2, 2 -> 1
    new_turns = state.num_turns + 1
    
    return state._replace(
        board=new_board,
        current_player=new_player,
        num_turns=new_turns
    )


def calculate_jumped_men(ball_pos: Array, landing_pos: Array, board: Array) -> Array:
    """
    Calculate positions of men that would be jumped.
    Returns array of (row, col) pairs, padded with -1.
    """
    ball_row, ball_col = ball_pos[0], ball_pos[1]
    land_row, land_col = landing_pos[0], landing_pos[1]
    
    dr = jnp.sign(land_row - ball_row)
    dc = jnp.sign(land_col - ball_col)
    dist = jnp.maximum(jnp.abs(land_row - ball_row), jnp.abs(land_col - ball_col))
    
    # Collect jumped positions
    max_jump_dist = 21  # Maximum board dimension
    
    def get_jumped_pos(i):
        """Get position i steps from ball, or (-1, -1) if out of range."""
        step = i + 1  # Start from 1
        row = ball_row + step * dr
        col = ball_col + step * dc
        is_jumped = step < dist
        return lax.cond(
            is_jumped,
            lambda: jnp.array([row, col], dtype=jnp.int32),
            lambda: jnp.array([-1, -1], dtype=jnp.int32)
        )
    
    jumped = jax.vmap(get_jumped_pos)(jnp.arange(max_jump_dist))
    return jumped


def execute_jump(state: PhutballState, landing_pos: Array, config: EnvConfig) -> PhutballState:
    """Execute a jump to landing position, removing jumped men."""
    ball_pos = state.ball_pos
    board = state.board
    
    # Calculate which men are jumped
    jumped_positions = calculate_jumped_men(ball_pos, landing_pos, board)
    
    # Remove ball from current position
    new_board = board.at[ball_pos[0], ball_pos[1]].set(EMPTY)
    
    # Remove jumped men
    def remove_man(board, pos):
        row, col = pos[0], pos[1]
        return lax.cond(
            (row >= 0) & (col >= 0),
            lambda: board.at[row, col].set(EMPTY),
            lambda: board
        )
    
    new_board = lax.fori_loop(
        0, len(jumped_positions),
        lambda i, b: remove_man(b, jumped_positions[i]),
        new_board
    )
    
    # Place ball at landing position
    new_board = new_board.at[landing_pos[0], landing_pos[1]].set(BALL)
    
    # Check for win
    winner = check_winner(landing_pos, config)
    terminated = winner > 0
    
    # Update state - now in jump sequence, don't switch player yet
    return state._replace(
        board=new_board,
        ball_pos=landing_pos,
        is_jumping=jnp.array(True, dtype=jnp.bool_),
        terminated=terminated,
        winner=winner
    )


def execute_halt(state: PhutballState, config: EnvConfig) -> PhutballState:
    """End jump sequence and switch player."""
    new_player = 3 - state.current_player
    new_turns = state.num_turns + 1
    
    return state._replace(
        current_player=new_player,
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        num_turns=new_turns
    )


def step(state: PhutballState, action: Array, config: EnvConfig) -> PhutballState:
    """
    Execute one action and return new state.
    
    Action encoding:
    - 0 to rows*cols-1: place man at that position
    - rows*cols to 2*rows*cols-1: jump to that position  
    - 2*rows*cols: halt jump sequence
    """
    total_positions = config.rows * config.cols
    
    # Determine action type
    is_placement = action < total_positions
    is_jump = (action >= total_positions) & (action < 2 * total_positions)
    is_halt = action == 2 * total_positions
    
    # Get position for placement/jump
    position = action_to_position(action, config)
    
    # Execute the appropriate action
    # Using nested lax.cond to handle the three cases
    new_state = lax.cond(
        is_placement,
        lambda: execute_placement(state, position, config),
        lambda: lax.cond(
            is_jump,
            lambda: execute_jump(state, position, config),
            lambda: execute_halt(state, config)  # is_halt
        )
    )
    
    return new_state


# ============================================================================
# Pretty Printing (for debugging - not JIT compatible)
# ============================================================================

def render_board(state: PhutballState) -> str:
    """Render board as string for debugging."""
    board = state.board
    rows, cols = board.shape
    
    symbols = {
        EMPTY: '.',
        BALL: 'O',
        MAN: 'x',
        END_HI: '+',
        END_LO: '-',
    }
    
    lines = []
    lines.append(f"Turn: {int(state.num_turns)}, Player: {int(state.current_player)}")
    lines.append(f"Ball: ({int(state.ball_pos[0])}, {int(state.ball_pos[1])})")
    lines.append(f"Jumping: {bool(state.is_jumping)}, Terminated: {bool(state.terminated)}, Winner: {int(state.winner)}")
    lines.append("")
    
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            tile = int(board[r, c])
            row_str += symbols.get(tile, '?')
        lines.append(f"{r:2d} {row_str}")
    
    return "\n".join(lines)


# ============================================================================
# Test Utilities
# ============================================================================

def test_reset():
    """Test that reset produces valid initial state."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    # Check board shape
    assert state.board.shape == (21, 15), f"Board shape: {state.board.shape}"
    
    # Check ball position
    assert int(state.ball_pos[0]) == 10, f"Ball row: {state.ball_pos[0]}"
    assert int(state.ball_pos[1]) == 7, f"Ball col: {state.ball_pos[1]}"
    
    # Check ball is on board
    assert int(state.board[10, 7]) == BALL, f"Ball on board: {state.board[10, 7]}"
    
    # Check end zones
    assert jnp.all(state.board[0, :] == END_HI), "Top row should be END_HI"
    assert jnp.all(state.board[20, :] == END_LO), "Bottom row should be END_LO"
    
    # Check middle rows are empty (except ball)
    for r in range(1, 20):
        for c in range(15):
            expected = BALL if (r == 10 and c == 7) else EMPTY
            actual = int(state.board[r, c])
            assert actual == expected, f"Position ({r},{c}): expected {expected}, got {actual}"
    
    # Check initial player
    assert int(state.current_player) == 1, f"Initial player: {state.current_player}"
    
    # Check flags
    assert not bool(state.is_jumping), "Should not be jumping initially"
    assert not bool(state.terminated), "Should not be terminated initially"
    assert int(state.winner) == 0, "Should have no winner initially"
    
    print("✓ test_reset passed")
    return state


def test_action_encoding():
    """Test action space encoding/decoding."""
    config = EnvConfig(rows=21, cols=15)
    total_pos = 21 * 15  # 315
    
    # Test placement action
    action = jnp.array(0, dtype=jnp.int32)
    assert bool(is_placement_action(action, config)), "Action 0 should be placement"
    pos = action_to_position(action, config)
    assert int(pos[0]) == 0 and int(pos[1]) == 0, f"Action 0 -> position: {pos}"
    
    # Test another placement
    action = jnp.array(total_pos - 1, dtype=jnp.int32)  # 314
    assert bool(is_placement_action(action, config)), "Action 314 should be placement"
    pos = action_to_position(action, config)
    assert int(pos[0]) == 20 and int(pos[1]) == 14, f"Action 314 -> position: {pos}"
    
    # Test jump action
    action = jnp.array(total_pos, dtype=jnp.int32)  # 315
    assert bool(is_jump_action(action, config)), "Action 315 should be jump"
    pos = action_to_position(action, config)
    assert int(pos[0]) == 0 and int(pos[1]) == 0, f"Action 315 -> position: {pos}"
    
    # Test halt action
    action = jnp.array(2 * total_pos, dtype=jnp.int32)  # 630
    assert bool(is_halt_action(action, config)), "Action 630 should be halt"
    
    print("✓ test_action_encoding passed")


def test_winner_check():
    """Test win condition detection."""
    config = EnvConfig(rows=21, cols=15)
    
    # Ball in center - no winner
    ball_pos = jnp.array([10, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 0, "Center should have no winner"
    
    # Ball at row 1 - P2 wins
    ball_pos = jnp.array([1, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 2, "Row 1 should be P2 win"
    
    # Ball at row 0 - P2 wins
    ball_pos = jnp.array([0, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 2, "Row 0 should be P2 win"
    
    # Ball at row 19 - P1 wins
    ball_pos = jnp.array([19, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 1, "Row 19 should be P1 win"
    
    # Ball at row 20 - P1 wins
    ball_pos = jnp.array([20, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 1, "Row 20 should be P1 win"
    
    # Ball at row 2 - no winner yet
    ball_pos = jnp.array([2, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 0, "Row 2 should have no winner"
    
    # Ball at row 18 - no winner yet
    ball_pos = jnp.array([18, 7], dtype=jnp.int32)
    assert int(check_winner(ball_pos, config)) == 0, "Row 18 should have no winner"
    
    print("✓ test_winner_check passed")


def test_legal_placements():
    """Test legal placement mask."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    placement_mask = get_legal_placements(state, config)
    
    # Check shape
    assert placement_mask.shape == (21 * 15,), f"Shape: {placement_mask.shape}"
    
    # Row 0 should be all illegal (END_HI)
    row0_mask = placement_mask[:15]
    assert jnp.sum(row0_mask) == 0, f"Row 0 should have no legal placements: {jnp.sum(row0_mask)}"
    
    # Row 20 should be all illegal (END_LO)
    row20_mask = placement_mask[20*15:21*15]
    assert jnp.sum(row20_mask) == 0, f"Row 20 should have no legal placements: {jnp.sum(row20_mask)}"
    
    # Ball position (10, 7) should be illegal
    ball_idx = 10 * 15 + 7
    assert int(placement_mask[ball_idx]) == 0, f"Ball position should be illegal: {placement_mask[ball_idx]}"
    
    # Total legal placements should be (21-2) * 15 - 1 = 19 * 15 - 1 = 284
    total_legal = int(jnp.sum(placement_mask))
    expected = 19 * 15 - 1  # Rows 1-19 minus ball position
    assert total_legal == expected, f"Expected {expected} legal placements, got {total_legal}"
    
    print("✓ test_legal_placements passed")


def test_legal_jumps_empty_board():
    """Test that no jumps are legal on empty board."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    jump_mask = get_legal_jumps(state, config)
    
    # No men on board, so no jumps should be legal
    total_legal = int(jnp.sum(jump_mask))
    assert total_legal == 0, f"Expected 0 legal jumps on empty board, got {total_legal}"
    
    print("✓ test_legal_jumps_empty_board passed")


def test_placement_and_jump():
    """Test placing men and then jumping."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    # Place a man adjacent to ball (ball is at 10, 7)
    # Place at (9, 7) - directly above ball
    place_action = 9 * 15 + 7  # Row 9, col 7
    state = step(state, jnp.array(place_action, dtype=jnp.int32), config)
    
    # Check man was placed
    assert int(state.board[9, 7]) == MAN, f"Man should be at (9,7): {state.board[9,7]}"
    
    # Check player switched
    assert int(state.current_player) == 2, f"Player should be 2: {state.current_player}"
    
    # Now check legal jumps - should be able to jump over the man
    jump_mask = get_legal_jumps(state, config)
    
    # Landing position (8, 7) should be legal
    landing_idx = 8 * 15 + 7
    assert int(jump_mask[landing_idx]) == 1, f"Jump to (8,7) should be legal: {jump_mask[landing_idx]}"
    
    print("✓ test_placement_and_jump passed")


def test_jump_execution():
    """Test executing a jump."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    # Manually place a man at (9, 7)
    state = state._replace(board=state.board.at[9, 7].set(MAN))
    
    # Execute jump to (8, 7)
    total_positions = config.rows * config.cols
    jump_action = total_positions + 8 * 15 + 7  # Jump to (8, 7)
    state = step(state, jnp.array(jump_action, dtype=jnp.int32), config)
    
    # Check ball moved
    assert int(state.ball_pos[0]) == 8, f"Ball row should be 8: {state.ball_pos[0]}"
    assert int(state.ball_pos[1]) == 7, f"Ball col should be 7: {state.ball_pos[1]}"
    
    # Check man was removed
    assert int(state.board[9, 7]) == EMPTY, f"Man at (9,7) should be removed: {state.board[9,7]}"
    
    # Check ball on board
    assert int(state.board[8, 7]) == BALL, f"Ball should be at (8,7): {state.board[8,7]}"
    
    # Check old ball position is empty
    assert int(state.board[10, 7]) == EMPTY, f"Old ball position should be empty: {state.board[10,7]}"
    
    # Check is_jumping flag
    assert bool(state.is_jumping), "Should be in jump sequence"
    
    print("✓ test_jump_execution passed")


def test_halt():
    """Test halting a jump sequence."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    # Set up jump sequence manually
    state = state._replace(
        board=state.board.at[9, 7].set(MAN),
        is_jumping=jnp.array(True, dtype=jnp.bool_)
    )
    
    # Execute halt
    halt_action = 2 * config.rows * config.cols
    state = step(state, jnp.array(halt_action, dtype=jnp.int32), config)
    
    # Check player switched
    assert int(state.current_player) == 2, f"Player should switch to 2: {state.current_player}"
    
    # Check no longer jumping
    assert not bool(state.is_jumping), "Should not be in jump sequence"
    
    print("✓ test_halt passed")


def test_legal_actions_full():
    """Test full legal actions mask."""
    config = EnvConfig(rows=21, cols=15)
    state = reset(config)
    
    legal = get_legal_actions(state, config)
    
    # Check shape: placements + jumps + halt
    expected_size = 2 * 21 * 15 + 1  # 631
    assert legal.shape == (expected_size,), f"Shape: {legal.shape}"
    
    # Halt should be illegal (not in jump sequence)
    assert int(legal[-1]) == 0, f"Halt should be illegal: {legal[-1]}"
    
    # Some placements should be legal
    placements_legal = jnp.sum(legal[:21*15])
    assert int(placements_legal) == 284, f"Expected 284 legal placements: {placements_legal}"
    
    # No jumps should be legal (empty board)
    jumps_legal = jnp.sum(legal[21*15:-1])
    assert int(jumps_legal) == 0, f"Expected 0 legal jumps: {jumps_legal}"
    
    print("✓ test_legal_actions_full passed")


if __name__ == "__main__":
    print("Running JAX Phutball environment tests...\n")
    
    state = test_reset()
    print("\nInitial board:")
    print(render_board(state))
    print()
    
    test_action_encoding()
    test_winner_check()
    test_legal_placements()
    test_legal_jumps_empty_board()
    test_placement_and_jump()
    test_jump_execution()
    test_halt()
    test_legal_actions_full()
    
    print("\n✓ All tests passed!")