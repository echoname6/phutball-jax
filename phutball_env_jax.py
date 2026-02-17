"""
JAX-native Phutball Environment with Jump Sequence Tracking

This is a functional, JIT-compilable implementation of Phutball.
All operations are pure functions that take state in and return new state out.

Includes jump_sequence and jump_sequence_length to track the path
during multi-jump turns, enabling proper neural network input encoding.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple, Any

# Type alias for JAX arrays
Array = Any

# ============================================================================
# Constants
# ============================================================================

EMPTY = 0
BALL = -1
MAN = 1
END_HI = 2    # Rows 0-1 (top end zone, P1's goal)
END_LO = -2   # Rows (rows-2) to (rows-1) (bottom end zone, P2's goal)
OUT_OF_BOUNDS = jnp.array(3, dtype=jnp.int32)

# Fixed array size for jump sequence tracking (for JIT compilation and NN input)
# Must be large enough for biggest supported board. 256 covers up to ~22x22.
# The scan in state_to_network_input uses (rows*cols)//2 for actual iteration count
MAX_JUMP_SEQUENCE_LENGTH = 256

# ============================================================================
# State Definition (with jump sequence tracking)
# ============================================================================

class PhutballState(NamedTuple):
    """
    Complete game state as a JAX-compatible NamedTuple.
    All fields are JAX arrays for JIT compatibility.
    
    - jump_sequence: (MAX_JUMP_SEQUENCE_LENGTH, 2) array of (row, col) positions
                     visited during the current jump sequence
    - jump_sequence_length: () int32 - number of valid entries in jump_sequence
                            (includes starting position, so length=1 means "at start, no jumps yet")
    """
    board: Array               # (rows, cols) int32 - the game board
    ball_pos: Array            # (2,) int32 - [row, col] of ball
    current_player: Array      # () int32 - 1 or 2
    is_jumping: Array          # () bool - are we mid-jump-sequence?
    terminated: Array          # () bool - is game over?
    winner: Array              # () int32 - 0=none, 1=P1 wins, 2=P2 wins
    num_turns: Array           # () int32 - turn counter
    # Jump sequence tracking
    jump_sequence: Array       # (MAX_JUMP_SEQUENCE_LENGTH, 2) int32 - positions visited
    jump_sequence_length: Array  # () int32 - how many positions in sequence


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

    # Set end zones (2 rows at each end)
    board = board.at[0, :].set(END_HI)
    board = board.at[1, :].set(END_HI)
    board = board.at[rows - 2, :].set(END_LO)
    board = board.at[rows - 1, :].set(END_LO)

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
    
    # Initialize empty jump sequence (all -1s indicate invalid/unused)
    jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)
    
    return PhutballState(
        board=board,
        ball_pos=jnp.array([center_row, center_col], dtype=jnp.int32),
        current_player=jnp.array(1, dtype=jnp.int32),
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
        num_turns=jnp.array(0, dtype=jnp.int32),
        jump_sequence=jump_sequence,
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )


# ============================================================================
# Action Space Helpers
# ============================================================================

def get_action_space_size(config: EnvConfig) -> int:
    """Total number of actions: placements + jumps + halt."""
    total_positions = config.rows * config.cols
    return 2 * total_positions + 1


def action_to_position(action: Array, config: EnvConfig) -> Array:
    """Convert action index to (row, col) position."""
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
    """Get tile value at position. Returns OUT_OF_BOUNDS if invalid."""
    rows, cols = board.shape
    in_bounds = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return lax.cond(
        in_bounds,
        lambda: board[row, col],
        lambda: OUT_OF_BOUNDS,
    )


def is_empty(board: Array, row: Array, col: Array) -> Array:
    """Check if position is empty."""
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

    Player 1 wins by getting ball to row <= 1 (top end zone, END_HI)
    Player 2 wins by getting ball to row >= rows-2 (bottom end zone, END_LO)
    """
    ball_row = ball_pos[0]

    p1_wins = ball_row <= 1
    p2_wins = ball_row >= config.rows - 2

    winner = lax.cond(
        p1_wins,
        lambda: jnp.array(1, dtype=jnp.int32),
        lambda: lax.cond(
            p2_wins,
            lambda: jnp.array(2, dtype=jnp.int32),
            lambda: jnp.array(0, dtype=jnp.int32)
        )
    )
    return winner


# ============================================================================
# Legal Move Calculation
# ============================================================================

def get_legal_placements(state: PhutballState, config: EnvConfig) -> Array:
    """Get mask of legal placement positions."""
    board = state.board
    rows, cols = config.rows, config.cols
    
    row_indices = jnp.arange(rows)[:, None]
    not_end_zone = (row_indices > 0) & (row_indices < rows - 1)
    not_ball = board != BALL
    not_man = board != MAN
    
    legal_2d = not_end_zone & not_ball & not_man
    return legal_2d.flatten().astype(jnp.int8)


def get_legal_jumps(state: PhutballState, config: EnvConfig) -> Array:
    """Get mask of legal jump landing positions (vectorized)."""
    board = state.board
    ball_row, ball_col = state.ball_pos[0], state.ball_pos[1]
    rows, cols = config.rows, config.cols
    max_dist = max(rows, cols)
    
    directions = jnp.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ], dtype=jnp.int32)
    
    distances = jnp.arange(2, max_dist + 1, dtype=jnp.int32)
    
    def check_one_direction_distance(dir_dist):
        direction_idx = dir_dist[0].astype(jnp.int32)
        dist = dir_dist[1].astype(jnp.int32)
        dr = directions[direction_idx, 0]
        dc = directions[direction_idx, 1]
        
        land_row = ball_row + dist * dr
        land_col = ball_col + dist * dc
        
        in_bounds = (land_row >= 0) & (land_row < rows) & (land_col >= 0) & (land_col < cols)
        
        safe_land_row = jnp.clip(land_row, 0, rows - 1)
        safe_land_col = jnp.clip(land_col, 0, cols - 1)
        land_tile = board[safe_land_row, safe_land_col]
        landing_ok = (land_tile != BALL) & (land_tile != MAN)
        
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
        
        valid = in_bounds & landing_ok & all_intermediate_men
        
        pos_idx = land_row * cols + land_col
        safe_pos_idx = jnp.where(in_bounds, pos_idx, jnp.int32(0))
        
        return jnp.array([safe_pos_idx, valid.astype(jnp.int32)], dtype=jnp.int32)
    
    dir_indices = jnp.arange(8, dtype=jnp.int32)
    dist_indices = jnp.arange(len(distances), dtype=jnp.int32)
    
    dir_grid, dist_grid = jnp.meshgrid(dir_indices, dist_indices, indexing='ij')
    dir_dist_pairs = jnp.stack([
        dir_grid.flatten(),
        distances[dist_grid.flatten()]
    ], axis=1).astype(jnp.int32)
    
    results = jax.vmap(check_one_direction_distance)(dir_dist_pairs)
    
    positions = results[:, 0]
    valid_flags = results[:, 1]
    
    legal_mask = jnp.zeros(rows * cols, dtype=jnp.int8)
    
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
    
    num_pairs = 8 * (max_dist - 1)
    legal_mask = lax.fori_loop(0, num_pairs, update_mask, legal_mask)
    
    return legal_mask


def get_legal_actions(state: PhutballState, config: EnvConfig) -> Array:
    """Get full legal action mask."""
    rows, cols = config.rows, config.cols
    total_positions = rows * cols
    
    placement_mask = get_legal_placements(state, config)
    jump_mask = get_legal_jumps(state, config)
    
    placement_mask = lax.cond(
        state.is_jumping,
        lambda: jnp.zeros(total_positions, dtype=jnp.int8),
        lambda: placement_mask
    )
    
    halt_legal = state.is_jumping.astype(jnp.int8)
    
    return jnp.concatenate([placement_mask, jump_mask, jnp.array([halt_legal], dtype=jnp.int8)])


# ============================================================================
# Step Function - Execute Actions (with jump sequence tracking)
# ============================================================================

def apply_turn_limit(state: PhutballState, config: EnvConfig) -> PhutballState:
    """Apply the max_turns limit from config."""
    if config.max_turns <= 0:
        return state

    reached_limit = state.num_turns >= config.max_turns

    new_terminated = jnp.where(
        reached_limit,
        jnp.array(True, dtype=jnp.bool_),
        state.terminated,
    )

    new_winner = jnp.where(
        reached_limit & ~state.terminated,
        jnp.array(0, dtype=jnp.int32),
        state.winner,
    )

    return state._replace(
        terminated=new_terminated,
        winner=new_winner,
    )


def execute_placement(state: PhutballState, position: Array, config: EnvConfig) -> PhutballState:
    """Place a man at the given position and end turn."""
    row, col = position[0], position[1]
    new_board = state.board.at[row, col].set(MAN)
    
    new_player = 3 - state.current_player
    new_turns = state.num_turns + 1

    # Clear jump sequence (not jumping)
    new_jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)

    new_state = state._replace(
        board=new_board,
        current_player=new_player,
        num_turns=new_turns,
        jump_sequence=new_jump_sequence,
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )

    return apply_turn_limit(new_state, config)


def calculate_jumped_men(ball_pos: Array, landing_pos: Array, board: Array) -> Array:
    """Calculate positions of men that would be jumped."""
    ball_row, ball_col = ball_pos[0], ball_pos[1]
    land_row, land_col = landing_pos[0], landing_pos[1]
    
    dr = jnp.sign(land_row - ball_row)
    dc = jnp.sign(land_col - ball_col)
    dist = jnp.maximum(jnp.abs(land_row - ball_row), jnp.abs(land_col - ball_col))
    
    rows, cols = board.shape
    max_jump_dist = max(rows, cols)
    
    def get_jumped_pos(i):
        step = i + 1
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
    
    # Update jump sequence
    # If this is the first jump, record the starting position first
    new_jump_sequence = state.jump_sequence
    new_length = state.jump_sequence_length
    
    # If not already jumping, this is the START of a sequence
    # Record the starting position (ball_pos before jump) at index 0
    def start_new_sequence():
        seq = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)
        seq = seq.at[0].set(ball_pos)  # Starting position
        seq = seq.at[1].set(landing_pos)  # After first jump
        return seq, jnp.array(2, dtype=jnp.int32)
    
    # If already jumping, append the new landing position
    # Only update array if within bounds, but don't clamp the length counter
    # This allows unlimited jumps while keeping a fixed-size array for JIT
    def continue_sequence():
        seq = lax.cond(
            new_length < MAX_JUMP_SEQUENCE_LENGTH,
            lambda s: s.at[new_length].set(landing_pos),
            lambda s: s,  # Don't update array if full, but continue tracking
            new_jump_sequence
        )
        length = new_length + 1  # No clamping - allow unlimited jumps
        return seq, length
    
    new_jump_sequence, new_length = lax.cond(
        state.is_jumping,
        continue_sequence,
        start_new_sequence
    )
    
    return state._replace(
        board=new_board,
        ball_pos=landing_pos,
        is_jumping=jnp.array(True, dtype=jnp.bool_),
        terminated=terminated,
        winner=winner,
        jump_sequence=new_jump_sequence,
        jump_sequence_length=new_length,
    )


def execute_halt(state: PhutballState, config: EnvConfig) -> PhutballState:
    """End jump sequence and switch player."""
    new_player = 3 - state.current_player
    new_turns = state.num_turns + 1

    # Clear jump sequence
    new_jump_sequence = jnp.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=jnp.int32)

    new_state = state._replace(
        current_player=new_player,
        is_jumping=jnp.array(False, dtype=jnp.bool_),
        num_turns=new_turns,
        jump_sequence=new_jump_sequence,
        jump_sequence_length=jnp.array(0, dtype=jnp.int32),
    )

    return apply_turn_limit(new_state, config)


def step(state: PhutballState, action: Array, config: EnvConfig) -> PhutballState:
    """Execute one action and return new state."""
    total_positions = config.rows * config.cols
    
    is_placement = action < total_positions
    is_jump = (action >= total_positions) & (action < 2 * total_positions)
    
    position = action_to_position(action, config)
    
    new_state = lax.cond(
        is_placement,
        lambda: execute_placement(state, position, config),
        lambda: lax.cond(
            is_jump,
            lambda: execute_jump(state, position, config),
            lambda: execute_halt(state, config)
        )
    )
    
    return new_state


# ============================================================================
# Network Input Encoding
# ============================================================================

def state_to_network_input(state: PhutballState, config: EnvConfig) -> Array:
    """
    Convert PhutballState to neural network input.

    Returns:
        Array of shape (9, rows, cols):
        - Channel 0: ball position (binary)
        - Channel 1: men positions (binary)
        - Channel 2: my-goal endzone (endzone the current player attacks toward)
        - Channel 3: opponent-goal endzone
        - Channels 4-8: jump sequence encoding

    The board is encoded as 4 binary channels instead of raw tile values.
    My-goal is the endzone the current player is trying to push the ball into:
        P1 attacks rows 0-1 (END_HI), P2 attacks rows R-2:R-1 (END_LO).
    After the 180° spatial flip for P2, my-goal always appears at the top
    with the same positive value for both players.

    Jump sequence encoding:
        - For each position in the jump sequence, we mark it in one of 5 layers
        - If a position is visited multiple times, each visit uses the next layer
        - The VALUE at that position is the step index (0, 1, 2, ...) when visited
    """
    rows, cols = config.rows, config.cols
    is_p2 = state.current_player == 2

    # Channel 0: ball position (binary)
    ball_channel = (state.board == BALL).astype(jnp.float32)

    # Channel 1: men positions (binary)
    men_channel = (state.board == MAN).astype(jnp.float32)

    # Channel 2: my-goal endzone (where current player attacks toward)
    # Channel 3: opponent-goal endzone
    # P1 attacks toward rows 0-1 (END_HI), P2 attacks toward rows R-2:R-1 (END_LO)
    # Use row indices (not board values) so endzones show even when occupied by pieces
    row_indices = jnp.arange(rows)[:, None]
    end_hi_mask = jnp.broadcast_to((row_indices <= 1), (rows, cols)).astype(jnp.float32)
    end_lo_mask = jnp.broadcast_to((row_indices >= rows - 2), (rows, cols)).astype(jnp.float32)

    my_goal = jnp.where(is_p2, end_lo_mask, end_hi_mask)
    opp_goal = jnp.where(is_p2, end_hi_mask, end_lo_mask)

    # Channels 1-5: jump sequence encoding
    num_jump_layers = 5

    def encode_jump_sequence():
        layers = jnp.zeros((num_jump_layers, rows, cols), dtype=jnp.float32)

        # Count how many times each position has been visited so far
        # This determines which layer to use
        visit_counts = jnp.zeros((rows, cols), dtype=jnp.int32)

        def process_step(carry, step_idx):
            layers, visit_counts = carry

            # Get position for this step
            pos = state.jump_sequence[step_idx]
            r, c = pos[0], pos[1]

            # Check if this step is valid (within sequence length and valid coords)
            is_valid = (step_idx < state.jump_sequence_length) & (r >= 0) & (c >= 0) & (r < rows) & (c < cols)

            # Get which layer to use (based on how many times we've visited this position)
            safe_r = jnp.clip(r, 0, rows - 1)
            safe_c = jnp.clip(c, 0, cols - 1)
            layer_idx = jnp.clip(visit_counts[safe_r, safe_c], 0, num_jump_layers - 1)

            # Update layers: set the value at (layer_idx, r, c) to step_idx
            step_value = step_idx.astype(jnp.float32)

            # Conditional update
            new_layers = jnp.where(
                is_valid,
                layers.at[layer_idx, safe_r, safe_c].set(step_value + 1.0),  # +1 so 0 means "not visited"
                layers
            )

            # Update visit count for this position
            new_visit_counts = jnp.where(
                is_valid,
                visit_counts.at[safe_r, safe_c].add(1),
                visit_counts
            )

            return (new_layers, new_visit_counts), None

        # Process steps up to max possible jumps for this board size
        # (rows * cols) // 2 is theoretical max - can't jump more men than half the board
        max_jumps = (rows * cols) // 2
        (final_layers, _), _ = lax.scan(
            process_step,
            (layers, visit_counts),
            jnp.arange(max_jumps, dtype=jnp.int32)
        )

        return final_layers

    def no_sequence():
        return jnp.zeros((num_jump_layers, rows, cols), dtype=jnp.float32)

    jump_layers = lax.cond(
        state.jump_sequence_length > 0,
        encode_jump_sequence,
        no_sequence
    )

    # Stack all channels: (9, rows, cols)
    observation = jnp.concatenate([
        ball_channel[None, :, :],    # (1, rows, cols)
        men_channel[None, :, :],     # (1, rows, cols)
        my_goal[None, :, :],         # (1, rows, cols)
        opp_goal[None, :, :],        # (1, rows, cols)
        jump_layers                   # (5, rows, cols)
    ], axis=0)

    # Perspective normalization: P1 attacks towards row 0, P2 attacks towards row max
    # For P2, rotate ALL channels 180° so they also see themselves "attacking up"
    observation = lax.cond(
        is_p2,
        lambda: jnp.flip(jnp.flip(observation, axis=1), axis=2),  # 180° rotation on (C,H,W)
        lambda: observation
    )

    return observation


# ============================================================================
# Pretty Printing (for debugging)
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
    
    # Show jump sequence if active
    if int(state.jump_sequence_length) > 0:
        seq_str = " -> ".join([
            f"({int(state.jump_sequence[i, 0])},{int(state.jump_sequence[i, 1])})"
            for i in range(int(state.jump_sequence_length))
        ])
        lines.append(f"Jump sequence: {seq_str}")
    
    lines.append("")
    
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            tile = int(board[r, c])
            row_str += symbols.get(tile, '?')
        lines.append(f"{r:2d} {row_str}")
    
    return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_jump_sequence_tracking():
    """Test that jump sequences are properly tracked."""
    config = EnvConfig(rows=9, cols=9)
    state = reset(config)
    
    # Ball starts at (4, 4) for 9x9 board
    assert int(state.ball_pos[0]) == 4, f"Ball row: {state.ball_pos[0]}"
    assert int(state.ball_pos[1]) == 4, f"Ball col: {state.ball_pos[1]}"
    
    # Place men to enable a multi-jump sequence
    # Place man at (3, 4) - above ball
    place1 = 3 * 9 + 4  # Row 3, col 4
    state = step(state, jnp.array(place1, dtype=jnp.int32), config)
    
    # Place man at (2, 4) - for second jump (opponent places elsewhere, we skip that for simplicity)
    # Actually, let's set up the board manually for a clean test
    state = reset(config)
    state = state._replace(
        board=state.board.at[3, 4].set(MAN).at[2, 4].set(MAN)  # Two men above ball
    )
    
    # Now execute first jump: from (4,4) over (3,4) to land at (2,4)? 
    # Wait, (2,4) has a man. Let's think...
    # Jump over (3,4) to (2,4) - but there's a man there!
    # Let's place men at (3,4) only, and have (2,4) empty
    state = reset(config)
    state = state._replace(
        board=state.board.at[3, 4].set(MAN)
    )
    
    # Execute jump to (2, 4)
    total_pos = 9 * 9
    jump_to_2_4 = total_pos + 2 * 9 + 4
    state = step(state, jnp.array(jump_to_2_4, dtype=jnp.int32), config)
    
    # Check jump sequence
    assert bool(state.is_jumping), "Should be in jump sequence"
    assert int(state.jump_sequence_length) == 2, f"Sequence length should be 2: {state.jump_sequence_length}"
    
    # Sequence should be: [(4,4), (2,4)]
    assert int(state.jump_sequence[0, 0]) == 4 and int(state.jump_sequence[0, 1]) == 4, \
        f"First position should be (4,4): {state.jump_sequence[0]}"
    assert int(state.jump_sequence[1, 0]) == 2 and int(state.jump_sequence[1, 1]) == 4, \
        f"Second position should be (2,4): {state.jump_sequence[1]}"
    
    print("✓ test_jump_sequence_tracking passed")


def test_network_input_encoding():
    """Test that network input is correctly encoded."""
    config = EnvConfig(rows=9, cols=9)
    state = reset(config)
    
    # Set up a jump sequence manually
    state = state._replace(
        is_jumping=jnp.array(True, dtype=jnp.bool_),
        jump_sequence=state.jump_sequence.at[0].set(jnp.array([4, 4])).at[1].set(jnp.array([2, 4])),
        jump_sequence_length=jnp.array(2, dtype=jnp.int32),
    )
    
    obs = state_to_network_input(state, config)

    # Check shape (9 channels: 4 board + 5 jump)
    assert obs.shape == (9, 9, 9), f"Observation shape: {obs.shape}"

    # Check channel 0 is ball position (binary)
    assert float(obs[0, 4, 4]) == 1.0, f"Ball position in channel 0: {obs[0, 4, 4]}"

    # Check jump sequence channels (now channels 4-8)
    # Position (4,4) was visited at step 0, so layer 4 should have value 1.0 (step 0 + 1)
    assert float(obs[4, 4, 4]) == 1.0, f"Jump layer 0 at (4,4): {obs[4, 4, 4]}"
    # Position (2,4) was visited at step 1, so layer 4 should have value 2.0 (step 1 + 1)
    assert float(obs[4, 2, 4]) == 2.0, f"Jump layer 0 at (2,4): {obs[4, 2, 4]}"
    
    print("✓ test_network_input_encoding passed")


def test_repeated_position_visit():
    """Test encoding when the same position is visited multiple times."""
    config = EnvConfig(rows=9, cols=9)
    state = reset(config)
    
    # Simulate a sequence that visits (4,4) twice: (4,4) -> (2,4) -> (4,4)
    state = state._replace(
        is_jumping=jnp.array(True, dtype=jnp.bool_),
        jump_sequence=state.jump_sequence.at[0].set(jnp.array([4, 4]))
                                          .at[1].set(jnp.array([2, 4]))
                                          .at[2].set(jnp.array([4, 4])),  # Back to start!
        jump_sequence_length=jnp.array(3, dtype=jnp.int32),
        ball_pos=jnp.array([4, 4], dtype=jnp.int32),  # Ball returned to center
    )
    
    obs = state_to_network_input(state, config)
    
    # Position (4,4) visited twice (jump channels start at index 4):
    # - First visit at step 0 -> jump layer 0 (ch 4), value 1.0
    # - Second visit at step 2 -> jump layer 1 (ch 5), value 3.0
    assert float(obs[4, 4, 4]) == 1.0, f"Jump layer 0 at (4,4): {obs[4, 4, 4]} (expected 1.0)"
    assert float(obs[5, 4, 4]) == 3.0, f"Jump layer 1 at (4,4): {obs[5, 4, 4]} (expected 3.0)"

    # Position (2,4) visited once at step 1 -> jump layer 0 (ch 4), value 2.0
    assert float(obs[4, 2, 4]) == 2.0, f"Jump layer 0 at (2,4): {obs[4, 2, 4]} (expected 2.0)"
    
    print("✓ test_repeated_position_visit passed")


if __name__ == "__main__":
    print("Testing Phutball Environment...\n")
    
    # Basic tests
    config = EnvConfig(rows=9, cols=9)
    state = reset(config)
    print("Initial board:")
    print(render_board(state))
    print()
    
    test_jump_sequence_tracking()
    test_network_input_encoding()
    test_repeated_position_visit()
    
    print("\n✓ All tests passed!")