"""
Halma Environment in JAX - Drop-in replacement for Phutball env.

Halma rules:
- Square board (10x10 default, configurable)
- Each player has pieces in opposite corner "camps"
- Goal: move all your pieces to the opponent's starting camp
- Movement: step to adjacent empty cell OR chain jumps over any pieces
- First player to fill opponent's camp wins

Key similarity to Phutball: multi-jump sequences
Key difference: race game (fill camp) vs scoring game (reach endzone)
"""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
from functools import partial

# Board cell values
EMPTY = 0
P1_PIECE = 1
P2_PIECE = 2
P1_CAMP = 3   # P1's starting camp (P2's goal)
P2_CAMP = 4   # P2's starting camp (P1's goal)


@struct.dataclass
class HalmaConfig:
    """Environment configuration."""
    rows: int = 10
    cols: int = 10
    camp_size: int = 15  # pieces per player
    max_turns: int = 500


@struct.dataclass
class HalmaState:
    """Game state."""
    board: jnp.ndarray          # (rows, cols) - piece positions
    camp_mask_p1: jnp.ndarray   # (rows, cols) - P1's starting camp (P2's goal)
    camp_mask_p2: jnp.ndarray   # (rows, cols) - P2's starting camp (P1's goal)
    current_player: jnp.ndarray # scalar: 1 or 2
    turn_count: jnp.ndarray     # scalar
    terminated: jnp.ndarray     # scalar bool
    winner: jnp.ndarray         # scalar: 0 (ongoing/draw), 1, or 2


def _create_camp_mask(rows: int, cols: int, corner: str) -> jnp.ndarray:
    """Create triangular camp mask for a corner."""
    mask = jnp.zeros((rows, cols), dtype=jnp.bool_)

    # Camp size determines triangle size
    # For 10x10 with 15 pieces: triangle with rows of 5,4,3,2,1
    if corner == "top_left":
        # Top-left triangle
        for i in range(5):
            for j in range(5 - i):
                mask = mask.at[i, j].set(True)
    elif corner == "bottom_right":
        # Bottom-right triangle
        for i in range(5):
            for j in range(5 - i):
                mask = mask.at[rows - 1 - i, cols - 1 - j].set(True)

    return mask


def reset(config: HalmaConfig) -> HalmaState:
    """Initialize game state."""
    rows, cols = config.rows, config.cols

    # Create camp masks
    camp_mask_p1 = _create_camp_mask(rows, cols, "top_left")
    camp_mask_p2 = _create_camp_mask(rows, cols, "bottom_right")

    # Initialize board with pieces in camps
    board = jnp.zeros((rows, cols), dtype=jnp.int32)
    board = jnp.where(camp_mask_p1, P1_PIECE, board)
    board = jnp.where(camp_mask_p2, P2_PIECE, board)

    return HalmaState(
        board=board,
        camp_mask_p1=camp_mask_p1,
        camp_mask_p2=camp_mask_p2,
        current_player=jnp.array(1, dtype=jnp.int32),
        turn_count=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(0, dtype=jnp.int32),
    )


def _get_jump_destinations(board: jnp.ndarray, start_r: int, start_c: int,
                           rows: int, cols: int) -> jnp.ndarray:
    """Get all reachable positions via jump chains from start position."""
    # BFS to find all reachable jump destinations
    reachable = jnp.zeros((rows, cols), dtype=jnp.bool_)

    # 8 directions: N, NE, E, SE, S, SW, W, NW
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    # This needs to be iterative - we'll use a fixed number of iterations
    # since max chain length is bounded by board size
    visited = jnp.zeros((rows, cols), dtype=jnp.bool_)
    visited = visited.at[start_r, start_c].set(True)
    frontier = jnp.zeros((rows, cols), dtype=jnp.bool_)
    frontier = frontier.at[start_r, start_c].set(True)

    max_iterations = rows + cols  # Upper bound on chain length

    def jump_step(carry, _):
        visited, frontier, reachable = carry
        new_frontier = jnp.zeros((rows, cols), dtype=jnp.bool_)

        # For each position in frontier, check all jump directions
        for dr, dc in directions:
            # Position of piece to jump over
            mid_r = jnp.arange(rows)[:, None] + dr
            mid_c = jnp.arange(cols)[None, :] + dc
            # Landing position
            land_r = jnp.arange(rows)[:, None] + 2 * dr
            land_c = jnp.arange(cols)[None, :] + 2 * dc

            # Valid if: in bounds, mid has piece, land is empty, not visited
            valid_mid = (mid_r >= 0) & (mid_r < rows) & (mid_c >= 0) & (mid_c < cols)
            valid_land = (land_r >= 0) & (land_r < rows) & (land_c >= 0) & (land_c < cols)

            # Check conditions where valid
            mid_has_piece = jnp.where(
                valid_mid,
                board[jnp.clip(mid_r, 0, rows-1), jnp.clip(mid_c, 0, cols-1)] != EMPTY,
                False
            )
            land_empty = jnp.where(
                valid_land,
                board[jnp.clip(land_r, 0, rows-1), jnp.clip(land_c, 0, cols-1)] == EMPTY,
                False
            )
            land_not_visited = jnp.where(
                valid_land,
                ~visited[jnp.clip(land_r, 0, rows-1), jnp.clip(land_c, 0, cols-1)],
                False
            )

            can_jump = frontier & valid_mid & valid_land & mid_has_piece & land_empty & land_not_visited

            # Add landing positions to new frontier
            # This is tricky in JAX - we need scatter
            land_r_clipped = jnp.clip(land_r, 0, rows-1)
            land_c_clipped = jnp.clip(land_c, 0, cols-1)

            # For each valid jump, mark the landing position
            new_frontier = new_frontier | jnp.where(
                can_jump,
                jnp.zeros((rows, cols), dtype=jnp.bool_).at[land_r_clipped, land_c_clipped].set(True),
                jnp.zeros((rows, cols), dtype=jnp.bool_)
            )

        visited = visited | new_frontier
        reachable = reachable | new_frontier

        return (visited, new_frontier, reachable), None

    # Run BFS iterations
    (visited, _, reachable), _ = jax.lax.scan(
        jump_step, (visited, frontier, reachable), None, length=max_iterations
    )

    return reachable


def get_legal_actions(state: HalmaState, config: HalmaConfig) -> jnp.ndarray:
    """
    Get legal actions as a flat boolean array.

    Action encoding: from_idx * (rows * cols) + to_idx
    where idx = row * cols + col

    Total actions: (rows * cols) ^ 2
    """
    rows, cols = config.rows, config.cols
    board = state.board
    player = state.current_player
    player_piece = jnp.where(player == 1, P1_PIECE, P2_PIECE)

    total_positions = rows * cols
    legal = jnp.zeros((total_positions, total_positions), dtype=jnp.bool_)

    # 8 directions for step moves
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    # For each position, check if it's our piece and find legal moves
    for from_r in range(rows):
        for from_c in range(cols):
            from_idx = from_r * cols + from_c
            is_our_piece = board[from_r, from_c] == player_piece

            # Step moves (one cell in any direction)
            for dr, dc in directions:
                to_r, to_c = from_r + dr, from_c + dc
                if 0 <= to_r < rows and 0 <= to_c < cols:
                    to_idx = to_r * cols + to_c
                    is_empty = board[to_r, to_c] == EMPTY
                    legal = legal.at[from_idx, to_idx].set(
                        is_our_piece & is_empty
                    )

            # Jump moves - simplified: check immediate jumps only for now
            # Full chain jumps would need iterative BFS
            for dr, dc in directions:
                mid_r, mid_c = from_r + dr, from_c + dc
                to_r, to_c = from_r + 2*dr, from_c + 2*dc
                if (0 <= mid_r < rows and 0 <= mid_c < cols and
                    0 <= to_r < rows and 0 <= to_c < cols):
                    to_idx = to_r * cols + to_c
                    mid_has_piece = board[mid_r, mid_c] != EMPTY
                    land_empty = board[to_r, to_c] == EMPTY
                    legal = legal.at[from_idx, to_idx].set(
                        legal[from_idx, to_idx] | (is_our_piece & mid_has_piece & land_empty)
                    )

    return legal.reshape(-1)


def step(state: HalmaState, action: jnp.ndarray, config: HalmaConfig) -> HalmaState:
    """Execute action and return new state."""
    rows, cols = config.rows, config.cols
    total_positions = rows * cols

    # Decode action
    from_idx = action // total_positions
    to_idx = action % total_positions
    from_r, from_c = from_idx // cols, from_idx % cols
    to_r, to_c = to_idx // cols, to_idx % cols

    # Move piece
    player_piece = jnp.where(state.current_player == 1, P1_PIECE, P2_PIECE)
    board = state.board.at[from_r, from_c].set(EMPTY)
    board = board.at[to_r, to_c].set(player_piece)

    # Check win condition: all pieces in opponent's camp
    p1_in_p2_camp = jnp.sum((board == P1_PIECE) & state.camp_mask_p2)
    p2_in_p1_camp = jnp.sum((board == P2_PIECE) & state.camp_mask_p1)

    p1_wins = p1_in_p2_camp >= config.camp_size
    p2_wins = p2_in_p1_camp >= config.camp_size

    # Check for max turns
    turn_count = state.turn_count + 1
    max_turns_reached = turn_count >= config.max_turns

    terminated = p1_wins | p2_wins | max_turns_reached
    winner = jnp.where(p1_wins, 1, jnp.where(p2_wins, 2, 0))

    # Switch player
    next_player = jnp.where(state.current_player == 1, 2, 1)

    return HalmaState(
        board=board,
        camp_mask_p1=state.camp_mask_p1,
        camp_mask_p2=state.camp_mask_p2,
        current_player=next_player,
        turn_count=turn_count,
        terminated=terminated,
        winner=winner,
    )


def state_to_network_input(state: HalmaState, config: HalmaConfig) -> jnp.ndarray:
    """
    Convert state to network input tensor.

    Channels (6 total, same as Phutball):
    0: Current player's pieces
    1: Opponent's pieces
    2: Current player's goal camp
    3: Opponent's goal camp
    4: Current player indicator (all 1s if P1, all 0s if P2)
    5: Turn progress (normalized)
    """
    rows, cols = config.rows, config.cols
    board = state.board
    player = state.current_player

    # Determine piece types based on current player
    our_piece = jnp.where(player == 1, P1_PIECE, P2_PIECE)
    opp_piece = jnp.where(player == 1, P2_PIECE, P1_PIECE)

    # Our goal is opponent's starting camp
    our_goal = jnp.where(player == 1, state.camp_mask_p2, state.camp_mask_p1)
    opp_goal = jnp.where(player == 1, state.camp_mask_p1, state.camp_mask_p2)

    channels = jnp.stack([
        (board == our_piece).astype(jnp.float32),
        (board == opp_piece).astype(jnp.float32),
        our_goal.astype(jnp.float32),
        opp_goal.astype(jnp.float32),
        jnp.full((rows, cols), (player == 1).astype(jnp.float32)),
        jnp.full((rows, cols), state.turn_count / config.max_turns),
    ], axis=0)

    return channels


def render_board(state: HalmaState) -> str:
    """Render board as ASCII string."""
    board = state.board
    rows, cols = board.shape

    symbols = {
        EMPTY: '.',
        P1_PIECE: 'O',
        P2_PIECE: 'X',
    }

    lines = []
    lines.append(f"Turn {int(state.turn_count)}, Player {int(state.current_player)}")
    lines.append("  " + "".join(f"{c:2d}" for c in range(cols)))

    for r in range(rows):
        row_str = f"{r:2d} "
        for c in range(cols):
            cell = int(board[r, c])
            # Mark camps
            if state.camp_mask_p1[r, c] and cell == EMPTY:
                row_str += " o"  # P1 camp empty
            elif state.camp_mask_p2[r, c] and cell == EMPTY:
                row_str += " x"  # P2 camp empty
            else:
                row_str += f" {symbols.get(cell, '?')}"
        lines.append(row_str)

    lines.append(f"Winner: {int(state.winner)}" if state.terminated else "")
    return "\n".join(lines)


# Aliases to match Phutball interface
EnvConfig = HalmaConfig
PhutballState = HalmaState  # For compatibility


if __name__ == "__main__":
    # Quick test
    config = HalmaConfig(rows=10, cols=10)
    state = reset(config)
    print(render_board(state))
    print(f"\nBoard shape: {state.board.shape}")
    print(f"Camp mask P1 sum: {state.camp_mask_p1.sum()}")
    print(f"Camp mask P2 sum: {state.camp_mask_p2.sum()}")

    # Test network input
    net_input = state_to_network_input(state, config)
    print(f"Network input shape: {net_input.shape}")

    # Test legal actions
    legal = get_legal_actions(state, config)
    print(f"Legal actions shape: {legal.shape}")
    print(f"Num legal actions: {legal.sum()}")
