"""
Gomoku (Five in a Row) Environment in JAX.

Pure JAX implementation compatible with AlphaZero training.
Rules: Two players alternate placing stones on a grid. First to get
`win_length` consecutive stones (horizontally, vertically, or diagonally) wins.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

# Board cell values
EMPTY = 0
BLACK = 1  # Player 1
WHITE = 2  # Player 2


class GomokuConfig(NamedTuple):
    board_size: int = 9
    win_length: int = 5
    max_turns: int = 81  # board_size ** 2


class GomokuState(NamedTuple):
    board: jnp.ndarray           # (board_size, board_size) int32
    current_player: jnp.ndarray  # () int32: BLACK=1 or WHITE=2
    terminated: jnp.ndarray      # () bool
    winner: jnp.ndarray          # () int32: 0=ongoing/draw, 1=BLACK, 2=WHITE
    num_turns: jnp.ndarray       # () int32


def reset(config: GomokuConfig) -> GomokuState:
    """Return initial empty board state with BLACK to move."""
    return GomokuState(
        board=jnp.zeros((config.board_size, config.board_size), dtype=jnp.int32),
        current_player=jnp.array(BLACK, dtype=jnp.int32),
        terminated=jnp.array(False),
        winner=jnp.array(0, dtype=jnp.int32),
        num_turns=jnp.array(0, dtype=jnp.int32),
    )


def _check_winner(board: jnp.ndarray, player: jnp.ndarray,
                   board_size: int, win_length: int) -> jnp.ndarray:
    """Check if `player` has `win_length` in a row using 2D convolution."""
    mask = (board == player).astype(jnp.float32)
    x = mask[None, None, :, :]  # (1, 1, H, W) for conv

    # Four direction kernels
    k_h = jnp.ones((1, 1, 1, win_length))                                    # horizontal
    k_v = jnp.ones((1, 1, win_length, 1))                                    # vertical
    k_d1 = jnp.eye(win_length, dtype=jnp.float32)[None, None, :, :]          # diagonal \
    k_d2 = jnp.flip(jnp.eye(win_length, dtype=jnp.float32), axis=1)[None, None, :, :]  # diagonal /

    wl = jnp.float32(win_length)
    has_h = jnp.max(lax.conv(x, k_h, (1, 1), 'VALID')) >= wl
    has_v = jnp.max(lax.conv(x, k_v, (1, 1), 'VALID')) >= wl
    has_d1 = jnp.max(lax.conv(x, k_d1, (1, 1), 'VALID')) >= wl
    has_d2 = jnp.max(lax.conv(x, k_d2, (1, 1), 'VALID')) >= wl

    return has_h | has_v | has_d1 | has_d2


def step(state: GomokuState, action: jnp.ndarray,
         config: GomokuConfig) -> GomokuState:
    """
    Execute action (flat index into board) and return new state.
    If the game is already terminated or the cell is occupied, the state
    is returned unchanged.
    """
    board_size = config.board_size
    r = action // board_size
    c = action % board_size

    # Compute the board with the new stone placed
    placed_board = state.board.at[r, c].set(state.current_player)

    # Only apply if the cell was empty and the game isn't over
    valid = (state.board[r, c] == EMPTY) & ~state.terminated
    new_board = jnp.where(valid, placed_board, state.board)

    # Check for win by current player
    won = _check_winner(new_board, state.current_player,
                        board_size, config.win_length) & valid

    # Check for draw (board full, no winner)
    board_full = (jnp.sum(new_board == EMPTY) == 0) & ~won & valid

    terminated = state.terminated | won | board_full
    winner = jnp.where(won, state.current_player, state.winner)
    next_player = jnp.where(valid, 3 - state.current_player,
                            state.current_player)
    num_turns = state.num_turns + valid.astype(jnp.int32)

    return GomokuState(
        board=new_board,
        current_player=next_player,
        terminated=terminated,
        winner=winner,
        num_turns=num_turns,
    )


def get_legal_actions(state: GomokuState,
                      config: GomokuConfig) -> jnp.ndarray:
    """
    Return (board_size**2,) int8 mask: 1 = legal, 0 = illegal.
    Always ensures at least one action is "legal" to prevent MCTS crashes
    on terminal / full-board states.
    """
    legal = (state.board.reshape(-1) == EMPTY).astype(jnp.int8)
    # Fallback: if no empties (full board or terminated), mark action 0
    has_any = jnp.any(legal > 0)
    fallback = jnp.zeros_like(legal).at[0].set(1)
    return jnp.where(has_any, legal, fallback)


def state_to_network_input(state: GomokuState,
                           config: GomokuConfig) -> jnp.ndarray:
    """
    Convert state to network input: (3, board_size, board_size) float32.

    Channel 0: current player's stones
    Channel 1: opponent's stones
    Channel 2: colour indicator (1.0 if current player is BLACK, else 0.0)
    """
    board = state.board
    player = state.current_player
    opponent = 3 - player

    my_stones = (board == player).astype(jnp.float32)
    opp_stones = (board == opponent).astype(jnp.float32)
    color = jnp.full_like(my_stones, (player == BLACK).astype(jnp.float32))

    return jnp.stack([my_stones, opp_stones, color], axis=0)


def render_board(state: GomokuState) -> str:
    """Render board as ASCII string."""
    board = state.board
    size = board.shape[0]
    symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}

    player_sym = 'X' if int(state.current_player) == BLACK else 'O'
    lines = [f"Turn {int(state.num_turns)}, Player {int(state.current_player)} ({player_sym}) to move"]

    header = "   " + " ".join(f"{c:2d}" for c in range(size))
    lines.append(header)

    for r in range(size):
        row_str = f"{r:2d} "
        for c in range(size):
            cell = int(board[r, c])
            row_str += f" {symbols[cell]} "
        lines.append(row_str)

    if state.terminated:
        w = int(state.winner)
        if w > 0:
            ws = 'X' if w == BLACK else 'O'
            lines.append(f"Player {w} ({ws}) wins!")
        else:
            lines.append("Draw!")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aliases so train_gomoku.py can use a uniform interface
# ---------------------------------------------------------------------------
EnvConfig = GomokuConfig


if __name__ == "__main__":
    config = GomokuConfig(board_size=9, win_length=5)
    state = reset(config)
    print(render_board(state))
    print(f"Board shape: {state.board.shape}")
    print(f"Legal actions: {int(get_legal_actions(state, config).sum())}")
    print(f"Network input shape: {state_to_network_input(state, config).shape}")

    # Play a few moves
    state = step(state, jnp.array(40), config)  # Center
    state = step(state, jnp.array(0), config)
    state = step(state, jnp.array(31), config)  # Above center
    state = step(state, jnp.array(1), config)
    state = step(state, jnp.array(22), config)  # Two above
    state = step(state, jnp.array(2), config)
    state = step(state, jnp.array(49), config)  # Below center
    state = step(state, jnp.array(3), config)
    state = step(state, jnp.array(58), config)  # Two below  -> 5 in a column!
    print("\n" + render_board(state))

    # Test JIT compilation
    step_jit = jax.jit(lambda s, a: step(s, a, config))
    legal_jit = jax.jit(lambda s: get_legal_actions(s, config))
    obs_jit = jax.jit(lambda s: state_to_network_input(s, config))

    state2 = reset(config)
    _ = step_jit(state2, jnp.array(0))
    _ = legal_jit(state2)
    _ = obs_jit(state2)
    print("\nJIT compilation: OK")
