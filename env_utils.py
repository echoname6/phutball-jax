# env_utils.py
"""
Utilities for visualization/debugging. Not JIT-compiled.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, List
from .phutball_env_jax import PhutballState, EnvConfig, EMPTY, BALL, MAN, END_HI, END_LO, MAX_JUMP_SEQUENCE_LENGTH
import jax.numpy as jnp


class TileType(Enum):
    """Tile types for game_manager compatibility."""
    EMPTY = 0
    PHUTBALL = -1
    PHUTBALL_MAN = 1
    END_HI = 2
    END_LO = -2


def state_to_snapshot(state: PhutballState) -> dict:
    """Convert PhutballState to dict for saving/external use."""
    seq_len = int(state.jump_sequence_length)
    jump_seq = []
    for i in range(seq_len):
        r, c = int(state.jump_sequence[i, 0]), int(state.jump_sequence[i, 1])
        if r >= 0 and c >= 0:
            jump_seq.append((r, c))

    return {
        'board': np.array(state.board),
        'current_player': int(state.current_player),
        'phutball_pos': (int(state.ball_pos[0]), int(state.ball_pos[1])),
        'num_turns': int(state.num_turns),
        'winner': int(state.winner),
        'is_jumping': bool(state.is_jumping),
        'jump_seq': jump_seq,
    }


def snapshot_to_state(snapshot: dict, config: EnvConfig) -> PhutballState:
    """Convert dict snapshot back to PhutballState."""
    jump_seq = snapshot.get('jump_seq', [])
    jump_sequence = np.full((MAX_JUMP_SEQUENCE_LENGTH, 2), -1, dtype=np.int32)
    for i, pos in enumerate(jump_seq[:MAX_JUMP_SEQUENCE_LENGTH]):
        jump_sequence[i] = pos

    return PhutballState(
        board=jnp.array(snapshot['board'], dtype=jnp.int32),
        ball_pos=jnp.array(snapshot['phutball_pos'], dtype=jnp.int32),
        current_player=jnp.array(snapshot['current_player'], dtype=jnp.int32),
        is_jumping=jnp.array(snapshot.get('is_jumping', len(jump_seq) > 0), dtype=jnp.bool_),
        terminated=jnp.array(snapshot.get('winner', 0) != 0, dtype=jnp.bool_),
        winner=jnp.array(snapshot.get('winner', 0), dtype=jnp.int32),
        num_turns=jnp.array(snapshot.get('num_turns', 0), dtype=jnp.int32),
        jump_sequence=jnp.array(jump_sequence, dtype=jnp.int32),
        jump_sequence_length=jnp.array(len(jump_seq), dtype=jnp.int32),
    )


def text_render(state: PhutballState, for_player: int = 1) -> str:
    """Render board as text with row/column coordinates."""
    board = np.array(state.board)
    rows, cols = board.shape

    grid = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            tile = board[r, c]
            if tile == END_HI:
                row_chars.append("+")
            elif tile == END_LO:
                row_chars.append("-")
            elif tile == BALL:
                row_chars.append("O")
            elif tile == MAN:
                row_chars.append("x")
            else:
                row_chars.append(".")
        grid.append(row_chars)

    # Build row indices for display (matrix coordinates)
    row_indices = list(range(rows))
    col_indices = list(range(cols))

    if for_player == 2:
        # Flip board visually but keep matrix coordinate labels
        swap = {"+": "-", "-": "+"}
        grid = [[swap.get(c, c) for c in row[::-1]] for row in reversed(grid)]
        row_indices = list(reversed(row_indices))
        col_indices = list(reversed(col_indices))

    header = f"TURN: {int(state.num_turns) + 1} PLAYER: {int(state.current_player)}"
    lines = ["<board_state>", header]
    if bool(state.is_jumping):
        lines.append("JUMP SEQUENCE ACTIVE")

    # Column header with stacked tens/ones digits for alignment
    tens_row = "   " + " ".join(str(c // 10) if c >= 10 else " " for c in col_indices)
    ones_row = "   " + " ".join(str(c % 10) for c in col_indices)
    lines.append(tens_row)
    lines.append(ones_row)

    # Board rows with row index on left only
    for row_chars, row_idx in zip(grid, row_indices):
        row_str = f"{row_idx:2d} " + " ".join(row_chars)
        lines.append(row_str)
    lines.append("</board_state>")

    # Ball position in matrix coordinates
    ball_row, ball_col = int(state.ball_pos[0]), int(state.ball_pos[1])
    lines.append(f"Ball at: ({ball_row}, {ball_col})")

    if int(state.winner) > 0:
        lines.append(f"Winner: {int(state.winner)}")

    return "\n".join(lines)


def check_single_jump(start: Tuple[int, int], end: Tuple[int, int], board: np.ndarray) -> bool:
    """Validate a single jump segment."""
    rows, cols = board.shape
    r0, c0 = start
    r1, c1 = end

    if not (0 <= r1 < rows and 0 <= c1 < cols):
        return False
    if r0 == r1 and c0 == c1:
        return False

    dr, dc = r1 - r0, c1 - c0
    if abs(dr) <= 1 and abs(dc) <= 1:
        return False
    if dr != 0 and dc != 0 and abs(dr) != abs(dc):
        return False

    r_step, c_step = np.sign(dr), np.sign(dc)
    steps = max(abs(dr), abs(dc))

    for i in range(1, steps):
        if board[r0 + i * r_step, c0 + i * c_step] != MAN:
            return False

    landing = board[r1, c1]
    return landing != MAN and landing != BALL


class PhutballEnv:
    """Thin wrapper providing class-based interface over JAX functions."""

    def __init__(self, rows: int = 21, cols: int = 15):
        from .phutball_env_jax import reset, step, get_legal_actions, get_legal_placements, get_legal_jumps
        self._jax_reset = reset
        self._jax_step = step
        self._jax_get_legal_actions = get_legal_actions
        self._jax_get_legal_placements = get_legal_placements
        self._jax_get_legal_jumps = get_legal_jumps

        self.rows = rows
        self.cols = cols
        self.total_positions = rows * cols
        self._config = EnvConfig(rows=rows, cols=cols)
        self._state: Optional[PhutballState] = None
        self.reset()

    def reset(self, seed=None, options=None):
        self._state = self._jax_reset(self._config)
        return self._get_observation(), {}

    @property
    def _board(self):
        return np.array(self._state.board)

    @property
    def board(self):
        return self._board

    @property
    def current_player(self):
        return int(self._state.current_player)

    @property
    def _current_player(self):
        return self.current_player

    @property
    def phutball_pos(self):
        return (int(self._state.ball_pos[0]), int(self._state.ball_pos[1]))

    @property
    def _phutball_pos(self):
        return self.phutball_pos

    @property
    def num_turns(self):
        return int(self._state.num_turns)

    def is_jump_seq_underway(self) -> bool:
        return bool(self._state.is_jumping)

    def get_state_snapshot(self) -> dict:
        snap = state_to_snapshot(self._state)
        # Add jump_seq_layers for compatibility with game_manager
        snap['jump_seq_layers'] = self._build_jump_seq_layers()
        return snap

    def _build_jump_seq_layers(self) -> np.ndarray:
        layers = np.zeros((5, self.rows, self.cols), dtype=np.float32)
        seq_len = int(self._state.jump_sequence_length)
        visit_counts = {}
        for step_idx in range(seq_len):
            r = int(self._state.jump_sequence[step_idx, 0])
            c = int(self._state.jump_sequence[step_idx, 1])
            if r >= 0 and c >= 0:
                layer = visit_counts.get((r, c), 0)
                if layer < 5:
                    layers[layer, r, c] = step_idx + 1
                visit_counts[(r, c)] = layer + 1
        return layers

    def load_state_snapshot(self, snapshot: dict):
        self._state = snapshot_to_state(snapshot, self._config)

    def step(self, action: int):
        self._state = self._jax_step(self._state, jnp.array(action, dtype=jnp.int32), self._config)
        terminated = bool(self._state.terminated)
        winner = int(self._state.winner)
        reward = 1.0 if terminated and winner == (3 - self.current_player) else 0.0
        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((6, self.rows, self.cols), dtype=np.float32)
        obs[0] = np.array(self._state.board)
        obs[1:] = self._build_jump_seq_layers()
        return obs

    def get_legal_actions(self) -> np.ndarray:
        return np.array(self._jax_get_legal_actions(self._state, self._config))

    def get_legal_placements(self, board: np.ndarray) -> np.ndarray:
        temp = self._state._replace(board=jnp.array(board, dtype=jnp.int32))
        return np.array(self._jax_get_legal_placements(temp, self._config))

    def get_legal_jumps(self, board: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
        temp = self._state._replace(
            board=jnp.array(board, dtype=jnp.int32),
            ball_pos=jnp.array(pos, dtype=jnp.int32),
        )
        return np.array(self._jax_get_legal_jumps(temp, self._config))

    def get_legal_actions_for_state(self, snapshot: dict) -> np.ndarray:
        temp_state = snapshot_to_state(snapshot, self._config)
        return np.array(self._jax_get_legal_actions(temp_state, self._config))

    def check_single_jump(self, start: Tuple[int, int], end: Tuple[int, int],
                          board: Optional[np.ndarray] = None) -> bool:
        return check_single_jump(start, end, board if board is not None else self._board)

    def text_render(self, max_width: Optional[int] = None, for_player: Optional[int] = None) -> str:
        return text_render(self._state, for_player or self.current_player)
