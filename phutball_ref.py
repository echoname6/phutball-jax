"""
Standalone reference implementation of Phutball.
Extracted from the original PyTorch env for comparison testing.
No gymnasium or numba dependencies.
"""

import numpy as np
from enum import Enum

class TileType(Enum):
    EMPTY = 0
    PHUTBALL = -1
    PHUTBALL_MAN = 1
    END_HI = 2
    END_LO = -2

EMPTY_VAL = TileType.EMPTY.value
PHUTBALL_VAL = TileType.PHUTBALL.value
MAN_VAL = TileType.PHUTBALL_MAN.value
END_HI_VAL = TileType.END_HI.value
END_LO_VAL = TileType.END_LO.value


def _get_legal_placements(board: np.ndarray, rows: int, cols: int, total_positions: int) -> np.ndarray:
    """Find legal placement positions."""
    legal_mask = np.zeros(total_positions, dtype=np.int8)
    for r in range(1, rows - 1):  # Rows 1 to rows-2 inclusive
        for c in range(cols):
            tile = board[r, c]
            if not (tile == PHUTBALL_VAL or tile == MAN_VAL):
                idx = r * cols + c
                legal_mask[idx] = 1
    return legal_mask


def _get_legal_jumps(board: np.ndarray, start_r: int, start_c: int, rows: int, cols: int, total_positions: int) -> np.ndarray:
    """Find legal jump landing positions."""
    legal_mask = np.zeros(total_positions, dtype=np.int8)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    max_dim = max(rows, cols)

    for dr, dc in directions:
        num_men_jumped_over = 0
        for i in range(1, max_dim):
            check_r = start_r + i * dr
            check_c = start_c + i * dc

            if not (0 <= check_r < rows and 0 <= check_c < cols):
                break

            tile = board[check_r, check_c]

            if tile == MAN_VAL:
                num_men_jumped_over += 1
                continue
            elif tile == EMPTY_VAL or tile == END_HI_VAL or tile == END_LO_VAL:
                if num_men_jumped_over > 0:
                    idx = check_r * cols + check_c
                    legal_mask[idx] = 1
                break
            else:
                break

    return legal_mask


class PhutballEnvRef:
    """Reference Phutball environment (no gymnasium)."""

    def __init__(self, rows: int = 21, cols: int = 15):
        if rows % 2 == 0 or cols % 2 == 0:
            raise ValueError("Board dimensions must be odd numbers.")
        
        self.rows = rows
        self.cols = cols
        self.total_positions = rows * cols
        self.action_space_n = 2 * self.total_positions + 1

        self._board = None
        self._current_player = None
        self._phutball_pos = None
        self._num_turns = None
        self._winner = 0
        self._jump_seq = None

        self.reset()

    def reset(self):
        self._board = np.zeros((self.rows, self.cols), dtype=np.int32)
        self._board[0, :] = END_HI_VAL
        self._board[self.rows - 1, :] = END_LO_VAL

        self._phutball_pos = (self.rows // 2, self.cols // 2)
        self._board[self._phutball_pos] = PHUTBALL_VAL

        self._current_player = 1
        self._num_turns = 0
        self._winner = 0
        self._jump_seq = []

        return self._get_observation()

    @property
    def board(self):
        return self._board

    def _get_observation(self):
        return self._board.copy()

    def is_jump_seq_underway(self):
        return len(self._jump_seq) > 0

    def check_for_win(self):
        """Check if ball reached goal line."""
        if self._phutball_pos[0] <= 1:
            return 2  # Player 2 wins
        if self._phutball_pos[0] >= self.rows - 2:
            return 1  # Player 1 wins
        return 0

    def _end_jump_seq(self):
        self._jump_seq = []
        self._winner = self.check_for_win()
        return self._winner != 0

    def end_turn(self):
        self._current_player = 3 - self._current_player
        self._num_turns += 1
        return False

    def get_action_pos(self, action):
        if action < self.total_positions:
            board_pos = action
        elif action < 2 * self.total_positions:
            board_pos = action - self.total_positions
        else:
            return None
        return board_pos // self.cols, board_pos % self.cols

    def get_legal_placements(self):
        return _get_legal_placements(self._board, self.rows, self.cols, self.total_positions)

    def get_legal_jumps(self):
        start_r, start_c = self._phutball_pos
        return _get_legal_jumps(self._board, start_r, start_c, self.rows, self.cols, self.total_positions)

    def get_legal_actions(self):
        if self.is_jump_seq_underway():
            placement_mask = np.zeros(self.total_positions, dtype=np.int8)
            jump_mask = self.get_legal_jumps()
            halt_legal = 1
        else:
            placement_mask = self.get_legal_placements()
            jump_mask = self.get_legal_jumps()
            halt_legal = 0
        
        return np.concatenate([
            placement_mask,
            jump_mask,
            np.array([halt_legal], dtype=np.int8)
        ])

    def _calculate_jumped_men(self, start_pos, end_pos):
        jumped = []
        r0, c0 = start_pos
        r1, c1 = end_pos
        dr = r1 - r0
        dc = c1 - c0
        if not (r0 == r1 or c0 == c1 or abs(dr) == abs(dc)):
            return []
        r_step = np.sign(dr)
        c_step = np.sign(dc)
        steps = max(abs(dr), abs(dc))
        for i in range(1, steps):
            check_r, check_c = r0 + i * r_step, c0 + i * c_step
            if 0 <= check_r < self.rows and 0 <= check_c < self.cols:
                if self._board[check_r, check_c] == MAN_VAL:
                    jumped.append((check_r, check_c))
        return jumped

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        player_at_start = self._current_player

        halt_action_index = 2 * self.total_positions
        placement_end_index = self.total_positions

        if action == halt_action_index:
            terminated = self._end_jump_seq()
            if not terminated:
                truncated = self.end_turn()
        elif action < placement_end_index:
            placement_pos = self.get_action_pos(action)
            self._board[placement_pos] = MAN_VAL
            truncated = self.end_turn()
        elif action < halt_action_index:
            jump_target_pos = self.get_action_pos(action)
            start_pos = self._phutball_pos

            if not self.is_jump_seq_underway():
                self._jump_seq = [start_pos]

            self._jump_seq.append(jump_target_pos)

            # Board updates
            jumped_over = self._calculate_jumped_men(start_pos, jump_target_pos)
            self._board[start_pos] = EMPTY_VAL
            for r_jmp, c_jmp in jumped_over:
                self._board[r_jmp, c_jmp] = EMPTY_VAL
            self._phutball_pos = jump_target_pos
            self._board[self._phutball_pos] = PHUTBALL_VAL

            # Win check
            if self.check_for_win():
                self._winner = self._current_player
                terminated = True
                self._end_jump_seq()

        if terminated:
            if self._winner == player_at_start:
                reward = 1.0
            elif self._winner == (3 - player_at_start):
                reward = -1.0

        obs = self._get_observation()
        info = {}
        return obs, reward, terminated, truncated, info


def test_ref_env():
    """Quick test of reference env."""
    env = PhutballEnvRef(rows=21, cols=15)
    
    # Check initial state
    assert env._phutball_pos == (10, 7)
    assert env._current_player == 1
    assert not env.is_jump_seq_underway()
    
    # Check legal actions
    legal = env.get_legal_actions()
    assert legal.shape == (631,)  # 2 * 315 + 1
    assert np.sum(legal[:315]) == 284  # Placements
    assert np.sum(legal[315:630]) == 0  # Jumps (none legal)
    assert legal[630] == 0  # Halt not legal
    
    # Place a man
    env.step(9 * 15 + 7)  # Place at (9, 7)
    assert env._board[9, 7] == MAN_VAL
    assert env._current_player == 2
    
    # Check jump is now legal
    legal = env.get_legal_actions()
    jump_idx = 315 + 8 * 15 + 7  # Jump to (8, 7)
    assert legal[jump_idx] == 1
    
    print("✓ Reference env tests passed")


if __name__ == "__main__":
    test_ref_env()