"""Fast pure-Python Phutball engine with the SAME rules as phutball_env_jax.py.

Used by the lookahead expert (expert/oracle.py), which needs millions of cheap
jump-chain enumerations. Verified move-for-move against the JAX env by
expert/test_engine_vs_jax.py.

Rules (from phutball_env_jax.py):
- Board rows x cols; rows 0-1 are END_HI (P1 attacks it), rows R-2..R-1 END_LO (P2 attacks it).
- Placement: any square in rows 1..R-2 that is not the ball or a man (end-zone squares included).
- Jump: in one of 8 directions over one or more CONTIGUOUS men, landing on the first square
  that is not a man (empty or end zone), in bounds, distance >= 2. Jumped men are removed.
- After any landing: row <= 1 -> P1 wins, row >= R-2 -> P2 wins, immediately (whoever jumped).
- A jump sequence continues until HALT (or a win); placements are illegal mid-sequence.
- Actions: placement = r*cols+c, jump landing = R*C + r*cols+c, halt = 2*R*C.
- Turn limit (draw) at max_turns completed turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EMPTY, BALL, MAN, END_HI, END_LO = 0, -1, 1, 2, -2
DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class State:
    rows: int
    cols: int
    board: list            # flat, rows*cols
    ball: int              # flat index
    player: int = 1        # 1 or 2
    jumping: bool = False
    winner: int = 0
    turns: int = 0
    seq: list = field(default_factory=list)   # positions visited in the current jump sequence (JAX encoding needs it)

    def copy(self) -> "State":
        return State(self.rows, self.cols, self.board[:], self.ball, self.player, self.jumping, self.winner,
                     self.turns, self.seq[:])

    @property
    def done(self) -> bool:
        return self.winner != 0 or self.draw

    draw: bool = False


def new_game(rows: int = 21, cols: int = 15) -> State:
    b = [EMPTY] * (rows * cols)
    for c in range(cols):
        b[c] = END_HI; b[cols + c] = END_HI
        b[(rows - 2) * cols + c] = END_LO; b[(rows - 1) * cols + c] = END_LO
    ball = (rows // 2) * cols + cols // 2
    b[ball] = BALL
    return State(rows, cols, b, ball)


def winner_at(rows: int, r: int) -> int:
    return 1 if r <= 1 else (2 if r >= rows - 2 else 0)


def jump_landings(board: list, ball: int, rows: int, cols: int) -> list[tuple[int, tuple]]:
    """[(landing index, jumped indices)] for the ball's single jumps from this board."""
    out = []
    br, bc = divmod(ball, cols)
    for dr, dc in DIRS:
        r, c = br + dr, bc + dc; jumped = []
        while 0 <= r < rows and 0 <= c < cols and board[r * cols + c] == MAN:
            jumped.append(r * cols + c); r += dr; c += dc
        if jumped and 0 <= r < rows and 0 <= c < cols and board[r * cols + c] != BALL:
            out.append((r * cols + c, tuple(jumped)))
    return out


def legal_actions(s: State) -> list[int]:
    n = s.rows * s.cols
    acts = []
    if not s.jumping:
        for i in range(s.cols, n - s.cols):       # rows 1..R-2
            if s.board[i] != BALL and s.board[i] != MAN:
                acts.append(i)
    acts += [n + land for land, _ in jump_landings(s.board, s.ball, s.rows, s.cols)]
    if s.jumping:
        acts.append(2 * n)
    return acts


def step(s: State, a: int, max_turns: int = 7200) -> State:
    """Return the next state (s is not modified)."""
    s = s.copy(); n = s.rows * s.cols
    if a < n:                                      # placement
        s.board[a] = MAN; s.player = 3 - s.player; s.turns += 1; s.seq = []
    elif a < 2 * n:                                # jump
        land = a - n
        lands = dict(jump_landings(s.board, s.ball, s.rows, s.cols))
        if land not in lands:
            raise ValueError("illegal jump")
        if not s.jumping:
            s.seq = [s.ball]
        s.board[s.ball] = EMPTY
        for j in lands[land]:
            s.board[j] = EMPTY
        s.board[land] = BALL; s.ball = land; s.jumping = True; s.seq.append(land)
        s.winner = winner_at(s.rows, land // s.cols)
        return s
    else:                                          # halt
        if not s.jumping:
            raise ValueError("halt without jump")
        s.jumping = False; s.player = 3 - s.player; s.turns += 1; s.seq = []
    if max_turns > 0 and s.turns >= max_turns and not s.winner:
        s.draw = True
    return s
