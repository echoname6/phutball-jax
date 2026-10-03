"""Phutball positions as text for language models, and engine-checked answers.

Board: row 0 at the top. Player 1 scores by ending a jump in rows 0-1 (the top goal), Player 2 in the last two rows
(the bottom goal). Squares are named column letter + row number, e.g. h10. Ball '@', men 'O', empty '.'.

Answers (the last line of the reply):  ANSWER: JUMP h8 h6 f4   |   ANSWER: NO WIN   |   ANSWER: PLACE e7
A jump is named by its landing square (from a given ball position each landing square is reached by exactly one
direction), so a turn of jumps is the list of landing squares.
"""
from __future__ import annotations

import re

from expert.engine import BALL, MAN, State, jump_landings, winner_at

COLS = "abcdefghijklmnopqrstuvwxyz"


def sq(i: int, cols: int) -> str:
    r, c = divmod(i, cols)
    return f"{COLS[c]}{r}"


def parse_sq(t: str, rows: int, cols: int) -> int | None:
    m = re.fullmatch(r"([a-z])(\d{1,2})", t.strip().lower())
    if not m: return None
    c, r = COLS.index(m.group(1)), int(m.group(2))
    if c >= cols or r >= rows: return None
    return r * cols + c


RULES = """Phutball (Philosopher's Football) rules:
- The board has {rows} rows (0 at the top) and {cols} columns (a-{lastc}). There is one ball (@) and any number of men (O).
- Player 1 wins by ending a jump with the ball in row 0 or 1 (the top goal). Player 2 wins by ending a jump with the ball in row {g2a} or {g2b} (the bottom goal).
- On a turn a player EITHER places one man on any empty square in rows 1 to {g2a} (this ends the turn), OR jumps the ball.
- A jump: the ball moves in a straight line (any of the 8 directions) over one or more men that are directly adjacent and contiguous in that line, and lands on the first square beyond them that is not a man (it must be on the board). Every man jumped over is removed.
- After a jump the same player may jump again from the ball's new square, as many times as they like, or stop.
- The game ends immediately when the ball lands in a goal: in the top goal Player 1 wins, in the bottom goal Player 2 wins, whoever made the jump."""


def render(s: State) -> str:
    rows, cols = s.rows, s.cols
    head = "     " + " ".join(COLS[:cols])
    lines = [head]
    for r in range(rows):
        cells = []
        for c in range(cols):
            v = s.board[r * cols + c]
            cells.append("@" if r * cols + c == s.ball else "O" if v == MAN else ".")
        tag = "   <- top goal (Player 1 scores here)" if r <= 1 else "   <- bottom goal (Player 2 scores here)" if r >= rows - 2 else ""
        lines.append(f"{r:>3}  " + " ".join(cells) + tag)
    return "\n".join(lines)


def first_jumps_text(s: State) -> str:
    js = jump_landings(s.board, s.ball, s.rows, s.cols)
    if not js: return "The ball has no legal jump from its square."
    parts = [f"{sq(land, s.cols)} (over {', '.join(sq(j, s.cols) for j in jumped)})" for land, jumped in js]
    return "Legal first jumps from the ball's square: " + "; ".join(parts) + "."


def rules(s: State) -> str:
    return RULES.format(rows=s.rows, cols=s.cols, lastc=COLS[s.cols - 1], g2a=s.rows - 2, g2b=s.rows - 1)


def position_text(s: State) -> str:
    return (f"{rules(s)}\n\nPosition (Player {s.player} to move; the ball is at {sq(s.ball, s.cols)}):\n\n{render(s)}\n\n"
            f"{first_jumps_text(s)}")


PROMPTS = {
    "win": ("{pos}\n\nQuestion: can Player {p} win THIS turn by jumping? If yes, give one complete winning sequence of "
            "jumps as its landing squares, in order. If no sequence of jumps wins this turn, say so.\n"
            "End your reply with one line, exactly in one of these forms:\nANSWER: JUMP <square> <square> ...\nANSWER: NO WIN"),
    "block": ("{pos}\n\nQuestion: it is Player {p}'s turn. Player {o} threatens to win on their next turn by jumping, and "
              "Player {p} has no jump that defends. Find a placement for Player {p} after which Player {o} has no winning "
              "sequence of jumps.\nEnd your reply with one line, exactly:\nANSWER: PLACE <square>"),
    "prevent": ("{pos}\n\nQuestion: it is Player {p}'s turn. If Player {p} does nothing useful, Player {o} can place a man "
                "that creates an unstoppable threat (a win next turn whatever Player {p} does). Find a move for Player {p} after "
                "which Player {o} has neither a winning sequence of jumps nor such an unstoppable placement. The move may be a "
                "placement, or a sequence of jumps (the turn ends where your sequence stops).\nEnd your reply with one line, "
                "exactly in one of these forms:\nANSWER: PLACE <square>\nANSWER: JUMP <square> <square> ..."),
    "forced": ("{pos}\n\nQuestion: Player {p} cannot win this turn. Find a placement after which Player {p} is guaranteed to "
               "win on their next turn, whatever the opponent does in between (the opponent may place a man anywhere or "
               "jump the ball).\nEnd your reply with one line, exactly:\nANSWER: PLACE <square>"),
}


def prompt(task: str, s: State) -> str:
    return PROMPTS[task].format(pos=position_text(s), p=s.player, o=3 - s.player)


ANS_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


def parse_answer(reply: str):
    """('jump', [squares]) | ('nowin', None) | ('place', square) | (None, None)."""
    m = None
    for m in ANS_RE.finditer(reply or ""): pass
    if not m: return None, None
    body = m.group(1).strip().strip("`*. ").lower()
    if body.startswith("no win") or body.startswith("nowin"): return "nowin", None
    toks = re.findall(r"[a-z]\d{1,2}", body)
    if body.startswith("jump"): return "jump", toks
    if body.startswith("place") and toks: return "place", toks[0]
    return None, None


def replay_jumps(s: State, squares: list[str]):
    """Replay a turn of jumps that may stop anywhere. Returns (board, ball, winner_or_0, error)."""
    rows, cols = s.rows, s.cols; board, ball = s.board[:], s.ball
    for k, t in enumerate(squares):
        land = parse_sq(t, rows, cols)
        if land is None: return None, None, 0, f"bad square {t!r}"
        jl = dict(jump_landings(board, ball, rows, cols))
        if land not in jl: return None, None, 0, f"illegal jump {k + 1} to {t}"
        board[ball] = 0
        for j in jl[land]: board[j] = 0
        board[land] = BALL; ball = land
        w = winner_at(rows, land // cols)
        if w: return board, ball, w, None
    return board, ball, 0, None


def check_win_sequence(s: State, squares: list[str]):
    """Replay a turn of jumps. Returns (wins, reason)."""
    rows, cols, me = s.rows, s.cols, s.player
    board, ball = s.board[:], s.ball
    for k, t in enumerate(squares):
        land = parse_sq(t, rows, cols)
        if land is None: return False, f"bad square {t!r}"
        jl = dict(jump_landings(board, ball, rows, cols))
        if land not in jl: return False, f"illegal jump {k + 1} to {t}"
        board[ball] = 0
        for j in jl[land]: board[j] = 0
        board[land] = BALL; ball = land
        w = winner_at(rows, land // cols)
        if w: return (w == me, "wins" if w == me else "lands in the opponent's goal") if k == len(squares) - 1 else \
            ((w == me), "wins (extra jumps after the win ignored)" if w == me else "lands in the opponent's goal")
    return False, "the sequence ends without reaching the goal"
