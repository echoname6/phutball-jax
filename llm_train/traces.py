"""Engine-derived reasoning traces for the phutball puzzle tasks (stage 1 of the LLM curriculum).

Each trace is a mechanical procedure written from the engine's own search, so every claim in it is computed, and its
length follows the search the position needs (an easy position gives a short trace, a hard one a long trace):
  win      depth-first search over the jump tree, jumps toward the goal first; dead ends, jumps into the opponent's goal
           and repeated positions are noted; stop at the first winning sequence, or exhaust the tree: NO WIN.
  block    the attacker's winning sequences; a placement can only break a sequence by occupying one of its landing
           squares, so those are the candidates, most-used first; each failure is refuted with a sequence that still
           wins; the first square that leaves no winning sequence is the answer.
  forced   placements that give a winning sequence, those with the most winning sequences first; each refuted by the
           reply that stops it (a jump sequence, or a placement on a landing square); for the answer, why no reply works.
  prevent  the attacker's unstoppable placements; candidates: squares on those threats (most-used first), then squares
           next to the ball (they give an escape jump); each failure refuted with the threat that survives; the first
           move after which the attacker has neither a winning sequence nor an unstoppable placement. If no placement
           saves, the jump sequences that do.
Every function returns (thinking text, answer line) or None when the search exceeds its cap (the example is skipped;
the builder counts skips so the length distribution is not silently truncated).
"""
from __future__ import annotations

from collections import Counter

from expert.engine import BALL, EMPTY, MAN, State, jump_landings, winner_at
from expert.oracle import chains
from expert.puzzle_place import CapHit, attacker_threatens, best_completions, unstoppable
from llm_bench.text import sq

MAX_WIN_NODES = 600         # jump-tree nodes in a win trace (beyond this the example is skipped and counted)
MAX_CANDIDATES = 30         # candidate moves in a placement trace
SHOW_SEQS = 4               # winning sequences listed when describing a threat


def goal_text(rows: int, player: int) -> str:
    return "rows 0-1 (the top)" if player == 1 else f"rows {rows - 2}-{rows - 1} (the bottom)"


def seq_text(path, cols) -> str:
    return " ".join(sq(x, cols) for x in path)


def interior(x: int, rows: int, cols: int) -> bool:
    return cols <= x < rows * cols - cols                               # placeable rows 1 .. rows-2


def win_paths(board, ball, rows, cols, player, cap=20000) -> list[tuple]:
    return [p for p, nb, nbl, w in chains(board, ball, rows, cols, cap) if w == player]


def seqs_text(paths, cols) -> str:
    shown = sorted(paths, key=lambda p: (len(p), p))[:SHOW_SEQS]
    more = len(paths) - len(shown)
    return "; ".join(seq_text(p, cols) for p in shown) + (f" (and {more} more)" if more > 0 else "")


def by_use(paths, board, rows, cols) -> list[int]:
    """Empty interior squares the paths land on, most-used first (ties: earlier in a sequence, then square index)."""
    use, first = Counter(), {}
    for p in paths:
        for k, x in enumerate(p):
            if interior(x, rows, cols) and board[x] not in (BALL, MAN):
                use[x] += 1; first[x] = min(first.get(x, 99), k)
    return sorted(use, key=lambda x: (-use[x], first[x], x))


def neighbours(ball, rows, cols) -> list[int]:
    r, c = divmod(ball, cols)
    return [(r + dr) * cols + c + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols]


# --------------------------------------------------------------------------------------------------------------
def trace_win(s: State, max_nodes: int = MAX_WIN_NODES):
    rows, cols, me = s.rows, s.cols, s.player; opp = 3 - me
    lines = [f"Player {me} needs the ball to end a jump in {goal_text(rows, me)}. I search the jump sequences from "
             f"{sq(s.ball, cols)}, jumps toward the goal first."]
    seen = set(); nodes = [0]; found = []
    toward = (lambda x: x // cols) if me == 1 else (lambda x: rows - 1 - x // cols)

    def dfs(board, ball, removed, path, depth):
        nodes[0] += 1
        if nodes[0] > max_nodes: raise CapHit("trace")
        ind = "  " * depth
        js = sorted(jump_landings(board, ball, rows, cols), key=lambda t: (toward(t[0]), t[0]))
        if not js:
            lines.append(f"{ind}From {sq(ball, cols)}: no jumps. Dead end."); return False
        lines.append(f"{ind}From {sq(ball, cols)}: " + "; ".join(
            f"{sq(l, cols)} (over {', '.join(sq(j, cols) for j in jd)})" for l, jd in js) + ".")
        for land, jumped in js:
            w = winner_at(rows, land // cols)
            if w == opp:
                lines.append(f"{ind}- {sq(land, cols)} is in Player {opp}'s goal: that loses."); continue
            if w == me:
                lines.append(f"{ind}- {sq(land, cols)} is in row {land // cols}, my goal: win.")
                found.extend(path + [land]); return True
            key = (land, removed | frozenset(jumped))
            if key in seen:
                lines.append(f"{ind}- {sq(land, cols)}: same position as before, already searched."); continue
            seen.add(key)
            nb = board[:]; nb[ball] = EMPTY
            for j in jumped: nb[j] = EMPTY
            nb[land] = BALL
            lines.append(f"{ind}- {sq(land, cols)}:")
            if dfs(nb, land, key[1], path + [land], depth + 1): return True
        return False

    try:
        won = dfs(s.board[:], s.ball, frozenset(), [], 0)
    except CapHit:
        return None
    if won:
        seq = seq_text(found, cols)
        lines.append(f"Winning sequence: {seq}.")
        return "\n".join(lines), f"ANSWER: JUMP {seq}"
    lines.append("Every jump sequence is searched and none ends in my goal, so there is no win this turn.")
    return "\n".join(lines), "ANSWER: NO WIN"


# --------------------------------------------------------------------------------------------------------------
def trace_block(s: State, good: set[int]):
    rows, cols, me = s.rows, s.cols, s.player; att = 3 - me
    wins = win_paths(s.board, s.ball, rows, cols, att)
    if not wins: return None
    cands = by_use(wins, s.board, rows, cols)
    lines = [f"Player {att} wins next turn with {len(wins)} jump sequence{'s' * (len(wins) > 1)}: {seqs_text(wins, cols)}.",
             "My jumps do not defend, so I place a man. A man can only break a sequence by occupying a square it lands on. "
             "Candidates, most-used first: " + ", ".join(sq(x, cols) for x in cands[:12]) + ("..." if len(cands) > 12 else "") + "."]
    for p in cands[:MAX_CANDIDATES]:
        nb = s.board[:]; nb[p] = MAN
        rest = win_paths(nb, s.ball, rows, cols, att)
        if not rest:
            if p not in good: return None                               # disagrees with the generator's labels
            lines.append(f"{sq(p, cols)}: Player {att} has no winning sequence left. This blocks.")
            return "\n".join(lines), f"ANSWER: PLACE {sq(p, cols)}"
        lines.append(f"{sq(p, cols)}: still wins with {seq_text(min(rest, key=len), cols)}.")
    return None


# --------------------------------------------------------------------------------------------------------------
def refute_threat(s: State, p: int):
    """Why placing p does not force a win for s.player (text), or None if it does (same checks as unstoppable())."""
    rows, cols, me = s.rows, s.cols, s.player; opp = 3 - me
    nb = s.board[:]; nb[p] = MAN
    mine = win_paths(nb, s.ball, rows, cols, me)
    if not mine: return "it gives me no winning sequence"
    for path, b2, bl2, w in chains(nb, s.ball, rows, cols, 20000):
        if w == opp: return f"Player {opp} jumps {seq_text(path, cols)} into their goal and wins"
        if w: continue
        if not win_paths(b2, bl2, rows, cols, me, 4000):
            return f"Player {opp} jumps {seq_text(path, cols)} and I have no winning sequence from {sq(bl2, cols)}"
    for x in by_use(mine, nb, rows, cols):
        b3 = nb[:]; b3[x] = MAN
        if not win_paths(b3, s.ball, rows, cols, me, 4000):
            return f"Player {opp} places {sq(x, cols)} and none of my sequences works"
    return None


def trace_forced(s: State, good: set[int], threats: list[int]):
    rows, cols, me = s.rows, s.cols, s.player; opp = 3 - me
    mine = {}
    for p in threats:
        nb = s.board[:]; nb[p] = MAN
        w = win_paths(nb, s.ball, rows, cols, me)
        if w: mine[p] = w
    if not mine: return None
    order = sorted(mine, key=lambda p: (-len(mine[p]), p))
    lines = [f"I have no winning jump now. A placement must give me a win that Player {opp} cannot stop with one move.",
             "Placements that give me a winning sequence, most sequences first: " +
             "; ".join(f"{sq(p, cols)} ({len(mine[p])}: {seq_text(min(mine[p], key=len), cols)})" for p in order[:10]) +
             ("..." if len(order) > 10 else "") + ".",
             f"For each, Player {opp} can reply by jumping, or by placing a man on a square my sequences land on."]
    for p in order[:MAX_CANDIDATES]:
        why = refute_threat(s, p)
        if why is None:
            if p not in good: return None
            nb = s.board[:]; nb[p] = MAN
            jr = [(path, b2, bl2) for path, b2, bl2, w in chains(nb, s.ball, rows, cols, 20000) if not w]
            parts = [f"{sq(p, cols)}: my sequences: {seqs_text(mine[p], cols)}."]
            if not jr: parts.append(f"Player {opp} has no jump reply.")
            elif len(jr) <= 4:
                parts.append("Jump replies: " + "; ".join(
                    f"{seq_text(path, cols)} -> I still win with {seq_text(min(win_paths(b2, bl2, rows, cols, me, 4000), key=len), cols)}"
                    for path, b2, bl2 in jr) + ".")
            else: parts.append(f"Each of Player {opp}'s {len(jr)} jump replies still leaves me a winning sequence.")
            lands = by_use(mine[p], nb, rows, cols)
            if lands:
                rep = []
                for x in lands:
                    b3 = nb[:]; b3[x] = MAN
                    rep.append(f"{sq(x, cols)} -> I still win with {seq_text(min(win_paths(b3, s.ball, rows, cols, me, 4000), key=len), cols)}")
                parts.append("Placement replies on my landing squares: " + "; ".join(rep) + ".")
            parts.append("Unstoppable.")
            lines.append(" ".join(parts))
            return "\n".join(lines), f"ANSWER: PLACE {sq(p, cols)}"
        lines.append(f"{sq(p, cols)}: no, {why}.")
    return None


# --------------------------------------------------------------------------------------------------------------
def attacker_unstoppables(t: State, cap: int = 6) -> list[tuple[int, tuple]]:
    """t: attacker to move. [(placement, a shortest winning path after it)] for unstoppable placements (up to cap)."""
    rows, cols = t.rows, t.cols; out = []
    for p in range(cols, rows * cols - cols):
        if t.board[p] in (BALL, MAN): continue
        nb = t.board[:]; nb[p] = MAN
        if best_completions(State(rows, cols, nb, t.ball, t.player), cap_nodes=20_000) and unstoppable(t, p):
            out.append((p, min(win_paths(nb, t.ball, rows, cols, t.player), key=len)))
            if len(out) >= cap: break
    return out


def why_not_saved(s: State, nb, ball, att) -> str:
    rows, cols = s.rows, s.cols
    w = win_paths(nb, ball, rows, cols, att)
    if w: return f"Player {att} wins at once with {seq_text(min(w, key=len), cols)}"
    left = attacker_unstoppables(State(rows, cols, nb, ball, att), cap=1)
    return f"Player {att} still has {sq(left[0][0], cols)} (then {seq_text(left[0][1], cols)})" if left else "a threat remains"


def trace_prevent(s: State, good_place: set[int], good_jump_first: set[int]):
    rows, cols, me = s.rows, s.cols, s.player; att = 3 - me
    threats = attacker_unstoppables(State(rows, cols, s.board[:], s.ball, att), cap=8)
    if not threats: return None
    lines = [f"Player {att} has no win now, but has unstoppable placements: " +
             "; ".join(f"{sq(p, cols)} (then {seq_text(w, cols)})" for p, w in threats[:SHOW_SEQS]) +
             (f" (and {len(threats) - SHOW_SEQS} more)" if len(threats) > SHOW_SEQS else "") + ".",
             f"I need a move after which Player {att} has neither a winning sequence nor an unstoppable placement. Candidates: "
             f"squares on those threats, then squares next to the ball (a man there gives me an escape jump later)."]
    paths = [(p,) + w for p, w in threats]
    cands = list(dict.fromkeys(by_use(paths, s.board, rows, cols) +
                               [x for x in neighbours(s.ball, rows, cols) if interior(x, rows, cols) and s.board[x] not in (BALL, MAN)]))
    for p in cands[:MAX_CANDIDATES]:
        nb = s.board[:]; nb[p] = MAN
        if not attacker_threatens(State(rows, cols, nb, s.ball, att)):
            if p not in good_place: return None
            lines.append(f"{sq(p, cols)}: Player {att} has neither a winning sequence nor an unstoppable placement. Saved.")
            return "\n".join(lines), f"ANSWER: PLACE {sq(p, cols)}"
        lines.append(f"{sq(p, cols)}: no, {why_not_saved(s, nb, s.ball, att)}.")
    if good_place: return None                                          # a saving placement exists beyond the candidates
    lines.append("No placement saves. I look for a jump sequence that leaves the ball safe.")
    for path, nb, nbl, w in sorted(chains(s.board, s.ball, rows, cols, 20000), key=lambda c: len(c[0])):
        if w: continue
        if path[0] in good_jump_first and not attacker_threatens(State(rows, cols, nb, nbl, att)):
            lines.append(f"Jump {seq_text(path, cols)} and stop: Player {att} then has neither a winning sequence nor an "
                         f"unstoppable placement. Saved.")
            return "\n".join(lines), f"ANSWER: JUMP {seq_text(path, cols)}"
        lines.append(f"Jump {seq_text(path, cols)}: no, {why_not_saved(s, nb, nbl, att)}.")
        if len(lines) > MAX_CANDIDATES + 4: return None
    return None
