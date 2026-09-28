"""Placement puzzles, built from the jump-chain puzzles on the fast engine.

  threat      Threat creation. Take a winning-chain puzzle and remove ONE stone from its shortest winning chain, choosing
              the jump (leg) UNIFORMLY first and then a stone within it, so the gap is as often next to the ball as next
              to the goal. Keep it if no win remains this turn. Target: every placement after which the side to move
              would have a winning chain (if it moved again), softmax-weighted toward shorter resulting wins.
              Value: unknown (the opponent may stop it) -> value weight 0.
  forced      Unstoppable threat (win in two). A threat position where some placement survives EVERY reply: each
              opponent jump chain (moves the ball; an opponent win fails it) and each opponent placement on a landing
              square of one of my winning chains (a placement anywhere else cannot break a chain). Target: only the
              surviving placements. Value: +1 (forced), value weight 1.
  block       Blocking. A winning-chain puzzle with the OTHER side to move; kept only if no defending JUMP chain exists
              (both sides jump the same ball), so a placement is the only defence and the labels are complete. Target: every placement that leaves the
              attacker without a winning chain. Value: unknown -> weight 0.

Training uses make_forced and make_block only. Stoppable threats are not trained: a threat the opponent can answer
may be a bad move, while an unstoppable one is a proven win. (Threat positions: ~3/4 have a forced win; make_threat
is kept as the building block and for analysis.)
"""
from __future__ import annotations

import math
import random

from expert.engine import BALL, EMPTY, MAN, State, jump_landings, step
from expert.oracle import chains as _chains
from expert.puzzle_data import CapHit, target_weights
from expert.puzzle_data import best_completions as _best_completions


# Every search here is strict: if it hits its cap its answer may be incomplete, so it raises CapHit and the puzzle
# builders (make_forced / make_block / make_prevent) drop that puzzle instead of trusting a partial answer.
CAP_DROPS = [0]


def best_completions(s: State, cap_nodes: int = 200_000) -> dict:
    return _best_completions(s, cap_nodes, strict=True)


def chains(board, ball, rows, cols, cap):
    out = _chains(board, ball, rows, cols, cap)
    if len(out) >= cap: raise CapHit("chains")
    return out


def _dropping_capped(fn):
    def wrap(*a, **k):
        try: return fn(*a, **k)
        except CapHit: CAP_DROPS[0] += 1; return None
    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


def _wins_for(board, ball, rows, cols, player, cap=20000):
    return [(p, nbl) for p, nb, nbl, w in chains(board, ball, rows, cols, cap) if w == player]


def shortest_chain_legs(s: State):
    """Legs of one shortest winning chain: [(landing, [jumped stones])]."""
    comp = best_completions(s)
    if not comp: return None
    legs = []; cur = s
    for _ in range(64):
        comp = best_completions(cur)
        if not comp: return None
        nxt = min(comp, key=lambda m: (comp[m][0], -comp[m][1], m))
        jumped = dict(jump_landings(cur.board, cur.ball, cur.rows, cur.cols))[nxt]
        legs.append((nxt, list(jumped))); cur = step(cur, cur.rows * cur.cols + nxt)
        if cur.winner: return legs
    return None


def threat_placements(s: State, temp: float = 0.5):
    """{placement square: weight} over placements that give the side to move a winning chain next (as if moving again)."""
    rows, cols, me = s.rows, s.cols, s.player; best = {}
    for i in range(cols, rows * cols - cols):
        if s.board[i] in (BALL, MAN): continue                        # goal rows 1 / R-2 hold zone markers but are placeable
        nb = s.board[:]; nb[i] = MAN; t = State(rows, cols, nb, s.ball, me)
        comp = best_completions(t, cap_nodes=20_000)
        if comp: best[i] = (min(v[0] for v in comp.values()), 0)
    return target_weights(best, temp, 0.0)


def unstoppable(s: State, place: int) -> bool:
    rows, cols, me = s.rows, s.cols, s.player; opp = 3 - me
    nb = s.board[:]; nb[place] = MAN
    mine = _wins_for(nb, s.ball, rows, cols, me)
    if not mine: return False
    for path, b2, bl2, w in chains(nb, s.ball, rows, cols, 20000):      # opponent jump replies (they move the ball)
        if w == opp: return False
        if w: continue
        if not _wins_for(b2, bl2, rows, cols, me, 4000): return False
    landing_sq = {l for p, _ in mine for l in p}                          # placements that could break a chain
    for sq in landing_sq:
        if not (cols <= sq < rows * cols - cols) or nb[sq] in (BALL, MAN): continue
        b3 = nb[:]; b3[sq] = MAN
        if not _wins_for(b3, s.ball, rows, cols, me, 4000): return False
    return True


def make_threat(base: State, rng: random.Random):
    legs = shortest_chain_legs(base)
    if not legs: return None
    order = list(range(len(legs))); rng.shuffle(order)
    for li in order:                                                      # leg chosen uniformly, then a stone in it
        stones = legs[li][1][:]; rng.shuffle(stones)
        for st in stones:
            b = base.board[:]; b[st] = EMPTY; s = State(base.rows, base.cols, b, base.ball, base.player)
            if best_completions(s): continue                              # still a win this turn: not a threat puzzle
            tw = threat_placements(s)
            if tw: return s, tw, {"removed": st, "leg": li, "legs": len(legs)}
    return None


@_dropping_capped
def make_forced(base: State, rng: random.Random):
    r = make_threat(base, rng)
    if not r: return None
    s, tw, meta = r
    good = [p for p in tw if unstoppable(s, p)]
    if not good: return None
    return s, {p: 1.0 / len(good) for p in good}, {**meta, "threats": len(tw), "forced": len(good)}


@_dropping_capped
def make_block(base: State):
    """The attacker (base.player) threatens a win; the defender moves. Both sides jump the same ball, so the defender
    could also defend by jumping: keep the puzzle only if NO defending jump chain exists (every chain either lands in
    the attacker's goal or leaves the attacker a win), so a placement is the only defence and the labels are complete."""
    rows, cols = base.rows, base.cols; att = base.player; dfd = 3 - att
    s = State(rows, cols, base.board[:], base.ball, dfd)
    if not _wins_for(s.board, s.ball, rows, cols, att): return None
    for path, nb, nbl, w in chains(s.board, s.ball, rows, cols, 20000):
        if w == dfd: return None                                          # the defender can simply win
        if w: continue                                                    # into the attacker's goal: a loss
        if not _wins_for(nb, nbl, rows, cols, att, 4000): return None     # a jump defends: not a pure placement puzzle
    good = []
    for i in range(cols, rows * cols - cols):
        if s.board[i] in (BALL, MAN): continue                        # goal rows 1 / R-2 hold zone markers but are placeable
        nb = s.board[:]; nb[i] = MAN
        if not _wins_for(nb, s.ball, rows, cols, att, 4000): good.append(i)
    if not good: return None
    return s, {p: 1.0 / len(good) for p in good}, {"blocks": len(good)}


def attacker_threatens(t: State) -> bool:
    """t: attacker to move. True if the attacker wins this turn or has an unstoppable placement (win in two)."""
    rows, cols = t.rows, t.cols
    if _wins_for(t.board, t.ball, rows, cols, t.player): return True
    for p in range(cols, rows * cols - cols):
        if t.board[p] in (BALL, MAN): continue
        nb = t.board[:]; nb[p] = MAN
        if best_completions(State(rows, cols, nb, t.ball, t.player), cap_nodes=20_000) and unstoppable(t, p): return True
    return False


@_dropping_capped
def make_prevent(base: State, rng: random.Random):
    """Denial, engine-verified. From an unstoppable-threat position, give the move to the DEFENDER one turn early.
    Target: every defender move (placement, or the halting point of a jump chain, labelled by its first landing)
    after which the attacker has neither a winning chain nor an unstoppable placement. Kept only if the defender
    cannot win outright, some move saves, and not every move does. Value: unknown -> weight 0.
    Returns (state, {placement square: w}, {first jump landing: w}, meta)."""
    r = make_forced(base, rng)
    if not r: return None
    s_att, _, meta = r; rows, cols = s_att.rows, s_att.cols; att = s_att.player; dfd = 3 - att
    s = State(rows, cols, s_att.board[:], s_att.ball, dfd)
    place_ok, jump_ok, total = [], {}, 0
    for p in range(cols, rows * cols - cols):
        if s.board[p] in (BALL, MAN): continue
        nb = s.board[:]; nb[p] = MAN; total += 1
        if not attacker_threatens(State(rows, cols, nb, s.ball, att)): place_ok.append(p)
    for path, nb, nbl, w in chains(s.board, s.ball, rows, cols, 20000):
        if w == dfd: return None                                          # the defender just wins
        total += 1
        if w or attacker_threatens(State(rows, cols, nb, nbl, att)): continue
        jump_ok.setdefault(path[0], []).append(path)                      # a saving chain starts with this jump
    good = len(place_ok) + sum(len(v) for v in jump_ok.values())
    if good == 0 or good == total: return None
    k = len(place_ok) + len(jump_ok)
    return s, {p: 1.0 / k for p in place_ok}, {l: 1.0 / k for l in jump_ok}, \
        {**meta, "saving_placements": len(place_ok), "saving_first_jumps": len(jump_ok), "moves": total}
