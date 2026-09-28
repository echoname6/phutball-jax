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

Each record: (state, {placement square: weight}, value, value_weight, meta).
"""
from __future__ import annotations

import math
import random

from expert.engine import BALL, EMPTY, MAN, State, jump_landings, step
from expert.oracle import chains
from expert.puzzle_data import best_completions, target_weights


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


def make_forced(base: State, rng: random.Random):
    r = make_threat(base, rng)
    if not r: return None
    s, tw, meta = r
    good = [p for p in tw if unstoppable(s, p)]
    if not good: return None
    return s, {p: 1.0 / len(good) for p in good}, {**meta, "threats": len(tw), "forced": len(good)}


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
