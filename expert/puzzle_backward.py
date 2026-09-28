"""'Jump backwards to go forwards' puzzles, built on the fast engine.

A puzzle is kept only if the side to move can win this turn AND no winning chain is monotone forward (every jump moving
toward the goal): at least one sideways or backward jump is REQUIRED, so "always jump toward the goal" can never solve it.

Construction (for player 1, whose goal is rows 0-1; player 2 puzzles are the 180-degree rotation):
  1. choose a leg sequence of J jumps with >= 1 non-forward leg (sideways: dr = 0, backward: dr > 0); the last leg is
     forward into the goal; each leg jumps over L contiguous stones;
  2. walk it from a random start, placing the stones; intermediate landings must stay out of both end zones (any
     landing there ends the game) and every square used must be free;
  3. optional noise stones (random) or real clutter (patterns from expert games);
  4. verify with the engine (chains): a win exists and none is all-forward; else retry.
"""
from __future__ import annotations

import random

from expert.engine import BALL, EMPTY, END_HI, END_LO, MAN, State, new_game
from expert.oracle import chains

FWD = [(-1, -1), (-1, 0), (-1, 1)]; SIDE = [(0, -1), (0, 1)]; BACK = [(1, -1), (1, 0), (1, 1)]


def _is_forward_chain(start: int, path, cols: int) -> bool:
    prev = start
    for land in path:
        if land // cols >= prev // cols: return False          # not strictly toward row 0
        prev = land
    return True


def needs_non_forward(s: State) -> bool | None:
    """True if a win exists and every winning chain has a non-forward jump; False if some winning chain is all-forward;
    None if there is no win. (Player-1 orientation.)"""
    wins = [p for p, nb, nbl, w in chains(s.board, s.ball, s.rows, s.cols, 20000) if w == 1]
    if not wins: return None
    return not any(_is_forward_chain(s.ball, p, s.cols) for p in wins)


def rotate180(s: State) -> State:
    n = s.rows * s.cols; b = s.board[::-1]
    b = [(-v if v in (END_HI, END_LO) else v) for v in b]     # end-zone markers swap sides
    return State(s.rows, s.cols, b, n - 1 - s.ball, 2)


def greedy_forward_solves(s: State, tie: int = 0) -> bool:
    """Does 'always take the jump that lands closest to my goal' win this turn? (the trap test). tie=0/1 breaks ties
    between equally close landings by lowest/highest square index."""
    from expert.engine import jump_landings, step
    me = s.player; cur = s
    for _ in range(40):
        js = jump_landings(cur.board, cur.ball, cur.rows, cur.cols)
        if not js: return False
        key = (lambda l: (l // cur.cols if me == 1 else -(l // cur.cols), l if tie == 0 else -l))
        cur = step(cur, cur.rows * cur.cols + min((l for l, _ in js), key=key))
        if cur.winner: return cur.winner == me
    return False


def generate(rng: random.Random, J: int, L: int, rows=21, cols=15, noise=0, clutter=None, max_tries=400,
             decoys: bool = True) -> State | None:
    """decoys: at every takeoff point where the chain needs a sideways/backward jump, add 1-2 stones giving a tempting
    FORWARD jump that dead-ends; the puzzle is then kept only if 'jump toward the goal' fails on it."""
    for _ in range(max_tries):
        legs = [rng.choice(FWD + SIDE + BACK) for _ in range(J - 1)] + [rng.choice(FWD)]
        if all(d in FWD for d in legs): legs[rng.randrange(J - 1)] = rng.choice(SIDE + BACK)
        s = new_game(rows, cols); b = s.board; b[s.ball] = EMPTY
        # start low enough that the chain can reach the goal
        span_up = sum(L + 1 for d in legs if d[0] < 0) - sum(L + 1 for d in legs if d[0] > 0)
        lo, hi = max(3, span_up - (L + 1) + 2), min(rows - 4, span_up + 6)
        if lo > hi: continue
        r0 = rng.randint(lo, hi); c0 = rng.randrange(cols)
        r, c = r0, c0; used = {(r, c)}; ok = True; takeoffs = []
        for k, (dr, dc) in enumerate(legs):
            if (dr, dc) not in FWD: takeoffs.append((r, c))
            for i in range(1, L + 1):
                rr, cc = r + dr * i, c + dc * i
                if not (2 <= rr <= rows - 3 and 0 <= cc < cols) or (rr, cc) in used: ok = False; break
                used.add((rr, cc)); b[rr * cols + cc] = MAN
            if not ok: break
            r, c = r + dr * (L + 1), c + dc * (L + 1)
            last = k == len(legs) - 1
            if not (0 <= c < cols) or (r, c) in used: ok = False; break
            if last:
                if not (0 <= r <= 1): ok = False; break              # must land in the goal
            elif not (2 <= r <= rows - 3): ok = False; break          # intermediate landings outside both end zones
            used.add((r, c))
        if not ok: continue
        b[r0 * cols + c0] = BALL; s.ball = r0 * cols + c0; s.player = 1
        if decoys:
            for (tr, tc) in takeoffs:                                 # tempting forward jumps that go nowhere
                for _ in range(8):
                    dr, dc = rng.choice(FWD); m = rng.randint(1, 2)
                    sq = [(tr + dr * i, tc + dc * i) for i in range(1, m + 1)]; land = (tr + dr * (m + 1), tc + dc * (m + 1))
                    if all(2 <= q[0] <= rows - 3 and 0 <= q[1] < cols and q not in used for q in sq) and \
                            2 <= land[0] <= rows - 3 and 0 <= land[1] < cols and land not in used:
                        for q in sq: b[q[0] * cols + q[1]] = MAN; used.add(q)
                        used.add(land); break
        if noise:
            free = [i for i in range(cols, rows * cols - cols) if b[i] == EMPTY and divmod(i, cols) not in used]
            for i in rng.sample(free, min(noise, len(free))): b[i] = MAN
        if clutter is not None:
            cand = [i for i in range(cols, rows * cols - cols) if clutter[i] and b[i] == EMPTY and divmod(i, cols) not in used]
            rng.shuffle(cand)
            for i in cand[: rng.randint(4, 24)]: b[i] = MAN
        if needs_non_forward(s) and (not decoys or not (greedy_forward_solves(s, 0) or greedy_forward_solves(s, 1))):
            return s if rng.random() < 0.5 else rotate180(s)
    return None
