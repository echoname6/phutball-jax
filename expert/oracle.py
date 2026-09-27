"""Lookahead expert for Phutball (a deterministic teacher for DAgger / pretraining).

Decision = 2-ply search over a pruned move set:
  my candidates:  every jump chain I can make (any halting point), plus placements from a
                  small set of plausible patterns (below);
  their replies:  every jump chain they can make, plus one "quiet" reply (a placement,
                  scored statically);
  leaf score:     ball progress toward my goal, my best jump threat when it is my move,
                  and +/-WIN for positions where the side to move can win by jumping.

Placement patterns ("plausible and efficient"):
  - ring 1 and ring 2 around the ball (creates or extends jumps next turn);
  - the landing squares of my current single jumps and the squares just past my chain
    endpoints (lengthen / continue a chain);
  - the landing squares of the opponent's best chains (redirect or spoil them).
Ties go to fewer stones and to squares nearer the ball.

Stateless: the action depends only on the position, so it can label any state (DAgger).
"""
from __future__ import annotations

from functools import lru_cache

from expert.engine import BALL, MAN, EMPTY, DIRS, State, jump_landings, winner_at

WIN = 1000.0


def goal_dist(rows: int, ball: int, cols: int, player: int) -> int:
    r = ball // cols
    return r - 1 if player == 1 else (rows - 2) - r


def chains(board: list, ball: int, rows: int, cols: int, cap: int = 4000):
    """All halting points reachable by jumping from `ball` (at least one jump).
    Returns [(path tuple of landings, final board, final ball, winner)]; a winning landing ends the chain."""
    out, seen = [], set()
    stack = [((), board, ball, frozenset())]
    while stack and len(out) < cap:
        path, b, bl, removed = stack.pop()
        for land, jumped in jump_landings(b, bl, rows, cols):
            nrem = removed | frozenset(jumped)
            key = (land, nrem)
            if key in seen:
                continue
            seen.add(key)
            nb = b[:]; nb[bl] = EMPTY
            for j in jumped: nb[j] = EMPTY
            nb[land] = BALL
            w = winner_at(rows, land // cols)
            npath = path + (land,)
            out.append((npath, nb, land, w))
            if not w:
                stack.append((npath, nb, land, nrem))
    return out


class Expert:
    def __init__(self, threat_w: float = 0.6, quiet_penalty: float = 0.15, max_chain_cap: int = 4000,
                 reply_cap: int = 600, leaf_cap: int = 300, budget: int = 300_000):
        # Work limits (crowded boards made the 3-level search explode: one game in ~24 took > 20 s):
        #   cap        my own chains at the root;  reply_cap  the opponent's replies;  leaf_cap  the leaf threat estimate;
        #   budget     chains visited per decision (deterministic, so labels stay reproducible); candidates are
        #              tried most-promising first, and once the budget is spent the best so far is returned.
        self.threat_w, self.quiet_penalty, self.cap = threat_w, quiet_penalty, max_chain_cap
        self.reply_cap, self.leaf_cap, self.budget = reply_cap, leaf_cap, budget
        self._work = 0

    def _chains(self, board, ball, rows, cols, cap):
        out = chains(board, ball, rows, cols, cap); self._work += len(out) + 1
        return out

    # --- scoring -----------------------------------------------------------------
    def _best_chain(self, board, ball, rows, cols, player):
        """(can_win, best progress gain) for `player` to move now."""
        d0 = goal_dist(rows, ball, cols, player); best = 0.0
        for path, nb, nbl, w in self._chains(board, ball, rows, cols, self.leaf_cap):
            if w == player:
                return True, float(d0)
            if w:                                   # lands in the other goal: never chosen
                continue
            best = max(best, d0 - goal_dist(rows, nbl, cols, player))
        return False, best

    def leaf(self, board, ball, rows, cols, me, to_move) -> float:
        """Static score from `me`'s point of view with `to_move` about to play."""
        prog = -goal_dist(rows, ball, cols, me)
        win, gain = self._best_chain(board, ball, rows, cols, to_move)
        if win:
            return WIN if to_move == me else -WIN
        return prog + (self.threat_w * gain if to_move == me else -self.threat_w * gain)

    def after_my_move(self, board, ball, rows, cols, me) -> float:
        """2-ply: the opponent replies with its best jump chain or a quiet placement."""
        opp = 3 - me
        worst = self.leaf(board, ball, rows, cols, me, me) - self.quiet_penalty   # quiet reply
        for path, nb, nbl, w in self._chains(board, ball, rows, cols, self.reply_cap):
            if w == opp:
                return -WIN
            if w == me:
                continue                           # they will not jump into my goal
            worst = min(worst, self.leaf(nb, nbl, rows, cols, me, me))
        return worst

    # --- candidates ----------------------------------------------------------------
    def placement_candidates(self, s: State) -> list[int]:
        rows, cols, b = s.rows, s.cols, s.board
        br, bc = divmod(s.ball, cols); cand = set()
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                r, c = br + dr, bc + dc
                if (dr or dc) and 1 <= r <= rows - 2 and 0 <= c < cols:
                    cand.add(r * cols + c)
        for land, _ in jump_landings(b, s.ball, rows, cols):       # lengthen my single jumps
            cand.add(land)
        for path, nb, nbl, w in chains(b, s.ball, rows, cols, 200):  # squares just past chain ends, and their landings
            lr, lc = divmod(nbl, cols)
            for dr, dc in DIRS:
                r, c = lr + dr, lc + dc
                if 1 <= r <= rows - 2 and 0 <= c < cols:
                    cand.add(r * cols + c)
        return [i for i in cand if b[i] != BALL and b[i] != MAN]

    # --- decisions -------------------------------------------------------------------
    def choose_move(self, s: State) -> tuple[float, tuple]:
        """Best complete move from a position with no jump in progress.
        Returns (score, ('jump', path) | ('place', index))."""
        rows, cols, me = s.rows, s.cols, s.player
        best = None; self._work = 0
        mine = chains(s.board, s.ball, rows, cols, self.cap)
        for path, nb, nbl, w in mine:
            if w == me:
                return WIN, ("jump", path)
        # most forward-moving chains first, so a spent budget still leaves the best candidates scored
        mine = sorted((c for c in mine if not c[3]), key=lambda c: goal_dist(rows, c[2], cols, me))
        for path, nb, nbl, w in mine:
            if best is not None and self._work > self.budget:
                break
            v = self.after_my_move(nb, nbl, rows, cols, me)
            key = (v, -len(path))
            if best is None or key > best[0]:
                best = (key, ("jump", path))
        br0, bc0 = divmod(s.ball, cols)
        for i in sorted(self.placement_candidates(s), key=lambda i: max(abs(i // cols - br0), abs(i % cols - bc0))):
            if best is not None and self._work > self.budget:
                break
            nb = s.board[:]; nb[i] = MAN
            v = self.after_my_move(nb, s.ball, rows, cols, me)
            br, bc = divmod(s.ball, cols); r, c = divmod(i, cols)
            key = (v, 0.5 - 0.01 * max(abs(r - br), abs(c - bc)))     # placements rank just after equal jumps... nearer is better
            if best is None or key > best[0]:
                best = (key, ("place", i))
        return best[0][0], best[1]

    def continue_jump(self, s: State) -> int:
        """Mid-sequence: best of halting now vs continuing; returns the next micro-action."""
        rows, cols, me = s.rows, s.cols, s.player; n = rows * cols; self._work = 0
        best_v = self.after_my_move(s.board, s.ball, rows, cols, me); best_a = 2 * n
        mine = chains(s.board, s.ball, rows, cols, self.cap)
        for path, nb, nbl, w in mine:
            if w == me:
                return n + path[0]
        for path, nb, nbl, w in sorted((c for c in mine if not c[3]), key=lambda c: goal_dist(rows, c[2], cols, me)):
            if self._work > self.budget:
                break
            v = self.after_my_move(nb, nbl, rows, cols, me)
            if v > best_v:
                best_v, best_a = v, n + path[0]
        return best_a

    def action(self, s: State) -> int:
        """The next micro-action (placement index, jump landing + R*C, or halt 2*R*C)."""
        n = s.rows * s.cols
        if s.jumping:
            return self.continue_jump(s)
        _, mv = self.choose_move(s)
        return mv[1] if mv[0] == "place" else n + mv[1][0]
