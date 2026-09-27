"""Play matches between policies on the fast engine (micro-action level)."""
from __future__ import annotations

import argparse
import random
import sys
import time

sys.path.insert(0, ".")
from expert.engine import MAN, jump_landings, legal_actions, new_game, step  # noqa: E402
from expert.oracle import Expert, chains, goal_dist  # noqa: E402


class RandomPolicy:
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def action(self, s):
        acts = legal_actions(s); n = s.rows * s.cols
        jumps = [a for a in acts if a >= n]
        return self.rng.choice(jumps) if (s.jumping and jumps) else self.rng.choice(acts)


class GreedyPolicy:
    """1-ply: take a winning chain; else the chain with the most progress if it gains >= 2 rows;
    else place a man next to the ball on the forward side (random among them)."""
    def __init__(self, seed=0): self.rng = random.Random(seed); self.plan = []
    def action(self, s):
        n = s.rows * s.cols
        if s.jumping:
            return n + self.plan.pop(0) if self.plan else 2 * n
        best, bestg = None, 1
        d0 = goal_dist(s.rows, s.ball, s.cols, s.player)
        for path, nb, nbl, w in chains(s.board, s.ball, s.rows, s.cols, 2000):
            if w == s.player: best = path; break
            if w: continue
            g = d0 - goal_dist(s.rows, nbl, s.cols, s.player)
            if g > bestg: best, bestg = path, g
        if best:
            self.plan = list(best[1:]); return n + best[0]
        br, bc = divmod(s.ball, s.cols); fwd = -1 if s.player == 1 else 1
        opts = [(br + fwd) * s.cols + c for c in (bc - 1, bc, bc + 1)
                if 0 <= c < s.cols and 1 <= br + fwd <= s.rows - 2 and s.board[(br + fwd) * s.cols + c] == 0]
        return self.rng.choice(opts) if opts else self.rng.choice([a for a in legal_actions(s) if a < n])


def play(p1, p2, rows=21, cols=15, max_turns=600):
    s = new_game(rows, cols); pol = {1: p1, 2: p2}
    while not s.winner and not s.draw and s.turns < max_turns:
        s = step(s, pol[s.player].action(s))
    return s.winner, s.turns


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--rows", type=int, default=21); ap.add_argument("--cols", type=int, default=15)
    x = ap.parse_args()
    mk = {"random": RandomPolicy, "greedy": GreedyPolicy, "expert": lambda seed=0: Expert()}
    res = {"a": 0, "b": 0, "draw": 0}; t0 = time.time(); turns = []
    for g in range(x.games):
        a, b = mk[x.a](seed=g), mk[x.b](seed=1000 + g)
        if g % 2 == 0: w, t = play(a, b, x.rows, x.cols); res["a" if w == 1 else "b" if w == 2 else "draw"] += 1
        else:          w, t = play(b, a, x.rows, x.cols); res["a" if w == 2 else "b" if w == 1 else "draw"] += 1
        turns.append(t)
    print(f"{x.a} vs {x.b} on {x.rows}x{x.cols}, {x.games} games (colours alternate): {x.a} {res['a']}, {x.b} {res['b']}, draws {res['draw']}; "
          f"mean length {sum(turns)/len(turns):.0f} turns; {time.time()-t0:.0f}s")
