"""An exact Python port of phutball.net's AlphaZero opponent (phutball-app, phutball-frontend/public/alphaZeroWorker.js
on the production branch), and the site's own rules for adjudicating a game, so that a line found offline replays in
the browser.

The worker's search model differs from the site's rules (and from the training engine), and the port keeps its quirks:
  - a jump that lands in EITHER end zone ends the game with the JUMPER winning (`s.winner = s.currentPlayer`); the
    site (and the training engine) award a landing in your own back rows to the opponent;
  - no passthrough: the worker stops a jump sequence at the first end-zone landing; the site lets a sequence pass
    through an end zone and decides on the final landing only;
  - placements only on EMPTY cells of rows 1..R-2 (goal-row tiles are END_HI / END_LO, so not placeable); a ball
    leaving a tile sets it to EMPTY.
Search (as the worker): PUCT c = 2.5, eager children in ascending action order, select the first maximum, terminal
value +1 if winner == the node's player to move, backup negated across player changes, fresh tree per decision,
greedy on visits (first maximum in ascending action order); a full move = one search per micro-action until a halt,
a placement, or the game ending.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field

import numpy as np

EMPTY, BALL, MAN, END_HI, END_LO = 0, -1, 1, 2, -2
DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class WState:
    rows: int
    cols: int
    board: np.ndarray                     # (rows, cols) int
    ball: tuple
    player: int = 1
    jumping: bool = False
    winner: int = 0
    seq: list = field(default_factory=list)

    def copy(self):
        return WState(self.rows, self.cols, self.board.copy(), self.ball, self.player, self.jumping, self.winner,
                      list(self.seq))


def new_game(rows=21, cols=15) -> WState:
    b = np.zeros((rows, cols), np.int64); b[0:2] = END_HI; b[rows - 2:] = END_LO
    ball = (rows // 2, cols // 2); b[ball] = BALL
    return WState(rows, cols, b, ball)


def check_for_win(s) -> int:
    r = s.ball[0]
    return 1 if r <= 1 else 2 if r >= s.rows - 2 else 0


def jumps(s):
    """[(landing (r, c), jumped cells)] as the worker's legal mask (landing on EMPTY or an end-zone tile)."""
    out = []; sr, sc = s.ball
    for dr, dc in DIRS:
        jumped = []; r, c = sr + dr, sc + dc
        while 0 <= r < s.rows and 0 <= c < s.cols and s.board[r, c] == MAN:
            jumped.append((r, c)); r += dr; c += dc
        if jumped and 0 <= r < s.rows and 0 <= c < s.cols and s.board[r, c] in (EMPTY, END_HI, END_LO):
            out.append(((r, c), jumped))
    return out


def legal(s) -> list[int]:
    N = s.rows * s.cols; acts = []
    if not s.jumping:
        for r in range(1, s.rows - 1):
            for c in range(s.cols):
                if s.board[r, c] == EMPTY: acts.append(r * s.cols + c)
    acts += sorted(N + r * s.cols + c for (r, c), _ in jumps(s))
    if s.jumping: acts.append(2 * N)
    return acts                                                   # ascending action order, as the worker's mask loop


def worker_apply(s, a):
    """The worker's applyAction (its search model, quirks included). Mutates s."""
    N = s.rows * s.cols
    if a < N:
        r, c = divmod(a, s.cols); s.board[r, c] = MAN; s.player = 3 - s.player; s.jumping = False; s.seq = []
        s.winner = check_for_win(s); return s
    if a < 2 * N:
        r1, c1 = divmod(a - N, s.cols); r0, c0 = s.ball
        if not s.jumping: s.seq = [(r0, c0)]; s.jumping = True
        dr, dc = np.sign(r1 - r0), np.sign(c1 - c0); steps = max(abs(r1 - r0), abs(c1 - c0))
        for i in range(1, steps): s.board[r0 + i * dr, c0 + i * dc] = EMPTY
        s.board[r0, c0] = EMPTY; s.board[r1, c1] = BALL; s.ball = (r1, c1); s.seq.append((r1, c1))
        if check_for_win(s): s.winner = s.player; s.jumping = False; s.seq = []     # the jumper "wins" either way
        return s
    s.jumping = False; s.winner = check_for_win(s); s.seq = []
    if s.winner == 0: s.player = 3 - s.player
    return s


def observation(s) -> np.ndarray:
    R, C = s.rows, s.cols; obs = np.zeros((9, R, C), np.float32)
    obs[0][s.board == BALL] = 1; obs[1][s.board == MAN] = 1
    hi = np.zeros((R, C), bool); hi[:2] = True; lo = np.zeros((R, C), bool); lo[R - 2:] = True
    obs[2][lo if s.player == 2 else hi] = 1; obs[3][hi if s.player == 2 else lo] = 1
    if s.seq:
        visits = {}
        for step, (r, c) in enumerate(s.seq):
            layer = min(visits.get((r, c), 0), 4); obs[4 + layer, r, c] = step + 1.0; visits[(r, c)] = visits.get((r, c), 0) + 1
    if s.player == 2: obs = obs[:, ::-1, ::-1].copy()
    return obs


class Net:
    def __init__(self, params_path):
        import jax
        if not hasattr(jax.core, "get_opaque_trace_state"):
            import jax.extend.core; jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
        from network import create_transformer_network
        net = create_transformer_network(rows=21, cols=15, d_model=128, n_layers=6, n_heads=4, ffn_dim=256,
                                         pos_encoding="goal_distance")
        d = pickle.load(open(params_path, "rb")); params = d.get("params", d)
        self.apply = jax.jit(lambda x: net.apply({"params": params}, x, train=False)); self.calls = 0

    def values(self, states):
        """Batched value heads (side to move of each state)."""
        import numpy as _np
        out = []
        for i in range(0, len(states), 64):
            x = _np.stack([observation(t) for t in states[i:i + 64]]); _, v = self.apply(x); out += list(_np.array(v).reshape(-1))
        self.calls += len(states); return out

    def __call__(self, s):
        self.calls += 1
        lg, v = self.apply(observation(s)[None]); lg = np.array(lg[0], np.float64); v = float(np.array(v).reshape(-1)[0])
        if s.player == 2:                                         # side-to-move view -> physical
            N = s.rows * s.cols; lg[:N] = lg[:N][::-1].copy(); lg[N:2 * N] = lg[N:2 * N][::-1].copy()
        return lg, v


class Node:
    """A worker MCTSNode. The worker builds every child's state at expansion; here a child's state is built on its
    first selection (an unvisited child has mean 0, so its player-to-move never affects the scores): same search."""
    __slots__ = ("_s", "parent", "action", "prior", "visits", "total", "children")

    def __init__(self, s, parent, action, prior):
        self._s, self.parent, self.action, self.prior = s, parent, action, prior
        self.visits, self.total, self.children = 0, 0.0, []

    @property
    def s(self):
        if self._s is None: self._s = worker_apply(self.parent.s.copy(), self.action)
        return self._s

    def mean(self): return self.total / self.visits if self.visits else 0.0


def expand(node, logits, acts):
    z = np.array([logits[a] for a in acts], np.float64); e = np.exp(z - z.max()); p = e / e.sum()
    node.children = [Node(None, node, a, pr) for a, pr in zip(acts, p)]


def search(net, s, sims):
    """One worker runMCTS: visit counts {action: visits}."""
    root = Node(s.copy(), None, -1, 0.0); lg, _ = net(root.s); acts = legal(root.s)
    if not acts: return {}
    expand(root, lg, acts)
    for _ in range(sims):
        node = root
        while node.children:
            best, bs = None, -math.inf; sq = math.sqrt(node.visits)
            for ch in node.children:
                q = 0.0 if not ch.visits else (-ch.mean() if ch.s.player != node.s.player else ch.mean())
                sc = q + 2.5 * ch.prior * sq / (1 + ch.visits)
                if sc > bs: bs, best = sc, ch
            node = best
        if node.s.winner:
            v = 1.0 if node.s.winner == node.s.player else -1.0
        else:
            lg, v = net(node.s); a2 = legal(node.s)
            if a2: expand(node, lg, a2)
        cur = node
        while cur is not None:
            cur.visits += 1; cur.total += v
            if cur.parent is not None and cur.s.player != cur.parent.s.player: v = -v
            cur = cur.parent
    return {ch.action: ch.visits for ch in root.children}


def greedy(visits: dict) -> int:
    best, mv = 0, -1
    for a in sorted(visits):
        if visits[a] > mv: mv, best = visits[a], a
    return best


def bot_move(net, s, sims=400):
    """The worker's selectFullMove: ('place', (r, c)) or ('jump', [landings]) (empty list: halt mid-jump)."""
    N = s.rows * s.cols; w = s.copy()
    a = greedy(search(net, w, sims))
    if a == 2 * N: return ("jump", [])
    if a < N: return ("place", divmod(a, s.cols))
    path = [divmod(a - N, s.cols)]; worker_apply(w, a)
    for _ in range(50):
        if w.winner: break
        a = greedy(search(net, w, sims))
        if a >= 2 * N or a < N: break
        path.append(divmod(a - N, s.cols)); worker_apply(w, a)
    return ("jump", path)


# --- the site's rules (PhutballEngine.js): what actually decides a game ------------------------------------------
def site_apply(s, move):
    """Apply a full move under the site's rules: a placement, or a jump path (passthrough allowed) then halt; the
    final landing decides (rows 0-1: player 1 wins; rows R-2..R-1: player 2). Returns a new state (winner set)."""
    s = s.copy(); kind, arg = move
    if kind == "place":
        r, c = arg
        assert s.board[r, c] == EMPTY and 1 <= r <= s.rows - 2, f"illegal placement {arg}"
        s.board[r, c] = MAN; s.player = 3 - s.player; return s
    for (r1, c1) in arg:
        land = {l: j for l, j in jumps(s)}
        assert (r1, c1) in land, f"illegal jump to {(r1, c1)}"
        r0, c0 = s.ball
        for jr, jc in land[(r1, c1)]: s.board[jr, jc] = EMPTY
        s.board[r0, c0] = EMPTY; s.board[r1, c1] = BALL; s.ball = (r1, c1)
    s.winner = check_for_win(s)
    if not s.winner: s.player = 3 - s.player
    return s
