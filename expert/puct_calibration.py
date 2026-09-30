"""Calibrate the browser opponent (PUCT, c = 2.5, N playouts, greedy on visits, as in the fixed
phutball-frontend/public/alphaZeroWorker.js) against the tournament's Gumbel search, with the same network.

Plays --games games per colour of PUCT-N vs Gumbel-S x --considered for each S in --gumbel-sims, on the Python engine
(games past --max-turns are draws), under the round robin's conditions: the empty board, the Gumbel side sampling at
temperature 0.25 (PUCT is greedy, as in the browser). Each match reports its distinct games. The Elo gap from each score, 400 log10(s / (1 - s)), places "network + browser
search" on the tournament scale (which rates networks at 32 x 16), and with it any human results against the browser.

  python -m expert.puct_calibration --params selfplay_21x15_best_it100.pkl --out puct_calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class PUCT:
    """The browser worker's search (after the sign / player-2 fixes), with children created on first visit."""

    def __init__(self, net_policy, playouts, c=2.5, rows=21, cols=15):
        self.np, self.n, self.c, self.N = net_policy, playouts, c, rows * cols

    def evaluate(self, s):
        x = self.np.enc(self.np._jax_state(s))[None]
        lg, v = self.np.apply(x); lg = np.array(lg[0], np.float64); v = float(np.array(v).reshape(-1)[0])
        if s.player == 2: lg[:self.N] = lg[:self.N][::-1].copy(); lg[self.N:2 * self.N] = lg[self.N:2 * self.N][::-1].copy()
        return lg, v

    def action(self, s):
        from expert.engine import legal_actions, step
        # node = [state, player, prior, visits, value_sum, children(list) or None, actions, priors]
        def make(state, prior): return {"s": state, "pl": state.player, "p": prior, "n": 0, "w": 0.0, "ch": None}

        def expand(node):
            lg, v = self.evaluate(node["s"]); acts = legal_actions(node["s"])
            z = lg[acts]; e = np.exp(z - z.max()); e /= e.sum()
            node["acts"], node["pri"], node["ch"] = acts, e, [None] * len(acts)
            return v

        root = make(s, 0.0); expand(root)
        for _ in range(self.n):
            node, path = root, [root]
            while node["ch"] is not None and len(node["ch"]):
                best, bi = -1e9, 0
                sq = math.sqrt(node["n"])
                for k, a in enumerate(node["acts"]):
                    c = node["ch"][k]
                    if c is None: q, n_, cpl = 0.0, 0, None
                    else: q, n_, cpl = (c["w"] / c["n"] if c["n"] else 0.0), c["n"], c["pl"]
                    if cpl is not None and cpl != node["pl"]: q = -q          # value from the chooser's side
                    sc = q + self.c * node["pri"][k] * sq / (1 + n_)
                    if sc > best: best, bi = sc, k
                if node["ch"][bi] is None:
                    node["ch"][bi] = make(step(node["s"], int(node["acts"][bi])), float(node["pri"][bi]))
                node = node["ch"][bi]; path.append(node)
                if node["n"] == 0: break
            if node["s"].winner: v = 1.0 if node["s"].winner == node["s"].player else -1.0
            elif node["ch"] is None: v = expand(node)
            else: v = 0.0
            for k in range(len(path) - 1, -1, -1):
                path[k]["n"] += 1; path[k]["w"] += v
                if k > 0 and path[k]["pl"] != path[k - 1]["pl"]: v = -v
        visits = [(c["n"] if c else 0) for c in root["ch"]]
        return int(root["acts"][int(np.argmax(visits))])


class Gumbel:
    """The tournament's player: Gumbel search without root noise, the move sampled from the visit-based policy at
    `temperature` (0.25 as in expert/elo_tournament.py; 0 = greedy)."""
    def __init__(self, net, params, net_policy, sims, considered, rows=21, cols=15, temperature=0.25, seed=0):
        import jax
        from phutball_env_jax import EnvConfig
        from self_play_batched import make_transformer_recurrent_fn, transformer_mcts_policy
        cfg = EnvConfig(rows=rows, cols=cols); rf = make_transformer_recurrent_fn(net, cfg)
        self.jax, self.np, self.t = jax, net_policy, temperature; self.k = jax.random.PRNGKey(seed)
        self.rs = np.random.default_rng(seed)
        self.run = jax.jit(lambda st, r: transformer_mcts_policy({"network_params": params}, st, r, net, cfg, num_simulations=sims,
                                                                 temperature=1.0, max_num_considered_actions=considered,
                                                                 recurrent_fn=rf, dirichlet_fraction=0.0, gumbel_scale=0.0)[1])

    def action(self, s):
        st = self.jax.tree_util.tree_map(lambda x: x[None], self.np._jax_state(s)); self.k, r = self.jax.random.split(self.k)
        w = np.array(self.run(st, r), np.float64)[0]
        if self.t <= 0: return int(w.argmax())
        z = np.log(w + 1e-12) / self.t; q = np.exp(z - z.max()); q /= q.sum()
        return int(self.rs.choice(len(q), p=q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--playouts", type=int, default=400); ap.add_argument("--gumbel-sims", default="32,64,128")
    ap.add_argument("--considered", type=int, default=16); ap.add_argument("--games", type=int, default=10, help="per colour")
    ap.add_argument("--opponent", action="append", default=[], help="name=path: play PUCT (with --params) against these "
                    "networks at Gumbel --gumbel-sims (first value) and write the games as tournament extra results")
    ap.add_argument("--name", default="it100-puct400", help="player name for PUCT in --opponent mode")
    ap.add_argument("--max-turns", type=int, default=360); ap.add_argument("--openings", type=int, default=0,
                    help="random centre placements before play (default 0: the empty board, as in the round robin and the browser)")
    ap.add_argument("--temperature", type=float, default=0.25, help="Gumbel side's move sampling (the round robin's)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import jax
    if not hasattr(jax.core, "get_opaque_trace_state"):
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    import random
    from network import create_transformer_network
    from expert.engine import legal_actions, new_game, step
    from expert.net_policy import NetPolicy
    net = create_transformer_network(rows=21, cols=15, d_model=128, n_layers=6, n_heads=4, ffn_dim=256, pos_encoding="goal_distance")
    d = pickle.load(open(a.params, "rb")); params = d.get("params", d); npol = NetPolicy(net, params)
    puct = PUCT(npol, a.playouts); out = {}
    matchups = []                                       # (label, gumbel player)
    if a.opponent:
        sims = int(a.gumbel_sims.split(",")[0])
        for spec in a.opponent:
            name, path = spec.split("=", 1); d2 = pickle.load(open(path, "rb")); p2 = d2.get("params", d2)
            matchups.append((name, Gumbel(net, p2, NetPolicy(net, p2), sims, a.considered, temperature=a.temperature, seed=a.seed), sims))
    else:
        for sims in [int(x) for x in a.gumbel_sims.split(",")]:
            matchups.append((f"gumbel{sims}", Gumbel(net, params, npol, sims, a.considered, temperature=a.temperature, seed=a.seed), sims))
    games_out = []
    for label, gum, sims in matchups:
        sc = 0.0; w = l = 0; t0 = time.time(); seqs = set()
        for g in range(2 * a.games):
            puct_side = 1 if g % 2 == 0 else 2; rng = random.Random(g // 2)
            s = new_game(21, 15)
            for _ in range(a.openings):
                s = step(s, rng.choice([x for x in legal_actions(s) if x < 315 and 6 * 15 <= x < 15 * 15]))
            moves = []
            while not s.winner and s.turns < a.max_turns:
                mv = (puct if s.player == puct_side else gum).action(s); moves.append(mv); s = step(s, mv)
            seqs.add((puct_side,) + tuple(moves))
            r = 0.5 if not s.winner else (1.0 if s.winner == puct_side else 0.0); sc += r; w += r == 1.0; l += r == 0.0
            games_out.append([a.name, label, r])
            print(f"  PUCT-{a.playouts} vs {label} (Gumbel-{sims}x{a.considered}), game {g + 1}: {'win' if r == 1 else 'loss' if r == 0 else 'draw'} "
                  f"as P{puct_side} ({s.turns} turns) | {(time.time() - t0) / 60:.1f} min", flush=True)
        s_ = min(max(sc / (2 * a.games), 0.5 / (2 * a.games)), 1 - 0.5 / (2 * a.games))
        gap = 400 * math.log10(s_ / (1 - s_))
        out[label] = {"score": sc / (2 * a.games), "wins": w, "losses": l, "distinct_games": len(seqs),
                      "elo_gap_puct_minus_opponent": round(gap)}
        print(f"PUCT-{a.playouts} vs {label} (Gumbel-{sims}x{a.considered}): {sc / (2 * a.games):.1%} ({w}W {l}L) | "
              f"{len(seqs)} distinct games of {2 * a.games} -> PUCT is {gap:+.0f} Elo", flush=True)
    a.out.write_text(json.dumps(games_out if a.opponent else out, indent=1))


if __name__ == "__main__":
    main()
