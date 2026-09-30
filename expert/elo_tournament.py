"""Round-robin Elo tournament between phutball checkpoints (GPU / Colab).

Every pair plays --games games per colour at a fixed Gumbel search (--sims x --considered, no root noise) on
--rows x --cols, games past --max-turns are draws (0.5). Ratings: Bradley-Terry maximum likelihood (minorisation-
maximisation), 95% intervals from --boot bootstrap resamples of the games, anchored so --anchor = --anchor-elo.
Human results can be added with --human "name:wins-losses-draws vs player" (e.g. "you:3-7-0 vs final").

  python -m expert.elo_tournament --out elo.json --anchor switch \\
      --player switch=selfplay_21x15_switch_s128_it50.pkl --player it80=selfplay_21x15_best_it80.pkl ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def bt_fit(names, results, iters=2000):
    """results: list of (i, j, score_i) per game (score 1 / 0.5 / 0). Returns log-strengths (natural units)."""
    n = len(names); W = np.zeros((n, n)); N = np.zeros((n, n))
    for i, j, s in results:
        W[i, j] += s; W[j, i] += 1 - s; N[i, j] += 1; N[j, i] += 1
    p = np.ones(n)
    for _ in range(iters):
        wins = W.sum(1) + 1e-3                                # tiny prior keeps all-loss players finite
        denom = np.array([sum(N[i, j] / (p[i] + p[j]) for j in range(n) if N[i, j]) for i in range(n)]) + 1e-9
        p = wins / denom; p /= np.exp(np.mean(np.log(p)))
    return np.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", action="append", required=True, help="name=path.pkl (repeatable)")
    ap.add_argument("--anchor", required=True); ap.add_argument("--anchor-elo", type=float, default=1000.0)
    ap.add_argument("--games", type=int, default=64, help="games per colour per pair")
    ap.add_argument("--sims", type=int, default=32); ap.add_argument("--considered", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=360); ap.add_argument("--rows", type=int, default=21)
    ap.add_argument("--cols", type=int, default=15); ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--human", action="append", default=[], help='"name:W-L-D vs player"')
    ap.add_argument("--extra-results", type=Path, action="append", default=[],
                    help="JSON list of [player, opponent, score] games to merge (e.g. from expert/puct_calibration.py --opponent)")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.25, help="move sampling temperature (game variety; see the "
                    "distinct-games count in each match line)")
    ap.add_argument("--from-results", type=Path, default=None, help="refit from a previous --out file's games instead of "
                    "playing the round robin (add --extra-results / --human on top)")
    a = ap.parse_args()

    import jax
    if not hasattr(jax.core, "get_opaque_trace_state"):
        import jax.extend.core
        jax.core.get_opaque_trace_state = jax.extend.core.get_opaque_trace_state
    from network import create_transformer_network
    from phutball_env_jax import EnvConfig
    from self_play_batched import make_transformer_recurrent_fn, play_games_batched, transformer_mcts_policy
    from functools import partial

    names, params = [], []
    for spec in a.player:
        name, path = spec.split("=", 1); d = pickle.load(open(path, "rb")); names.append(name); params.append(d.get("params", d))
    idx = {n: i for i, n in enumerate(names)}
    R, C = a.rows, a.cols; cfg = EnvConfig(rows=R, cols=C)
    net = create_transformer_network(rows=R, cols=C, d_model=128, n_layers=6, n_heads=4, ffn_dim=256, pos_encoding="goal_distance")
    rf = make_transformer_recurrent_fn(net, cfg)
    policy = partial(transformer_mcts_policy, dirichlet_fraction=0.0, gumbel_scale=0.0)      # evaluation: no root noise
    kw = dict(network=net, env_config=cfg, batch_size=a.games, max_turns=a.max_turns, max_moves=2 * a.max_turns,
              temperature=a.temperature, temp_threshold=R, temp_final=a.temperature, num_simulations=a.sims,
              max_num_considered_actions=a.considered, random_opponent_ratio=0.0, mcts_policy_fn=policy, recurrent_fn=rf)
    play = jax.jit(lambda p, r, o: play_games_batched({"network_params": p}, r, opponent_params={"network_params": o},
                                                      opponent_ratio=1.0, **kw))
    rng = jax.random.PRNGKey(a.seed); results = []; t0 = time.time()
    pairs = list(itertools.combinations(range(len(names)), 2))
    if a.from_results:                                                # refit only: reuse the saved games
        pairs = []
        for x, y, s in json.loads(a.from_results.read_text())["results"]:
            for n in (x, y):
                if n not in idx: idx[n] = len(names); names.append(n)
            results.append((idx[x], idx[y], float(s)))
    for i, j in pairs:
        sc = 0.0; w = l = 0; seqs = set()
        for _ in range(2):                                           # play_games_batched assigns sides at random
            rng, r = jax.random.split(rng); traj = play(params[i], r, params[j])
            _, _, side_rng, _ = jax.random.split(r, 4)
            main_p1 = np.array(jax.random.uniform(side_rng, (a.games,)) < 0.5)
            acts, valid = np.array(traj.actions), np.array(traj.valid_mask)
            seqs |= {(bool(m1),) + tuple(acts[g][valid[g]].tolist()) for g, m1 in enumerate(main_p1)}
            for win, m1 in zip(np.array(traj.winners), main_p1):
                s = 0.5 if win == 0 else (1.0 if (win == 1) == m1 else 0.0)
                results.append((i, j, s)); sc += s; w += s == 1.0; l += s == 0.0
        print(f"{names[i]} vs {names[j]}: {sc / (2 * a.games):.1%} ({w}W {l}L {2 * a.games - w - l}D) | {len(seqs)} distinct games "
              f"of {2 * a.games} | {(time.time() - t0) / 60:.1f} min", flush=True)
    for f in a.extra_results:                                         # e.g. the browser search (PUCT) as its own player
        for x, y, s in json.loads(f.read_text()):
            for n in (x, y):
                if n not in idx: idx[n] = len(names); names.append(n)
            results.append((idx[x], idx[y], float(s)))
    for h in a.human:                                                # e.g. "you:3-7-0 vs final"
        who, rest = h.split(":", 1); rec, _, opp = rest.partition(" vs ")
        wh, lh, dh = (int(x) for x in rec.split("-"))
        if who not in idx: idx[who] = len(names); names.append(who)
        for s, k in ((1.0, wh), (0.0, lh), (0.5, dh)): results += [(idx[who], idx[opp.strip()], s)] * k

    scale = 400 / np.log(10)
    def ratings(res):
        s = bt_fit(names, res) * scale; return s - s[idx[a.anchor]] + a.anchor_elo
    elo = ratings(results); rs = np.random.default_rng(a.seed); boot = []
    for _ in range(a.boot):
        pick = rs.integers(0, len(results), len(results)); boot.append(ratings([results[k] for k in pick]))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    order = np.argsort(-elo)
    head = f"refit of {a.from_results.name}" if a.from_results else f"{a.sims}x{a.considered} search, {a.games} games per colour per pair"
    print(f"\nElo ({head}; {len(results)} games; {a.anchor} = {a.anchor_elo:.0f})")
    for k in order: print(f"  {names[k]:14s} {elo[k]:7.0f}   95% [{lo[k]:.0f}, {hi[k]:.0f}]")
    a.out.write_text(json.dumps({"ratings": {names[k]: {"elo": round(float(elo[k]), 1), "lo": round(float(lo[k]), 1),
                                                         "hi": round(float(hi[k]), 1)} for k in order},
                                 "results": [[names[i], names[j], s] for i, j, s in results], "config": {k: str(v) for k, v in vars(a).items()}}, indent=1))


if __name__ == "__main__":
    main()
