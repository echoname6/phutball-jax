"""
Post-hoc tournament: Ladder (transfer learning) vs Tabula Rasa checkpoints.

For each board size on the ladder, loads checkpoints from both runs and plays
them against each other. Produces ELO ratings and win-rate curves suitable
for a research paper comparing sample efficiency.

Usage:
    python compare_ladder_vs_blank.py \
        --ladder-dir /path/to/ladder/checkpoints \
        --blank-dir  /path/to/blank/checkpoints \
        --board-size 15 9 \
        --games 20 \
        --sims 32 \
        --output results_15x9.json

    # Or run all board sizes at once:
    python compare_ladder_vs_blank.py \
        --ladder-root /path/to/phutball_checkpoints \
        --blank-root  /path/to/phutball_checkpoints \
        --all-sizes \
        --games 20 \
        --sims 32
"""

import argparse
import glob
import json
import os
import pickle
import re
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from phutball_env_jax import EnvConfig
from network import PhutballTransformer, create_transformer_network, init_transformer_network
from self_play_batched import (
    play_vs_checkpoint_batched,
    make_transformer_recurrent_fn,
    transformer_mcts_policy,
)
from train_batched import compute_elo_from_results


# Default architecture (must match training notebooks)
D_MODEL = 512
N_LAYERS = 12
LADDER_SIZES = ((13, 7), (15, 9), (17, 11), (19, 13), (21, 15))


def find_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoints in a directory, return sorted (iteration, path) pairs."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pkl")
    files = glob.glob(pattern)

    results = []
    for f in files:
        match = re.search(r'checkpoint_(\d+)\.pkl', os.path.basename(f))
        if match:
            results.append((int(match.group(1)), f))

    return sorted(results)


def load_params(path: str) -> dict:
    """Load just the params from a checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['params']


def load_checkpoint_metadata(path: str) -> dict:
    """Load metadata (iteration, total_games, etc.) from a checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    return {
        'iteration': checkpoint.get('iteration', 0),
        'total_games': checkpoint.get('total_games', 0),
        'total_examples': checkpoint.get('total_examples', 0),
    }


def run_tournament(
    ladder_checkpoints: List[Tuple[int, str]],
    blank_checkpoints: List[Tuple[int, str]],
    rows: int,
    cols: int,
    num_games: int = 20,
    num_simulations: int = 32,
    max_moves: int = 2048,
    d_model: int = D_MODEL,
    n_layers: int = N_LAYERS,
    sample_every: int = 1,
) -> dict:
    """Run a full tournament between ladder and blank checkpoints.

    Plays every ladder checkpoint against every blank checkpoint (sampled
    at `sample_every` interval to keep compute manageable).

    Returns a results dict with:
      - per-matchup W/D/L (both as P1 and P2)
      - ELO ratings for all checkpoints
      - metadata for plotting
    """
    env_config = EnvConfig(rows=rows, cols=cols, max_turns=512)
    network = create_transformer_network(
        rows=rows, cols=cols,
        d_model=d_model, n_layers=n_layers,
        n_heads=4, ffn_dim=d_model * 2,
        pos_encoding="goal_distance",
    )
    recurrent_fn = make_transformer_recurrent_fn(network, env_config)
    rng = jax.random.PRNGKey(42)

    # Sample checkpoints to keep tournament tractable
    ladder_sampled = ladder_checkpoints[::sample_every]
    blank_sampled = blank_checkpoints[::sample_every]

    all_names = []
    all_params = {}

    # Load all params
    print(f"Loading {len(ladder_sampled)} ladder + {len(blank_sampled)} blank checkpoints...")
    for iter_num, path in ladder_sampled:
        name = f"ladder_i{iter_num}"
        all_names.append(name)
        all_params[name] = load_params(path)

    for iter_num, path in blank_sampled:
        name = f"blank_i{iter_num}"
        all_names.append(name)
        all_params[name] = load_params(path)

    n = len(all_names)
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    W = np.zeros((n, n), dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)
    L = np.zeros((n, n), dtype=np.float64)

    matchup_results = {}
    total_matchups = len(ladder_sampled) * len(blank_sampled)
    completed = 0

    print(f"\nRunning {total_matchups} matchups ({num_games} games each)...")
    print(f"Board: {rows}x{cols}, MCTS sims: {num_simulations}\n")

    for l_iter, l_path in ladder_sampled:
        l_name = f"ladder_i{l_iter}"
        l_params = all_params[l_name]
        l_meta = load_checkpoint_metadata(l_path)

        for b_iter, b_path in blank_sampled:
            b_name = f"blank_i{b_iter}"
            b_params = all_params[b_name]
            b_meta = load_checkpoint_metadata(b_path)

            rng, eval_rng = jax.random.split(rng)

            t0 = time.time()
            (
                p1_wins, p1_draws, p1_losses,
                p2_wins, p2_draws, p2_losses,
                turns,
            ) = play_vs_checkpoint_batched(
                current_params=l_params,
                opponent_params=b_params,
                rng=eval_rng,
                network=network,
                env_config=env_config,
                num_games=num_games,
                max_moves=max_moves,
                num_simulations=num_simulations,
                temperature=0.0,
                dirichlet_fraction=0.0,
                mcts_policy_fn=transformer_mcts_policy,
                recurrent_fn=recurrent_fn,
            )
            elapsed = time.time() - t0

            wins = int(p1_wins) + int(p2_wins)
            draws = int(p1_draws) + int(p2_draws)
            losses = int(p1_losses) + int(p2_losses)

            _p1w, _p1d, _p1l = int(p1_wins), int(p1_draws), int(p1_losses)
            _p2w, _p2d, _p2l = int(p2_wins), int(p2_draws), int(p2_losses)

            # Update W/D/L matrices
            li, bi = name_to_idx[l_name], name_to_idx[b_name]
            W[li, bi] += wins
            D[li, bi] += draws
            L[li, bi] += losses
            W[bi, li] += losses
            D[bi, li] += draws
            L[bi, li] += wins

            total = wins + draws + losses
            wr = wins / total if total > 0 else 0.0

            completed += 1
            print(f"  [{completed}/{total_matchups}] {l_name} vs {b_name}: "
                  f"as P1: {_p1w}W-{_p1d}D-{_p1l}L | "
                  f"as P2: {_p2w}W-{_p2d}D-{_p2l}L | "
                  f"total: {wins}W-{draws}D-{losses}L ({wr:.0%}) "
                  f"[{elapsed:.1f}s]")

            matchup_results[f"{l_name}_vs_{b_name}"] = {
                "ladder_iter": l_iter,
                "blank_iter": b_iter,
                "ladder_games": l_meta["total_games"],
                "blank_games": b_meta["total_games"],
                "ladder_as_p1": {"w": _p1w, "d": _p1d, "l": _p1l},
                "ladder_as_p2": {"w": _p2w, "d": _p2d, "l": _p2l},
                "total": {"w": wins, "d": draws, "l": losses},
                "win_rate": wr,
            }

    # Compute ELO ratings
    print("\nComputing ELO ratings...")
    ratings_array = compute_elo_from_results(all_names, W, D, L)
    ratings = {name: float(ratings_array[i]) for i, name in enumerate(all_names)}

    # Sort and display
    ladder_ratings = {k: v for k, v in ratings.items() if k.startswith("ladder_")}
    blank_ratings = {k: v for k, v in ratings.items() if k.startswith("blank_")}

    print(f"\n{'='*60}")
    print(f"LADDER checkpoints ({rows}x{cols}):")
    for name in sorted(ladder_ratings, key=lambda x: int(x.split('_i')[1])):
        print(f"  {name}: {ladder_ratings[name]:.0f}")

    print(f"\nBLANK checkpoints ({rows}x{cols}):")
    for name in sorted(blank_ratings, key=lambda x: int(x.split('_i')[1])):
        print(f"  {name}: {blank_ratings[name]:.0f}")
    print(f"{'='*60}")

    # Build metadata for plotting
    ladder_curve = []
    for l_iter, l_path in ladder_sampled:
        name = f"ladder_i{l_iter}"
        meta = load_checkpoint_metadata(l_path)
        ladder_curve.append({
            "iteration": l_iter,
            "total_games": meta["total_games"],
            "elo": ratings[name],
            "source": "ladder",
        })

    blank_curve = []
    for b_iter, b_path in blank_sampled:
        name = f"blank_i{b_iter}"
        meta = load_checkpoint_metadata(b_path)
        blank_curve.append({
            "iteration": b_iter,
            "total_games": meta["total_games"],
            "elo": ratings[name],
            "source": "blank",
        })

    return {
        "board_size": f"{rows}x{cols}",
        "num_games_per_matchup": num_games,
        "num_simulations": num_simulations,
        "matchups": matchup_results,
        "ratings": ratings,
        "ladder_curve": ladder_curve,
        "blank_curve": blank_curve,
    }


def plot_comparison(results: dict, save_path: Optional[str] = None):
    """Plot ELO vs training games for ladder vs blank."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    ladder = results["ladder_curve"]
    blank = results["blank_curve"]
    board_size = results["board_size"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot by total training games (sample efficiency)
    if ladder:
        ax.plot(
            [p["total_games"] for p in ladder],
            [p["elo"] for p in ladder],
            'b-o', label="Ladder (transfer)", markersize=4,
        )
    if blank:
        ax.plot(
            [p["total_games"] for p in blank],
            [p["elo"] for p in blank],
            'r-s', label="Tabula rasa", markersize=4,
        )

    ax.set_xlabel("Training Games")
    ax.set_ylabel("ELO Rating")
    ax.set_title(f"Sample Efficiency: Ladder vs Tabula Rasa ({board_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare ladder vs tabula rasa training")

    # Single board size mode
    parser.add_argument("--ladder-dir", type=str, help="Ladder checkpoint directory")
    parser.add_argument("--blank-dir", type=str, help="Blank checkpoint directory")
    parser.add_argument("--board-size", type=int, nargs=2, metavar=("ROWS", "COLS"),
                        help="Board size (rows cols)")

    # All sizes mode
    parser.add_argument("--ladder-root", type=str, help="Root dir containing ladder checkpoint dirs")
    parser.add_argument("--blank-root", type=str, help="Root dir containing blank checkpoint dirs")
    parser.add_argument("--all-sizes", action="store_true", help="Run for all board sizes")

    # Tournament params
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (split evenly P1/P2)")
    parser.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--max-moves", type=int, default=2048, help="Max moves per game")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Sample every Nth checkpoint (to reduce compute)")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="Transformer d_model")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS, help="Transformer n_layers")

    # Output
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--plot", type=str, help="Save plot to this path (e.g. comparison.png)")

    args = parser.parse_args()

    if args.all_sizes:
        if not args.ladder_root or not args.blank_root:
            parser.error("--all-sizes requires --ladder-root and --blank-root")

        all_results = {}
        # Skip first rung (already tabula rasa)
        for rows, cols in LADDER_SIZES[1:]:
            ladder_dir = os.path.join(
                args.ladder_root,
                f"transformer_{LADDER_SIZES[0][0]}x{LADDER_SIZES[0][1]}_d{args.d_model}_l{args.n_layers}",
            )
            blank_dir = os.path.join(
                args.blank_root,
                f"blank_{rows}x{cols}_d{args.d_model}_l{args.n_layers}",
            )

            # For ladder, we need checkpoints that were saved during
            # training at this board size (the checkpoint dir may have
            # been renamed during escalation)
            ladder_size_dir = ladder_dir.replace(
                f"{LADDER_SIZES[0][0]}x{LADDER_SIZES[0][1]}",
                f"{rows}x{cols}",
            )

            if not os.path.exists(ladder_size_dir):
                print(f"\nSkipping {rows}x{cols}: ladder dir not found ({ladder_size_dir})")
                continue
            if not os.path.exists(blank_dir):
                print(f"\nSkipping {rows}x{cols}: blank dir not found ({blank_dir})")
                continue

            ladder_ckpts = find_checkpoints(ladder_size_dir)
            blank_ckpts = find_checkpoints(blank_dir)

            if not ladder_ckpts or not blank_ckpts:
                print(f"\nSkipping {rows}x{cols}: no checkpoints found")
                continue

            print(f"\n{'='*70}")
            print(f"TOURNAMENT: {rows}x{cols}")
            print(f"Ladder: {len(ladder_ckpts)} checkpoints from {ladder_size_dir}")
            print(f"Blank:  {len(blank_ckpts)} checkpoints from {blank_dir}")
            print(f"{'='*70}")

            results = run_tournament(
                ladder_ckpts, blank_ckpts,
                rows=rows, cols=cols,
                num_games=args.games,
                num_simulations=args.sims,
                max_moves=args.max_moves,
                d_model=args.d_model,
                n_layers=args.n_layers,
                sample_every=args.sample_every,
            )
            all_results[f"{rows}x{cols}"] = results

            if args.plot:
                base, ext = os.path.splitext(args.plot)
                plot_path = f"{base}_{rows}x{cols}{ext}"
                plot_comparison(results, save_path=plot_path)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nAll results saved to {args.output}")

    else:
        if not args.ladder_dir or not args.blank_dir or not args.board_size:
            parser.error("Need --ladder-dir, --blank-dir, and --board-size (or use --all-sizes)")

        rows, cols = args.board_size
        ladder_ckpts = find_checkpoints(args.ladder_dir)
        blank_ckpts = find_checkpoints(args.blank_dir)

        print(f"Ladder: {len(ladder_ckpts)} checkpoints")
        print(f"Blank:  {len(blank_ckpts)} checkpoints")

        results = run_tournament(
            ladder_ckpts, blank_ckpts,
            rows=rows, cols=cols,
            num_games=args.games,
            num_simulations=args.sims,
            max_moves=args.max_moves,
            d_model=args.d_model,
            n_layers=args.n_layers,
            sample_every=args.sample_every,
        )

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        if args.plot:
            plot_comparison(results, save_path=args.plot)
        else:
            plot_comparison(results)


if __name__ == "__main__":
    main()
