"""
AlphaZero Training Loop for Phutball - Batched Version

Uses batched self-play for TPU efficiency.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import pickle
from mcts import MCTSConfig
import glob

from phutball_env_jax import EnvConfig, reset, step, get_legal_actions
from network import (
    PhutballNetwork, create_network, init_network,
    create_optimizer, make_train_step_fn, predict
)
from self_play_batched import (
    play_games_batched,
    trajectory_to_training_examples,
    ReplayBuffer,
    compute_phutball_stats,
    play_match_batched,
    play_vs_random_batched, 
    batched_mcts_policy,
    batched_reset,
)


try:
    import wandb
except ImportError:
    wandb = None



@dataclass
class TrainConfig:
    """Full training configuration."""
    # Environment
    rows: int = 21
    cols: int = 15
    
    # Network architecture
    num_channels: int = 64
    num_res_blocks: int = 6
    
    # Self-play (batched)
    batch_size_games: int = 64       # Games played in parallel
    max_turns_per_game: int = 2048   # Max turns before game ends (safety cap)
    max_moves_per_game: int = 4096 # Memory cap on moves stored
    temperature: float = 1.0          # Sampling temperature
    num_simulations: int = 50         # MCTS simulations per move (0 = raw network)
    
    # Training
    batch_size_train: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100
    
    # Replay buffer
    buffer_size: int = 500000
    min_buffer_size: int = 1000  # Min examples before training starts
    
    # Iterations
    num_iterations: int = 500
    games_per_iteration: int = 256   # Total games per iteration (multiple batches)
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10  # iterations
    
    # Logging
    log_every: int = 1

    eval_enable: bool = False
    eval_max_prev_checkpoints: int = 5
    eval_games_per_color: int = 5      # 5 as P1 + 5 as P2 per opponent
    eval_num_simulations: int = 128    # MCTS sims per move during eval
    eval_max_moves: int = 2048         # cutoff to avoid marathon eval games
    eval_temperature: float = 0.0

    eval_vs_random_games: int = 100
    eval_vs_random_threshold: Optional[float] = None
    stop_when_beating_random: bool = False

    use_wandb: bool = False


    use_wandb: bool = False
    wandb_project: str = "phutball-az"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class AlphaZeroTrainer:
    """Main training class with batched self-play."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Environment config
        self.env_config = EnvConfig(rows=config.rows, cols=config.cols, max_turns=config.max_turns_per_game)
        
        # Create network
        self.network = create_network(
            rows=config.rows,
            cols=config.cols,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        )
        
        # Optimizer
        self.optimizer = create_optimizer(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize network
        self._init_network()
        
        # Create JIT-compiled train step
        self.train_step_fn = make_train_step_fn(self.network, self.optimizer)
        
        # Metrics
        self.iteration = 0
        self.total_games = 0
        self.total_examples = 0
        self.metrics_history = []
        self.last_self_play_stats = None

        self.wandb_run = None
        if self.config.use_wandb:
            if wandb is None:
                print("WARNING: use_wandb=True but wandb is not installed; disabling wandb.")
                self.config.use_wandb = False
            else:
                run_name = (
                    self.config.wandb_run_name
                    or f"phutball_az_{int(time.time())}"
                )
                cfg = asdict(self.config)
                # asdict(config) is all simple types, so fine for wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    config=cfg,
                    mode=self.config.wandb_mode,
                )
        
    def _init_network(self):
        """Initialize network parameters."""
        self.rng, init_rng = jax.random.split(self.rng)
        variables = init_network(init_rng, self.network, num_input_channels=6)
        
        self.params = variables['params']
        self.batch_stats = variables['batch_stats']
        self.opt_state = self.optimizer.init(self.params)
    
    def get_network_params(self) -> dict:
        """Get params dict for self-play."""
        return {
            'network_params': self.params,
            'batch_stats': self.batch_stats,
        }
    
    def run_self_play(self) -> int:
        """Run batched self-play games. Returns number of new examples."""
        start_time = time.time()
        
        total_states = []
        total_policies = []
        total_values = []

        stats_totals = {
            "num_games": 0,
            "total_moves": 0,
            "total_placements": 0,
            "total_jumps": 0,
            "num_jump_sequences": 0,
            "sum_jump_sequence_lengths": 0,
            "sum_jump_removed_tiles": 0,
            "adjacency_opportunities": 0,
            "adjacency_conversions": 0
        }

        p1_wins = 0
        p2_wins = 0
        draws = 0
        
        # Run multiple batches to get desired number of games
        num_batches = self.config.games_per_iteration // self.config.batch_size_games
        
        for batch_idx in range(num_batches):
            self.rng, game_rng = jax.random.split(self.rng)
            
            # Play games in parallel
            trajectory = play_games_batched(
                params=self.get_network_params(),
                rng=game_rng,
                network=self.network,
                env_config=self.env_config,
                batch_size=self.config.batch_size_games,
                max_turns=self.config.max_turns_per_game,
                max_moves=self.config.max_moves_per_game,
                temperature=self.config.temperature,
                num_simulations=self.config.num_simulations,
            )

            batch_stats = compute_phutball_stats(trajectory, self.env_config)
            for k in stats_totals:
                stats_totals[k] += batch_stats[k]
            
            winners_np = np.array(trajectory.winners)
            p1_wins += int(np.sum(winners_np == 1))
            p2_wins += int(np.sum(winners_np == 2))
            draws += int(np.sum(winners_np == 0))
            
            # Convert to training examples
            states, policies, values = trajectory_to_training_examples(trajectory)
            
            total_states.append(states)
            total_policies.append(policies)
            total_values.append(values)
        
        # Concatenate all examples
        if total_states:
            all_states = np.concatenate(total_states, axis=0)
            all_policies = np.concatenate(total_policies, axis=0)
            all_values = np.concatenate(total_values, axis=0)
            
            self.replay_buffer.add(all_states, all_policies, all_values)
            num_examples = len(all_states)
        else:
            num_examples = 0
        
        elapsed = time.time() - start_time
        games_total = num_batches * self.config.batch_size_games
        self.total_games += games_total
        self.total_examples += num_examples

        # ---- derive averages from stats_totals ----
        if stats_totals["num_games"] > 0:
            ng = stats_totals["num_games"]
            total_moves = stats_totals["total_moves"]
            total_placements = stats_totals["total_placements"]
            total_jumps = stats_totals["total_jumps"]
            num_seq = stats_totals["num_jump_sequences"]
            seq_len_sum = stats_totals["sum_jump_sequence_lengths"]
            removed_sum = stats_totals["sum_jump_removed_tiles"]
            adj_ops = stats_totals["adjacency_opportunities"]
            adj_conv = stats_totals["adjacency_conversions"]

            avg_moves_per_game = total_moves / ng
            avg_placements_per_jump = (
                float(total_placements) / total_jumps if total_jumps > 0 else 0.0
            )
            avg_jump_seq_len = (
                float(seq_len_sum) / num_seq if num_seq > 0 else 0.0
            )
            avg_jump_length = (
                float(removed_sum) / total_jumps if total_jumps > 0 else 0.0
            )
            adj_conv_rate = (
                float(adj_conv) / adj_ops if adj_ops > 0 else 0.0
            )
        else:
            avg_moves_per_game = avg_placements_per_jump = 0.0
            avg_jump_seq_len = avg_jump_length = adj_conv_rate = 0.0

        self.last_self_play_stats = {
            "games_total": games_total,
            "avg_moves_per_game": avg_moves_per_game,
            "avg_placements_per_jump": avg_placements_per_jump,
            "avg_jump_sequence_len": avg_jump_seq_len,
            "avg_jump_length": avg_jump_length,
            "adjacent_conversion_rate": adj_conv_rate,
        }
        
        games_per_sec = games_total / elapsed if elapsed > 0 else 0
        print(
            f"  Self-play: {num_examples} examples from {games_total} games "
            f"({games_per_sec:.1f} games/sec, {elapsed:.1f}s) | "
            f"W1/W2/D={p1_wins}/{p2_wins}/{draws} | "
            f"avg_moves/game={avg_moves_per_game:.1f}, "
            f"avg_jump_seq={avg_jump_seq_len:.2f}, "
            f"avg_jump_len={avg_jump_length:.2f}, "
            f"adj_conv={adj_conv_rate*100:.1f}%"
        )

        return num_examples
    
    def run_training(self) -> dict:
        """Run training steps. Returns average metrics."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            print(f"  Skipping training: buffer has {len(self.replay_buffer)}/{self.config.min_buffer_size} examples")
            return {}
        
        start_time = time.time()
        metrics_sum = {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        for step_idx in range(self.config.train_steps_per_iteration):
            self.rng, step_rng = jax.random.split(self.rng)
            
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size_train)
            
            # Train step
            self.params, self.batch_stats, self.opt_state, metrics = self.train_step_fn(
                self.params, self.batch_stats, self.opt_state, batch, step_rng
            )
            
            # Accumulate metrics
            for k, v in metrics.items():
                metrics_sum[k] += float(v)
        
        elapsed = time.time() - start_time
        steps_per_sec = self.config.train_steps_per_iteration / elapsed
        
        # Average metrics
        avg_metrics = {k: v / self.config.train_steps_per_iteration for k, v in metrics_sum.items()}
        
        print(f"  Training: {self.config.train_steps_per_iteration} steps "
              f"({steps_per_sec:.1f} steps/sec) | "
              f"policy_loss: {avg_metrics['policy_loss']:.4f}, "
              f"value_loss: {avg_metrics['value_loss']:.4f}")
        
        return avg_metrics
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{self.iteration:06d}.pkl")
        
        checkpoint = {
            'params': self.params,
            'batch_stats': self.batch_stats,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_examples': self.total_examples,
            'config': self.config,
            'metrics_history': self.metrics_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint['params']
        self.batch_stats = checkpoint['batch_stats']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.total_examples = checkpoint['total_examples']
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        print(f"Loaded checkpoint from iteration {self.iteration}")
    
    def train(self):
        """Main training loop."""

        # Auto-resume from latest checkpoint if exists
        existing = glob.glob(os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl"))
        if existing:
            latest = max(existing, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            self.load_checkpoint(latest)
            self.iteration += 1
        print(f"Resuming from iteration {self.iteration}")
        print("=" * 60)
        print("AlphaZero Training for Phutball (Batched)")
        print("=" * 60)
        print(f"Board: {self.config.rows}x{self.config.cols}")
        print(f"Network: {self.config.num_channels} channels, {self.config.num_res_blocks} res blocks")
        print(f"Self-play: {self.config.games_per_iteration} games/iter (batch={self.config.batch_size_games})")
        print(f"Training: {self.config.train_steps_per_iteration} steps/iter (batch={self.config.batch_size_train})")
        print(f"Devices: {jax.devices()}")
        print("=" * 60)
        print()
        
        for iteration in range(self.iteration, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()
            
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print("-" * 40)
            
            # Self-play
            num_examples = self.run_self_play()
            
            # Training
            metrics = self.run_training()
            
            # Record metrics
            if metrics:
                self.metrics_history.append({
                    'iteration': iteration,
                    'total_games': self.total_games,
                    'total_examples': self.total_examples,
                    'buffer_size': len(self.replay_buffer),
                    **metrics,
                })
            
            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint()

                if self.config.eval_enable:
                    # First check vs random
                    win_rate, stats = self.evaluate_vs_random_batched()
                    threshold = self.config.eval_vs_random_threshold
                    if threshold is not None:
                        # Threshold is active: gate the expensive comparison
                        if win_rate >= threshold:
                            print(
                                f"  [eval] Win rate {win_rate:.1%} >= {threshold:.0%}, "
                                f"running checkpoint comparison..."
                            )
                            self.evaluate_current_checkpoint_vs_recent()
                        else:
                            print(
                                f"  [eval] Win rate {win_rate:.1%} < {threshold:.0%}, "
                                f"skipping checkpoint comparison"
                            )

                        # Optional early stop if we are configured to do so
                        if self.config.stop_when_beating_random and win_rate >= threshold:
                            print(
                                f"  [train] Reached eval_vs_random_threshold={threshold:.0%}, "
                                f"stopping training early for this board size."
                            )
                            break
                    else:
                        # No threshold configured: don't gate on it at all.
                        # Behave like the old code: always run checkpoint-vs-checkpoint eval.
                        print(
                            "  [eval] eval_vs_random_threshold not set; "
                            "running checkpoint comparison unconditionally."
                        )
                        self.evaluate_current_checkpoint_vs_recent()


            # iteration timing + wandb logging
            iter_time = time.time() - iter_start
            print(f"  Iteration time: {iter_time:.1f}s | Buffer: {len(self.replay_buffer)} examples")
            print()

            if metrics and self.config.use_wandb and self.wandb_run is not None:
                global_step = (iteration + 1) * self.config.train_steps_per_iteration

                log_data = {
                    "iteration": iteration,
                    "total_games": self.total_games,
                    "total_examples": self.total_examples,
                    "buffer_size": len(self.replay_buffer),
                    "selfplay/examples_per_iter": num_examples,
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/total_loss": metrics["total_loss"],
                    "time/iteration_sec": iter_time,
                }

                # Phutball-specific stats if available
                if self.last_self_play_stats is not None:
                    for k, v in self.last_self_play_stats.items():
                        log_data[f"selfplay/{k}"] = v

                wandb.log(log_data, step=global_step)

        # Final checkpoint
        self.save_checkpoint()
        print("Training complete!")
        print(f"Total games: {self.total_games}")
        print(f"Total examples: {self.total_examples}")

    
    def evaluate_current_checkpoint_vs_recent(self):
        """
        After saving the *current* checkpoint, pit it against up to the
        last K previous checkpoints using the batched evaluator.

        For each opponent:
        - games_per_color with current as P1 (home=P1, away=P2)
        - games_per_color with current as P2 (home=P2, away=P1)

        All 2 * games_per_color games vs that opponent are played in a
        single batched call to play_match_batched.
        """
        if not self.config.eval_enable:
            return

        import glob, os, pickle
        import jax
        import numpy as np

        max_prev = self.config.eval_max_prev_checkpoints
        games_per_color = self.config.eval_games_per_color
        if max_prev <= 0 or games_per_color <= 0:
            return

        pattern = os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl")
        ckpt_paths = sorted(glob.glob(pattern))
        if len(ckpt_paths) < 2:
            return  # need at least current + one opponent

        # Current is the most recently saved
        current_path = ckpt_paths[-1]
        # Up to max_prev most recent previous checkpoints
        prev_paths = ckpt_paths[-(max_prev + 1):-1]
        prev_paths = list(reversed(prev_paths))  # newest first

        # Load current params once
        with open(current_path, "rb") as f:
            current_ckpt = pickle.load(f)
        current_params = {
            "network_params": current_ckpt["params"],
            "batch_stats": current_ckpt["batch_stats"],
        }
        current_name = os.path.basename(current_path)

        # Eval hyperparams from config
        num_simulations = self.config.eval_num_simulations
        max_moves = self.config.eval_max_moves
        temperature = self.config.eval_temperature

        print(f"  [eval] Batched evaluation for {current_name} vs {len(prev_paths)} recent checkpoints...")

        rng = self.rng  # thread rng through

        for opp_path in prev_paths:
            opp_name = os.path.basename(opp_path)
            with open(opp_path, "rb") as f:
                opp_ckpt = pickle.load(f)
            opp_params = {
                "network_params": opp_ckpt["params"],
                "batch_stats": opp_ckpt["batch_stats"],
            }

            rng, match_rng = jax.random.split(rng)

            (
                total_score_home,
                total_games,
                wins_home,
                draws,
                wins_away,
                per_game_turns,
                per_game_jumps_p1,
                per_game_jumps_p2,
                per_game_jumps_total,
                per_game_removed,
                winners,
                home_is_P1,
            ) = play_match_batched(
                home_params=current_params,   # treat A as "home"
                away_params=opp_params,       # treat B as "away"
                rng=match_rng,
                network=self.network,
                env_config=self.env_config,
                games_per_color=games_per_color,
                max_moves=max_moves,
                num_simulations=num_simulations,
                temperature=temperature,
            )

            # ---- convert to numpy / python ----
            total_score_home = float(total_score_home)
            total_games = int(total_games)
            wins_home = int(wins_home)
            wins_away = int(wins_away)
            draws = int(draws)

            per_game_turns = np.asarray(per_game_turns, dtype=np.int32)
            per_game_jumps_p1 = np.asarray(per_game_jumps_p1, dtype=np.int32)
            per_game_jumps_p2 = np.asarray(per_game_jumps_p2, dtype=np.int32)
            per_game_jumps_total = np.asarray(per_game_jumps_total, dtype=np.int32)
            per_game_removed = np.asarray(per_game_removed, dtype=np.int32)
            winners = np.asarray(winners, dtype=np.int32)
            home_is_P1 = np.asarray(home_is_P1, dtype=bool)

            assert total_games == len(per_game_turns)

            # Map P1/P2 jump counts to home/away
            jumps_home = np.where(home_is_P1, per_game_jumps_p1, per_game_jumps_p2)
            jumps_away = per_game_jumps_p1 + per_game_jumps_p2 - jumps_home

            avg_score_home = total_score_home / total_games if total_games > 0 else 0.0
            avg_turns = float(per_game_turns.mean())
            avg_jumps_home = float(jumps_home.mean())
            avg_jumps_away = float(jumps_away.mean())

            # Average jump length across all jumps in this match
            jump_mask = per_game_jumps_total > 0
            if jump_mask.any():
                total_removed = int(per_game_removed[jump_mask].sum())
                total_jumps = int(per_game_jumps_total[jump_mask].sum())
                avg_jump_len = total_removed / total_jumps
            else:
                avg_jump_len = 0.0

            # ---- aggregate line ----
            print(
                f"    [eval] {current_name} (home) vs {opp_name} (away): "
                f"W-D-L = {wins_home}-{draws}-{wins_away} "
                f"({total_score_home:+.1f} / {total_games}, avg {avg_score_home:+.3f})"
            )
            print(
                f"      turns/game={avg_turns:.1f}, "
                f"jumps/game home={avg_jumps_home:.2f}, away={avg_jumps_away:.2f}, "
                f"avg_jump_len={avg_jump_len:.2f}"
            )

            # ---- per-game detailed lines ----
            for g in range(total_games):
                if winners[g] == 0:
                    outcome = "draw"
                else:
                    home_won = (
                        (winners[g] == 1 and home_is_P1[g]) or
                        (winners[g] == 2 and not home_is_P1[g])
                    )
                    outcome = "home win" if home_won else "away win"

                role_str = "home=P1,away=P2" if home_is_P1[g] else "home=P2,away=P1"

                if per_game_jumps_total[g] > 0:
                    game_avg_jump_len = per_game_removed[g] / per_game_jumps_total[g]
                else:
                    game_avg_jump_len = 0.0

                print(
                    f"        game {g:2d}: {role_str}, "
                    f"turns={per_game_turns[g]}, "
                    f"jumps_home={int(jumps_home[g])}, jumps_away={int(jumps_away[g])}, "
                    f"avg_jump_len={game_avg_jump_len:.2f}, "
                    f"result={outcome}"
                )

        # persist RNG so eval randomness keeps moving forward
        self.rng = rng
    

    def evaluate_vs_random_batched(self):
        """
        Evaluate current checkpoint vs random, side-aware.
        Returns (overall_win_rate, stats_dict).
        """
        self.rng, eval_rng = jax.random.split(self.rng)

        (
            p1_wins, p1_draws, p1_losses,
            p2_wins, p2_draws, p2_losses,
            turns,
        ) = play_vs_random_batched(
            checkpoint_params=self.get_network_params(),
            rng=eval_rng,
            network=self.network,
            env_config=self.env_config,
            num_games=self.config.eval_vs_random_games,
            max_moves=self.config.eval_max_moves,
            num_simulations=self.config.eval_num_simulations,
            temperature=self.config.eval_temperature,
        )

        # Make sure these are Python ints
        p1_wins   = int(p1_wins)
        p1_draws  = int(p1_draws)
        p1_losses = int(p1_losses)
        p2_wins   = int(p2_wins)
        p2_draws  = int(p2_draws)
        p2_losses = int(p2_losses)

        # Totals
        total_p1 = p1_wins + p1_draws + p1_losses
        total_p2 = p2_wins + p2_draws + p2_losses
        total    = total_p1 + total_p2

        # Overall (checkpoint POV)
        total_wins   = p1_wins + p2_wins
        total_draws  = p1_draws + p2_draws
        total_losses = p1_losses + p2_losses

        win_rate_overall  = total_wins / total if total > 0 else 0.0
        draw_rate_overall = total_draws / total if total > 0 else 0.0
        loss_rate_overall = total_losses / total if total > 0 else 0.0

        # Per-side win rates (checkpoint as P1 / P2)
        win_rate_p1 = p1_wins / total_p1 if total_p1 > 0 else 0.0
        win_rate_p2 = p2_wins / total_p2 if total_p2 > 0 else 0.0

        # “Score” (win - loss) / total, overall and per-side
        score_overall = (total_wins - total_losses) / total if total > 0 else 0.0
        score_p1 = (p1_wins - p1_losses) / total_p1 if total_p1 > 0 else 0.0
        score_p2 = (p2_wins - p2_losses) / total_p2 if total_p2 > 0 else 0.0
        side_bias = score_p1 - score_p2  # positive => better as P1

        avg_turns = float(jnp.mean(turns))

        print(
            "  [eval] vs random (side-aware):\n"
            f"    as P1: {p1_wins}-{p1_draws}-{p1_losses} "
            f"(win: {win_rate_p1:.1%}, score: {score_p1:.3f})\n"
            f"    as P2: {p2_wins}-{p2_draws}-{p2_losses} "
            f"(win: {win_rate_p2:.1%}, score: {score_p2:.3f})\n"
            f"    combined: win {win_rate_overall:.1%}, draw {draw_rate_overall:.1%}, "
            f"loss {loss_rate_overall:.1%}, score {score_overall:.3f}, "
            f"side_bias {side_bias:.3f}, avg turns {avg_turns:.1f}"
        )

        stats = {
            "total_games": total,
            "overall": {
                "wins": total_wins,
                "draws": total_draws,
                "losses": total_losses,
                "win_rate": win_rate_overall,
                "draw_rate": draw_rate_overall,
                "loss_rate": loss_rate_overall,
                "score": score_overall,
            },
            "p1": {
                "wins": p1_wins,
                "draws": p1_draws,
                "losses": p1_losses,
                "win_rate": win_rate_p1,
                "score": score_p1,
            },
            "p2": {
                "wins": p2_wins,
                "draws": p2_draws,
                "losses": p2_losses,
                "win_rate": win_rate_p2,
                "score": score_p2,
            },
            "side_bias": side_bias,
            "avg_turns": avg_turns,
        }

        return win_rate_overall, stats


    def run_round_robin(
        self,
        games_per_color: int = 5,
        max_checkpoints: int | None = None,
        num_simulations: int | None = None,
        max_moves: int | None = None,
        temperature: float | None = None,
    ):
        import glob, os, pickle, numpy as np
        import jax

        if num_simulations is None:
            num_simulations = self.config.eval_num_simulations
        if max_moves is None:
            max_moves = self.config.eval_max_moves
        if temperature is None:
            temperature = self.config.eval_temperature

        pattern = os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl")
        ckpt_paths = sorted(glob.glob(pattern))
        if max_checkpoints is not None:
            ckpt_paths = ckpt_paths[-max_checkpoints:]

        n = len(ckpt_paths)
        if n < 2:
            print("[round-robin] Need at least 2 checkpoints.")
            return [], None, None, None

        names = [os.path.basename(p) for p in ckpt_paths]
        print(f"[round-robin] Using {n} checkpoints:")
        for name in names:
            print(f"  - {name}")

        # Load params
        params_list = []
        for path in ckpt_paths:
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            params_list.append(
                {
                    "network_params": ckpt["params"],
                    "batch_stats": ckpt["batch_stats"],
                }
            )

        scores = np.zeros((n, n), dtype=float)  # avg score in [-1, 1]
        games_mat = np.zeros((n, n), dtype=int)

        # Optional W/D/L matrices if you care
        W_mat = np.zeros((n, n), dtype=int)
        D_mat = np.zeros((n, n), dtype=int)
        L_mat = np.zeros((n, n), dtype=int)

        total_pairs = n * (n - 1) // 2
        pair_idx = 0

        rng = self.rng

        for i in range(n):
            for j in range(i + 1, n):
                pair_idx += 1
                print(f"[round-robin] Pair {pair_idx}/{total_pairs}: {names[i]} (home) vs {names[j]} (away)")

                home_params = params_list[i]
                away_params = params_list[j]

                rng, match_rng = jax.random.split(rng)

                (total_score_home,
                total_games,
                home_wins,
                draws,
                away_wins,
                per_game_turns,
                per_game_jumps_p1,
                per_game_jumps_p2,
                per_game_jumps_total,
                per_game_removed,
                winners,
                home_is_P1) = play_match_batched(
                    home_params=home_params,
                    away_params=away_params,
                    rng=match_rng,
                    network=self.network,
                    env_config=self.env_config,
                    games_per_color=games_per_color,
                    max_moves=max_moves,
                    num_simulations=num_simulations,
                    temperature=temperature,
                )

                # --- Convert to numpy / python ---
                total_score_home = float(total_score_home)
                total_games = int(total_games)
                home_wins = int(home_wins)
                away_wins = int(away_wins)
                draws = int(draws)

                per_game_turns = np.array(per_game_turns, dtype=np.int32)
                per_game_jumps_p1 = np.array(per_game_jumps_p1, dtype=np.int32)
                per_game_jumps_p2 = np.array(per_game_jumps_p2, dtype=np.int32)
                per_game_jumps_total = np.array(per_game_jumps_total, dtype=np.int32)
                per_game_removed = np.array(per_game_removed, dtype=np.int32)
                winners = np.array(winners, dtype=np.int32)
                home_is_P1 = np.array(home_is_P1, dtype=bool)

                assert total_games == len(per_game_turns)

                # Map jumps to "home" and "away" for each game
                jumps_home = np.where(home_is_P1, per_game_jumps_p1, per_game_jumps_p2)
                jumps_away = per_game_jumps_p1 + per_game_jumps_p2 - jumps_home

                # Aggregate stats
                avg_score_home = total_score_home / total_games if total_games > 0 else 0.0
                avg_turns = float(per_game_turns.mean())

                total_jumps_home = jumps_home.sum()
                total_jumps_away = jumps_away.sum()
                avg_jumps_home = total_jumps_home / total_games
                avg_jumps_away = total_jumps_away / total_games

                jump_mask = per_game_jumps_total > 0
                if jump_mask.any():
                    total_removed = per_game_removed[jump_mask].sum()
                    total_jumps = per_game_jumps_total[jump_mask].sum()
                    avg_jump_len = total_removed / total_jumps
                else:
                    avg_jump_len = 0.0

                # Store results from HOME's POV
                scores[i, j] = avg_score_home
                scores[j, i] = -avg_score_home
                games_mat[i, j] = games_mat[j, i] = total_games

                W_mat[i, j] = home_wins
                L_mat[i, j] = away_wins
                D_mat[i, j] = draws
                W_mat[j, i] = away_wins
                L_mat[j, i] = home_wins
                D_mat[j, i] = draws

                print(
                    f"  [round-robin] {names[i]} (home) vs {names[j]} (away): "
                    f"home W-D-L = {home_wins}-{draws}-{away_wins} "
                    f"(score={total_score_home:+.1f} over {total_games} games, "
                    f"avg {avg_score_home:+.3f})"
                )
                print(
                    f"    turns/game={avg_turns:.1f}, "
                    f"jumps/game home={avg_jumps_home:.2f}, away={avg_jumps_away:.2f}, "
                    f"avg_jump_len={avg_jump_len:.2f}"
                )

                # --- Per-game detailed report ---
                for g in range(total_games):
                    if winners[g] == 0:
                        outcome = "draw"
                    else:
                        # Did home win this game?
                        home_won = (
                            (winners[g] == 1 and home_is_P1[g]) or
                            (winners[g] == 2 and not home_is_P1[g])
                        )
                        outcome = "home win" if home_won else "away win"

                    # How does P1/P2 map to home/away?
                    role_str = "P1=home,P2=away" if home_is_P1[g] else "P1=away,P2=home"

                    if per_game_jumps_total[g] > 0:
                        game_avg_jump_len = per_game_removed[g] / per_game_jumps_total[g]
                    else:
                        game_avg_jump_len = 0.0

                    print(
                        f"      game {g:2d}: {role_str}, "
                        f"turns={per_game_turns[g]}, "
                        f"jumps_home={int(jumps_home[g])}, jumps_away={int(jumps_away[g])}, "
                        f"avg_jump_len={game_avg_jump_len:.2f}, "
                        f"result={outcome}"
                    )

        self.rng = rng


        # --- Simple Elo-ish rating based on average scores ---
        ratings = np.zeros(n, dtype=float)
        K = 16.0

        def expected_score(r_i, r_j):
            return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))

        for i in range(n):
            for j in range(n):
                if i == j or games_mat[i, j] == 0:
                    continue
                # Convert avg score in [-1,1] to [0,1]
                s_01 = 0.5 * (scores[i, j] + 1.0)
                e_01 = expected_score(ratings[i], ratings[j])
                ratings[i] += K * (s_01 - e_01)

        return names, scores, games_mat, ratings, (W_mat, D_mat, L_mat)

def compute_elo_from_results(
    names: List[str],
    W: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    base_elo: float = 1500.0,
    K: float = 32.0,
    iters: int = 50,
) -> np.ndarray:
    """
    Simple multi-player Elo solver from a full W/D/L matrix.

    W[i,j] = wins of i vs j
    D[i,j] = draws
    L[i,j] = losses of i vs j

    Returns:
        ratings: np.ndarray of shape (N,)
    """
    n = len(names)
    ratings = np.full(n, base_elo, dtype=np.float64)

    for _ in range(iters):
        for i in range(n):
            for j in range(i + 1, n):
                # Aggregate results for this pair
                wins_ij  = float(W[i, j])
                draws_ij = float(D[i, j])
                loss_ij  = float(L[i, j])
                N_ij = wins_ij + draws_ij + loss_ij
                if N_ij == 0:
                    continue

                # Actual total score for i vs j (win=1, draw=0.5, loss=0)
                S_i = wins_ij + 0.5 * draws_ij

                # Expected score per game for i vs j
                diff = (ratings[j] - ratings[i]) / 400.0
                p_i = 1.0 / (1.0 + 10.0**diff)
                E_i = N_ij * p_i

                # Batch Elo update (equivalent to N_ij separate games)
                delta = K * (S_i - E_i) / max(1.0, N_ij)
                ratings[i] += delta
                ratings[j] -= delta

    return ratings


def evaluate_vs_random(trainer: AlphaZeroTrainer, num_games: int = 20) -> float:
    """Evaluate trained model against random play."""
    from mcts import MCTSConfig, select_action
    
    wins = 0
    
    for game_idx in range(num_games):
        state = reset(trainer.env_config)
        trainer.rng, game_rng = jax.random.split(trainer.rng)
        
        # Alternate who goes first
        trained_player = 1 if game_idx % 2 == 0 else 2
        
        move = 0
        while not state.terminated and move < 500:
            current_player = int(state.current_player)
            
            if current_player == trained_player:
                # Trained model's turn - use MCTS
                game_rng, action_rng = jax.random.split(game_rng)
                action, _, _ = select_action(
                    trainer.get_network_params(),
                    action_rng,
                    state,
                    trainer.network,
                    trainer.env_config,
                    MCTSConfig(num_simulations=50, max_num_considered_actions=16),
                    temperature=0.1,
                )
            else:
                # Random player's turn
                game_rng, action_rng = jax.random.split(game_rng)
                legal_mask = get_legal_actions(state, trainer.env_config)
                legal_actions = jnp.where(legal_mask == 1)[0]
                action = int(jax.random.choice(action_rng, legal_actions))
            
            state = step(state, jnp.array(action, dtype=jnp.int32), trainer.env_config)
            move += 1
        
        winner = int(state.winner)
        if winner == trained_player:
            wins += 1
    
    win_rate = wins / num_games
    print(f"Evaluation vs Random: {wins}/{num_games} wins ({win_rate*100:.1f}%)")
    return win_rate


def get_buffer_size(rows, cols, games_per_iter=256, target_staleness_iters=12):
    estimated_moves_per_game = (rows * cols) // 3 * 2
    examples_per_iter = games_per_iter * estimated_moves_per_game
    return examples_per_iter * target_staleness_iters

def get_train_steps(buffer_size, batch_size=256, target_epochs_per_iter=1):
    raw_steps = (buffer_size * target_epochs_per_iter) // batch_size
    rounded = round(raw_steps / 100) * 100
    return max(100, rounded)

def make_train_config(
    rows: int,
    cols: int,
    checkpoint_dir: str,
    num_iterations: int = 256,
    checkpoint_every: int = 20,
    use_wandb: bool = False,
    wandb_run_name: Optional[str] = None,
    eval_vs_random_threshold: Optional[float] = 0.90,
    stop_when_beating_random: bool = True,
    num_simulations: int = 50
) -> TrainConfig:
    """
    Convenience factory for TrainConfig so we don't duplicate hyperparams
    in every notebook/script.
    """

    buffer_size = get_buffer_size(rows, cols)
    train_steps = get_train_steps(buffer_size)


    return TrainConfig(
        # Environment
        rows=rows,
        cols=cols,

        # Network
        num_channels=64,
        num_res_blocks=8,

        # Self-play
        batch_size_games=128,
        games_per_iteration=256,
        max_turns_per_game=512,
        max_moves_per_game=512,
        temperature=1.0,
        num_simulations=num_simulations,

        # Training
        batch_size_train=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        train_steps_per_iteration=train_steps,

        # Replay buffer
        buffer_size=buffer_size,
        min_buffer_size=1_000,

        # Iterations / checkpoints
        num_iterations=num_iterations,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        log_every=1,

        # Eval vs random
        eval_enable=True,
        eval_vs_random_games=100,
        eval_vs_random_threshold=eval_vs_random_threshold,
        stop_when_beating_random=stop_when_beating_random,
        eval_max_prev_checkpoints=5,
        eval_games_per_color=5,
        eval_num_simulations=128,
        eval_max_moves=512,
        eval_temperature=0.0,

        # Wandb
        use_wandb=use_wandb,
        wandb_project="phutball-az",
        wandb_run_name=wandb_run_name,
        wandb_mode="online" if use_wandb else "disabled",
    )




# ============================================================================
# Quick test
# ============================================================================

def test_training_loop():
    """Quick test of training loop with tiny config."""
    config = TrainConfig(
        # Small board for testing
        rows=9,
        cols=9,
        # Tiny network
        num_channels=16,
        num_res_blocks=2,
        # Batched self-play
        batch_size_games=8,
        max_turns_per_game=30,   # 30 turns max
        max_moves_per_game=200,  # Memory cap
        games_per_iteration=16,  # 2 batches of 8
        # Minimal training
        train_steps_per_iteration=5,
        batch_size_train=8,
        buffer_size=500,
        min_buffer_size=10,
        # Short run
        num_iterations=3,
        checkpoint_every=100,  # Don't checkpoint during test
    )
    
    trainer = AlphaZeroTrainer(config)
    
    # Run training
    trainer.train()
    
    print("\n✓ Training loop test passed!")
    print(f"  Total games: {trainer.total_games}")
    print(f"  Total examples: {trainer.total_examples}")
    print(f"  Final buffer size: {len(trainer.replay_buffer)}")


if __name__ == "__main__":
    print("Testing batched training loop...\n")
    test_training_loop()