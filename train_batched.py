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

    use_wandb: bool = False
    wandb_project: str = "phutball-az"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


class AlphaZeroTrainer:
    """Main training class with batched self-play."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Environment config
        self.env_config = EnvConfig(rows=config.rows, cols=config.cols)
        
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
            "adjacency_conversions": 0,
        }
        
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


    def _eval_play_one_game(
        self,
        params_p1,
        params_p2,
        mcts_config,
        max_moves: int,
        temperature: float,
        seed: int,
    ) -> float:
        """
        Single evaluation game between params_p1 (player 1) and params_p2 (player 2).

        Returns:
            +1.0 if player 1 wins
            -1.0 if player 1 loses
             0.0 on draw / max-move cutoff
        """
        from mcts import select_action

        rng = jax.random.PRNGKey(seed)
        state = reset(self.env_config)
        move = 0

        while (not bool(state.terminated)) and (move < max_moves):
            current_player = int(state.current_player)
            agent_params = params_p1 if current_player == 1 else params_p2

            rng, mcts_rng = jax.random.split(rng)
            action, _, _ = select_action(
                agent_params,
                mcts_rng,
                state,
                self.network,
                self.env_config,
                mcts_config,
                temperature=temperature,
            )
            state = step(state, jnp.array(action, dtype=jnp.int32), self.env_config)
            move += 1

        winner = int(state.winner)
        if winner == 0 or move >= max_moves:
            return 0.0
        elif winner == 1:
            return 1.0
        else:
            return -1.0


    def evaluate_current_checkpoint_vs_recent(self):
        """
        After saving the *current* checkpoint, pit it against up to the
        last K previous checkpoints.

        For each opponent:
          - play eval_games_per_color games with current as P1
          - play eval_games_per_color games with current as P2

        So at most:
          K opponents * 2 * eval_games_per_color games.
        """
        if not self.config.eval_enable:
            return

        import glob
        from mcts import MCTSConfig

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

        # 🔹 define these once here
        mcts_config = MCTSConfig(
            num_simulations=self.config.eval_num_simulations,
            max_num_considered_actions=32,
            dirichlet_alpha=0.3,
            root_exploration_fraction=0.25,
        )
        max_moves = self.config.eval_max_moves
        temperature = self.config.eval_temperature

        print(f"  [eval] Batched evaluation for {current_name} vs {len(prev_paths)} recent checkpoints...")
        seed_counter = 0

        for opp_path in prev_paths:
            opp_name = os.path.basename(opp_path)
            with open(opp_path, "rb") as f:
                opp_ckpt = pickle.load(f)
            opp_params = {
                "network_params": opp_ckpt["params"],
                "batch_stats": opp_ckpt["batch_stats"],
            }

            score_current = 0.0
            total_games = 0
            wins_current = 0
            losses_current = 0
            draws_current = 0

            # current as P1, opponent as P2
            for g in range(games_per_color):
                seed_counter += 1
                res = self._eval_play_one_game(
                    current_params,
                    opp_params,
                    mcts_config,
                    max_moves,
                    temperature,
                    seed=1000 + seed_counter,
                )
                # res from current's perspective
                score_current += res
                total_games += 1
                if res > 0:
                    wins_current += 1
                elif res < 0:
                    losses_current += 1
                else:
                    draws_current += 1

            # opponent as P1, current as P2 (flip perspective)
            for g in range(games_per_color):
                seed_counter += 1
                res_opp = self._eval_play_one_game(
                    opp_params,
                    current_params,
                    mcts_config,
                    max_moves,
                    temperature,
                    seed=2000 + seed_counter,
                )
                res = -res_opp  # flip back to "current" perspective
                score_current += res
                total_games += 1
                if res > 0:
                    wins_current += 1
                elif res < 0:
                    losses_current += 1
                else:
                    draws_current += 1

            avg_score = score_current / total_games if total_games > 0 else 0.0
            print(
                f"    [eval] {current_name} vs {opp_name}: "
                f"W-L-D = {wins_current}-{losses_current}-{draws_current} "
                f"({score_current:+.1f} / {total_games}, avg {avg_score:+.3f})"
            )

def run_round_robin(
    self,
    games_per_color: int = 5,
    max_checkpoints: int | None = None,
    num_simulations: int | None = None,
    max_moves: int | None = None,
    temperature: float | None = None,
):
    """
    Run a round-robin tournament over checkpoints in checkpoint_dir,
    using play_match_batched (which returns total_score_A and total_games).

    Returns:
        names:   list[str] of checkpoint filenames
        scores:  (N, N) matrix, scores[i, j] = avg score for i vs j in [-1, 1]
        games:   (N, N) matrix, games[i, j] = number of games in that pairing
        ratings: (N,) array of Elo-like ratings
    """
    import glob
    import os
    import numpy as np
    import pickle
    import jax

    # Use eval settings by default
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

    # Load all params once
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

    # scores[i, j] = avg score per game for i vs j in [-1, 1]
    scores = np.zeros((n, n), dtype=float)
    games_mat = np.zeros((n, n), dtype=int)

    total_pairs = n * (n - 1) // 2
    pair_idx = 0

    rng = self.rng  # reuse trainer RNG

    for i in range(n):
        for j in range(i + 1, n):
            pair_idx += 1
            print(f"[round-robin] Pair {pair_idx}/{total_pairs}: {names[i]} vs {names[j]}")

            params_i = params_list[i]
            params_j = params_list[j]

            # Each call plays 2 * games_per_color games internally
            rng, match_rng = jax.random.split(rng)

            total_score_A, total_games = play_match_batched(
                params_A=params_i,
                params_B=params_j,
                rng=match_rng,
                network=self.network,
                env_config=self.env_config,
                games_per_color=games_per_color,
                max_moves=max_moves,
                num_simulations=num_simulations,
                temperature=temperature,
            )

            total_score_A = float(total_score_A)
            total_games = int(total_games)
            avg_score_A = total_score_A / total_games  # [-1, 1]

            scores[i, j] = avg_score_A
            scores[j, i] = -avg_score_A  # symmetric
            games_mat[i, j] = games_mat[j, i] = total_games

            print(
                f"  [round-robin] {names[i]} vs {names[j]}: "
                f"score={total_score_A:+.1f} over {total_games} games "
                f"(avg {avg_score_A:+.3f})"
            )

    # Store updated RNG back into trainer
    self.rng = rng

    # --- Simple Elo-style ratings from score matrix ---

    ratings = np.zeros(n, dtype=float)
    K = 16.0

    def expected_score(r_i, r_j):
        return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))

    # Our avg_score in [-1, 1]; convert to [0, 1] like win/draw/loss score:
    #   -1 -> 0.0, 0 -> 0.5, +1 -> 1.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if games_mat[i, j] == 0:
                continue

            avg_score = scores[i, j]  # [-1, 1]
            s_01 = 0.5 * (avg_score + 1.0)  # [0, 1]
            e_01 = expected_score(ratings[i], ratings[j])
            ratings[i] += K * (s_01 - e_01)

    return names, scores, games_mat, ratings



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