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
from dataclasses import dataclass
from typing import Optional
import pickle

from phutball_env_jax import EnvConfig, reset, step, get_legal_actions
from network import (
    PhutballNetwork, create_network, init_network,
    create_optimizer, make_train_step_fn, predict
)
from self_play_batched import (
    play_games_batched, trajectory_to_training_examples, ReplayBuffer
)


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
    max_moves_per_game: int = 300    # Max moves before game ends
    temperature: float = 1.0          # Sampling temperature
    
    # Training
    batch_size_train: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100
    
    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000  # Min examples before training starts
    
    # Iterations
    num_iterations: int = 500
    games_per_iteration: int = 256   # Total games per iteration (multiple batches)
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10  # iterations
    
    # Logging
    log_every: int = 1


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
                max_moves=self.config.max_moves_per_game,
                temperature=self.config.temperature,
            )
            
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
        
        games_per_sec = games_total / elapsed if elapsed > 0 else 0
        print(f"  Self-play: {num_examples} examples from {games_total} games "
              f"({games_per_sec:.1f} games/sec, {elapsed:.1f}s)")
        
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
            
            iter_time = time.time() - iter_start
            print(f"  Iteration time: {iter_time:.1f}s | Buffer: {len(self.replay_buffer)} examples")
            print()
        
        # Final checkpoint
        self.save_checkpoint()
        print("Training complete!")
        print(f"Total games: {self.total_games}")
        print(f"Total examples: {self.total_examples}")


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
        max_moves_per_game=50,
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