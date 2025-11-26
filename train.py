"""
AlphaZero Training Loop for Phutball.

This is the main script that orchestrates:
1. Self-play to generate training data
2. Neural network training
3. Evaluation against baselines
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
from mcts import MCTSConfig
from self_play import SelfPlayConfig, play_games, ReplayBuffer


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Environment
    rows: int = 21
    cols: int = 15
    
    # Network architecture
    num_channels: int = 64
    num_res_blocks: int = 6
    
    # MCTS
    num_simulations: int = 200
    max_num_considered_actions: int = 32
    
    # Self-play
    games_per_iteration: int = 100
    temp_threshold: int = 15
    max_moves_per_game: int = 7200
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100
    
    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000  # Min examples before training starts
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10  # iterations
    
    # Logging
    log_every: int = 1  # iterations
    
    # Total training
    num_iterations: int = 1000


class AlphaZeroTrainer:
    """Main training class."""
    
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
        
        # MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
        )
        
        # Self-play config
        self.self_play_config = SelfPlayConfig(
            temp_threshold=config.temp_threshold,
            max_moves=config.max_moves_per_game,
            mcts_config=self.mcts_config,
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
        """Get params dict for MCTS."""
        return {
            'network_params': self.params,
            'batch_stats': self.batch_stats,
        }
    
    def run_self_play(self) -> int:
        """Run self-play games and add to buffer. Returns number of new examples."""
        self.rng, games_rng = jax.random.split(self.rng)
        
        start_time = time.time()
        examples = play_games(
            params=self.get_network_params(),
            rng=games_rng,
            network=self.network,
            env_config=self.env_config,
            self_play_config=self.self_play_config,
            num_games=self.config.games_per_iteration,
        )
        elapsed = time.time() - start_time
        
        self.replay_buffer.add(examples)
        self.total_games += self.config.games_per_iteration
        self.total_examples += len(examples)
        
        games_per_sec = self.config.games_per_iteration / elapsed
        print(f"  Self-play: {len(examples)} examples from {self.config.games_per_iteration} games "
              f"({games_per_sec:.1f} games/sec)")
        
        return len(examples)
    
    def run_training(self) -> dict:
        """Run training steps. Returns average metrics."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            print(f"  Skipping training: buffer has {len(self.replay_buffer)}/{self.config.min_buffer_size} examples")
            return {}
        
        start_time = time.time()
        metrics_sum = {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        for step in range(self.config.train_steps_per_iteration):
            self.rng, step_rng = jax.random.split(self.rng)
            
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            
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
        print("AlphaZero Training for Phutball")
        print("=" * 60)
        print(f"Board: {self.config.rows}x{self.config.cols}")
        print(f"Network: {self.config.num_channels} channels, {self.config.num_res_blocks} res blocks")
        print(f"MCTS: {self.config.num_simulations} simulations")
        print(f"Training: {self.config.games_per_iteration} games/iter, {self.config.train_steps_per_iteration} train steps/iter")
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


def evaluate_vs_random(trainer: AlphaZeroTrainer, num_games: int = 20) -> float:
    """Evaluate trained model against random play."""
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
                from mcts import select_action
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
        # Small board
        rows=9,
        cols=9,
        # Tiny network
        num_channels=16,
        num_res_blocks=2,
        # Fast MCTS
        num_simulations=4,
        max_num_considered_actions=8,
        # Minimal training
        games_per_iteration=2,
        train_steps_per_iteration=5,
        batch_size=8,
        buffer_size=500,
        min_buffer_size=10,
        # Short run
        num_iterations=3,
        checkpoint_every=100,  # Don't checkpoint during test
    )
    
    # Override max_moves for test
    trainer = AlphaZeroTrainer(config)
    trainer.self_play_config.max_moves = 30  # Short games for test
    
    # Run training
    trainer.train()
    
    print("\n✓ Training loop test passed!")
    print(f"  Total games: {trainer.total_games}")
    print(f"  Total examples: {trainer.total_examples}")
    print(f"  Final buffer size: {len(trainer.replay_buffer)}")


if __name__ == "__main__":
    print("Testing training loop...\n")
    test_training_loop()