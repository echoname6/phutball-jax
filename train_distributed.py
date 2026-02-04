"""
Distributed (multi-TPU/GPU) training for Phutball AlphaZero.

Supports:
- Data-parallel self-play across devices
- Data-parallel training with gradient aggregation
- Chimera network with configurable device allocation per board size
- Flexible device assignment strategies
"""

import jax
import jax.numpy as jnp
from jax import pmap, lax
from jax.sharding import PositionalSharding
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import os
import pickle

from phutball_env_jax import PhutballState, EnvConfig, reset, step, get_legal_actions
from network import (
    PhutballNetwork, create_network, init_network, create_optimizer,
    ChimeraNetwork, create_chimera_network, init_chimera_network,
)
from self_play_batched import (
    play_games_batched, trajectory_to_training_examples,
    TrajectoryData, compute_phutball_stats,
)
from train_batched import (
    ReplayBuffer, generate_curriculum_batch,
)


# =============================================================================
# Global Recent Buffer
# =============================================================================

class GlobalRecentBuffer:
    """
    Global buffer for recent examples across all board sizes.

    - Stores most recent N examples globally (FIFO)
    - Each example tagged with board size for proper head routing
    - Enables fast adaptation and cross-board learning signals
    - Can be fed from self-play OR external sources (human games, APIs)
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
        self.board_keys = []  # Which board size each example belongs to
        self.sources = []  # "self_play", "human", "api", etc.
        self.timestamps = []

    def add(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        board_key: str,
        source: str = "self_play",
    ):
        """Add examples to the global buffer."""
        n = len(states)
        timestamp = time.time()

        for i in range(n):
            self.states.append(states[i])
            self.policies.append(policies[i])
            self.values.append(values[i])
            self.board_keys.append(board_key)
            self.sources.append(source)
            self.timestamps.append(timestamp)

        # Trim if over capacity (FIFO)
        while len(self.states) > self.max_size:
            self.states.pop(0)
            self.policies.pop(0)
            self.values.pop(0)
            self.board_keys.pop(0)
            self.sources.pop(0)
            self.timestamps.pop(0)

    def sample(
        self,
        batch_size: int,
        board_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Sample from the buffer.

        Args:
            batch_size: Number of examples to sample
            board_key: If provided, only sample from this board size.
                       If None, sample from all boards (for value head training).

        Returns:
            states, policies, values, board_keys
        """
        if len(self.states) == 0:
            return None, None, None, []

        # Filter by board key if specified
        if board_key is not None:
            indices = [i for i, k in enumerate(self.board_keys) if k == board_key]
        else:
            indices = list(range(len(self.states)))

        if len(indices) == 0:
            return None, None, None, []

        # Sample
        sample_size = min(batch_size, len(indices))
        sampled_indices = np.random.choice(indices, size=sample_size, replace=False)

        states = np.array([self.states[i] for i in sampled_indices])
        policies = np.array([self.policies[i] for i in sampled_indices])
        values = np.array([self.values[i] for i in sampled_indices])
        keys = [self.board_keys[i] for i in sampled_indices]

        return states, policies, values, keys

    def sample_by_board(
        self,
        batch_size_per_board: int,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample separately for each board size.

        Useful for Chimera training where policy heads need board-specific data.
        """
        result = {}
        unique_boards = set(self.board_keys)

        for board_key in unique_boards:
            states, policies, values, _ = self.sample(batch_size_per_board, board_key)
            if states is not None and len(states) > 0:
                result[board_key] = (states, policies, values)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        board_counts = {}
        source_counts = {}

        for k in self.board_keys:
            board_counts[k] = board_counts.get(k, 0) + 1

        for s in self.sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "total_examples": len(self.states),
            "max_size": self.max_size,
            "board_counts": board_counts,
            "source_counts": source_counts,
        }

    def __len__(self):
        return len(self.states)


class ExternalDataIngestion:
    """
    API for external data ingestion (human games, partner submissions).

    Validates and adds games to the appropriate buffers.
    """

    def __init__(
        self,
        global_buffer: GlobalRecentBuffer,
        board_buffers: Dict[str, ReplayBuffer],
        env_configs: Dict[str, EnvConfig],
    ):
        self.global_buffer = global_buffer
        self.board_buffers = board_buffers
        self.env_configs = env_configs

    def submit_game(
        self,
        board_key: str,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        source: str = "external",
        quality_score: float = 1.0,
    ) -> bool:
        """
        Submit a game's worth of training data.

        Args:
            board_key: Board size (e.g., "11x9")
            states: Array of states (N, channels, rows, cols)
            policies: Array of policies (N, action_space)
            values: Array of values (N,)
            source: Data source identifier
            quality_score: Weight for this data (for future prioritized replay)

        Returns:
            True if accepted, False if rejected
        """
        if board_key not in self.env_configs:
            print(f"Unknown board size: {board_key}")
            return False

        # Validate shapes
        env_config = self.env_configs[board_key]
        expected_action_space = 2 * env_config.rows * env_config.cols + 1

        if policies.shape[-1] != expected_action_space:
            print(f"Invalid policy shape for {board_key}")
            return False

        # Add to both global and board-specific buffers
        self.global_buffer.add(states, policies, values, board_key, source)

        if board_key in self.board_buffers:
            self.board_buffers[board_key].add(states, policies, values)

        return True

    def export_data(
        self,
        board_key: str,
        num_examples: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Export training data for a board size.

        For companies/partners that want to acquire data.
        """
        states, policies, values, _ = self.global_buffer.sample(num_examples, board_key)
        return states, policies, values


# =============================================================================
# Device Management
# =============================================================================

@dataclass
class DeviceConfig:
    """Configuration for device allocation."""
    # Total devices available
    num_devices: int = field(default_factory=lambda: jax.device_count())

    # Device allocation strategy: "uniform", "weighted", "manual"
    strategy: str = "uniform"

    # For "weighted" strategy: relative weights per board size
    # e.g., {"11x9": 1, "15x11": 2, "21x15": 4} means 21x15 gets 4x the devices
    board_weights: Dict[str, int] = field(default_factory=dict)

    # For "manual" strategy: explicit device indices per board size
    # e.g., {"11x9": [0, 1], "15x11": [2, 3], "21x15": [4, 5, 6, 7]}
    board_devices: Dict[str, List[int]] = field(default_factory=dict)

    # Games per device during self-play
    games_per_device: int = 64

    # Training batch size per device
    train_batch_per_device: int = 128


def get_device_allocation(
    config: DeviceConfig,
    board_sizes: List[Tuple[int, int]],
) -> Dict[str, List[int]]:
    """
    Compute device allocation for each board size.

    Returns dict mapping "ROWSxCOLS" to list of device indices.
    """
    devices = list(range(config.num_devices))
    board_keys = [f"{r}x{c}" for r, c in board_sizes]

    if config.strategy == "manual":
        # Use explicit allocation
        allocation = {}
        for key in board_keys:
            if key in config.board_devices:
                allocation[key] = config.board_devices[key]
            else:
                # Default to all devices if not specified
                allocation[key] = devices
        return allocation

    elif config.strategy == "weighted":
        # Allocate proportionally based on weights
        total_weight = sum(config.board_weights.get(k, 1) for k in board_keys)
        allocation = {}
        device_idx = 0

        for key in board_keys:
            weight = config.board_weights.get(key, 1)
            num_devices = max(1, int(config.num_devices * weight / total_weight))
            allocation[key] = devices[device_idx:device_idx + num_devices]
            device_idx += num_devices

        # Assign remaining devices to last board
        if device_idx < config.num_devices:
            allocation[board_keys[-1]].extend(devices[device_idx:])

        return allocation

    else:  # "uniform"
        # All boards get all devices (replicated training)
        return {key: devices for key in board_keys}


# =============================================================================
# Distributed Self-Play
# =============================================================================

def make_pmap_play_fn(
    network: PhutballNetwork,
    env_config: EnvConfig,
    max_turns: int,
    max_moves: int,
    num_simulations: int,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    temp_final: float = 0.1,
) -> Callable:
    """
    Create a pmap'd self-play function.

    Each device runs `games_per_device` games in parallel.
    """

    def play_fn(params, rng, games_per_device):
        """Single-device self-play."""
        return play_games_batched(
            params=params,
            rng=rng,
            network=network,
            env_config=env_config,
            batch_size=games_per_device,
            max_turns=max_turns,
            max_moves=max_moves,
            temperature=temperature,
            temp_threshold=temp_threshold,
            temp_final=temp_final,
            num_simulations=num_simulations,
        )

    # pmap across devices, each gets different rng
    return pmap(play_fn, in_axes=(0, 0, None), static_broadcasted_argnums=(2,))


def distributed_self_play(
    pmap_play_fn: Callable,
    params: dict,
    rng: jnp.ndarray,
    num_devices: int,
    games_per_device: int,
) -> List[TrajectoryData]:
    """
    Run self-play across multiple devices.

    Returns list of trajectories, one per device.
    """
    # Replicate params to all devices
    replicated_params = jax.device_put_replicated(params, jax.devices()[:num_devices])

    # Split RNG for each device
    device_rngs = jax.random.split(rng, num_devices)

    # Run on all devices
    trajectories = pmap_play_fn(replicated_params, device_rngs, games_per_device)

    return trajectories


def merge_trajectories(trajectories: TrajectoryData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge trajectories from multiple devices into training examples.

    trajectories has shape (num_devices, ...) from pmap output.
    """
    num_devices = trajectories.states.shape[0]

    all_states = []
    all_policies = []
    all_values = []

    for d in range(num_devices):
        # Extract single-device trajectory
        device_traj = TrajectoryData(
            states=trajectories.states[d],
            policies=trajectories.policies[d],
            players=trajectories.players[d],
            valid_mask=trajectories.valid_mask[d],
            winners=trajectories.winners[d],
            actions=trajectories.actions[d],
        )

        states, policies, values = trajectory_to_training_examples(device_traj)
        all_states.append(states)
        all_policies.append(policies)
        all_values.append(values)

    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_policies, axis=0),
        np.concatenate(all_values, axis=0),
    )


# =============================================================================
# Distributed Training
# =============================================================================

def make_pmap_train_step(
    network: PhutballNetwork,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """
    Create a pmap'd training step with gradient aggregation.
    For standard (non-Chimera) networks.
    """

    def loss_fn(params, batch_stats, states, policies, values):
        """Compute combined policy and value loss."""
        variables = {'params': params, 'batch_stats': batch_stats}
        (policy_logits, value_preds), mutated_vars = network.apply(
            variables, states, train=True, mutable=['batch_stats']
        )

        # Policy loss (cross-entropy)
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            policy_logits, jnp.argmax(policies, axis=-1)
        ).mean()

        # Value loss (MSE)
        value_loss = jnp.mean((value_preds.squeeze() - values) ** 2)

        total_loss = policy_loss + value_loss

        return total_loss, (policy_loss, value_loss, mutated_vars['batch_stats'])

    def train_step(params, batch_stats, opt_state, states, policies, values):
        """Single training step."""
        (loss, (policy_loss, value_loss, new_batch_stats)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, batch_stats, states, policies, values)

        # Average gradients across devices (pmean)
        grads = lax.pmean(grads, axis_name='devices')

        # Apply updates
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Also sync batch stats
        new_batch_stats = lax.pmean(new_batch_stats, axis_name='devices')

        return new_params, new_batch_stats, new_opt_state, loss, policy_loss, value_loss

    return pmap(train_step, axis_name='devices')


def make_chimera_train_step(
    network: ChimeraNetwork,
    optimizer: optax.GradientTransformation,
    board_sizes: Tuple[Tuple[int, int], ...],
) -> Callable:
    """
    Create a training step for Chimera network.

    Key design:
    - Value head is SHARED: trained on examples from ALL board sizes
    - Policy heads are SEPARATE: each trained only on its board's examples
    - Backbone is SHARED: gets gradients from all board sizes

    Args:
        network: ChimeraNetwork instance
        optimizer: Optax optimizer
        board_sizes: Tuple of (rows, cols) for each board

    The training data should be organized as a dict:
        {
            "11x9": (states, policies, values),
            "15x11": (states, policies, values),
            ...
        }
    """
    board_keys = [f"{r}x{c}" for r, c in board_sizes]

    def chimera_loss_fn(params, batch_stats, board_data: Dict[str, Tuple]):
        """
        Compute Chimera loss across all board sizes.

        - Policy loss: per-board, using each board's policy head
        - Value loss: combined from all boards (shared value head learning)
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_examples = 0
        new_batch_stats = batch_stats

        for board_key in board_keys:
            if board_key not in board_data:
                continue

            states, policies, values = board_data[board_key]
            batch_size = states.shape[0]

            if batch_size == 0:
                continue

            variables = {'params': params, 'batch_stats': new_batch_stats}

            # Forward pass with board-specific head
            (policy_logits, value_preds), mutated_vars = network.apply(
                variables, states, board_key=board_key, train=True, mutable=['batch_stats']
            )
            new_batch_stats = mutated_vars['batch_stats']

            # Policy loss for this board's head
            board_policy_loss = optax.softmax_cross_entropy_with_integer_labels(
                policy_logits, jnp.argmax(policies, axis=-1)
            ).sum()

            # Value loss (contributes to shared value head)
            board_value_loss = jnp.sum((value_preds.squeeze() - values) ** 2)

            total_policy_loss += board_policy_loss
            total_value_loss += board_value_loss
            total_examples += batch_size

        # Average over all examples
        if total_examples > 0:
            avg_policy_loss = total_policy_loss / total_examples
            avg_value_loss = total_value_loss / total_examples
        else:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0

        total_loss = avg_policy_loss + avg_value_loss

        return total_loss, (avg_policy_loss, avg_value_loss, new_batch_stats)

    def train_step(params, batch_stats, opt_state, board_data: Dict[str, Tuple]):
        """Single Chimera training step."""
        (loss, (policy_loss, value_loss, new_batch_stats)), grads = jax.value_and_grad(
            chimera_loss_fn, has_aux=True
        )(params, batch_stats, board_data)

        # Apply updates
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_batch_stats, new_opt_state, loss, policy_loss, value_loss

    return train_step


def make_pmap_chimera_train_step(
    network: ChimeraNetwork,
    optimizer: optax.GradientTransformation,
    board_sizes: Tuple[Tuple[int, int], ...],
) -> Callable:
    """
    Create a pmap'd Chimera training step with gradient aggregation across devices.
    """
    base_train_step = make_chimera_train_step(network, optimizer, board_sizes)

    def pmap_train_step(params, batch_stats, opt_state, board_data):
        new_params, new_batch_stats, new_opt_state, loss, policy_loss, value_loss = \
            base_train_step(params, batch_stats, opt_state, board_data)

        # Sync across devices
        # Note: For Chimera with device allocation, might want selective sync
        return new_params, new_batch_stats, new_opt_state, loss, policy_loss, value_loss

    # pmap with gradient aggregation
    return pmap(
        lambda p, bs, os, bd: base_train_step(p, bs, os, bd),
        axis_name='devices',
    )


# =============================================================================
# Distributed Chimera Trainer
# =============================================================================

@dataclass
class DistributedChimeraConfig:
    """Configuration for distributed Chimera training."""

    # Board sizes to train on
    board_sizes: Tuple[Tuple[int, int], ...] = ((11, 9), (15, 11), (21, 15))

    # Device configuration
    device_config: DeviceConfig = field(default_factory=DeviceConfig)

    # Network architecture
    num_channels: int = 128
    num_res_blocks: int = 10

    # Self-play
    max_moves_per_game: Optional[int] = None  # Computed from board size if None
    num_simulations: int = 50
    temperature: float = 1.0
    temp_threshold: int = 30
    temp_final: float = 0.1

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 100

    # Replay buffer per board size
    buffer_size: int = 200000
    min_buffer_size: int = 1000

    # Global buffer sampling ratio (vs per-board historical)
    global_buffer_ratio: float = 0.3  # 30% from global recent buffer

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_initial_ratio: float = 0.5
    curriculum_final_ratio: float = 0.05
    curriculum_decay_iterations: int = 100
    curriculum_jump_distribution: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)

    # Iterations
    num_iterations: int = 500

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/distributed"
    checkpoint_every: int = 10

    # Logging
    use_wandb: bool = False
    wandb_project: str = "phutball-distributed"
    wandb_run_name: Optional[str] = None


class DistributedChimeraTrainer:
    """
    Distributed trainer for Chimera network.

    Supports:
    - Different device allocations per board size
    - Shared backbone with gradient aggregation from all board sizes
    - Shared value head trained on ALL games (learns general position evaluation)
    - Per-board policy heads trained on their respective board's games

    Training strategy:
    - Self-play runs on each board size (possibly on different device sets)
    - All examples feed into training
    - Backbone + value head get gradients from all board sizes
    - Each policy head only gets gradients from its board's examples
    """

    def __init__(self, config: DistributedChimeraConfig):
        self.config = config
        self.iteration = 0

        # Device setup
        self.num_devices = jax.device_count()
        print(f"Devices available: {self.num_devices}")
        print(f"Devices: {jax.devices()}")

        # Get device allocation per board size
        self.device_allocation = get_device_allocation(
            config.device_config,
            config.board_sizes,
        )
        print(f"Device allocation: {self.device_allocation}")

        # Environment configs per board size
        self.env_configs = {}
        self.max_moves = {}
        for rows, cols in config.board_sizes:
            key = f"{rows}x{cols}"
            max_moves = config.max_moves_per_game or (rows * cols * 2)
            self.env_configs[key] = EnvConfig(rows=rows, cols=cols)
            self.max_moves[key] = max_moves

        # Create Chimera network
        max_rows = max(r for r, c in config.board_sizes)
        max_cols = max(c for r, c in config.board_sizes)

        self.network = create_chimera_network(
            board_sizes=config.board_sizes,
            num_channels=config.num_channels,
            num_res_blocks=config.num_res_blocks,
        )

        # Initialize network
        self.rng = jax.random.PRNGKey(42)
        self.rng, init_rng = jax.random.split(self.rng)

        self.params, self.batch_stats = init_chimera_network(
            self.network,
            init_rng,
            config.board_sizes,
        )

        # Optimizer
        self.optimizer = create_optimizer(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.opt_state = self.optimizer.init(self.params)

        # Global recent buffer (shared across all boards)
        self.global_buffer = GlobalRecentBuffer(max_size=config.buffer_size)

        # Per-board historical buffers
        self.replay_buffers = {
            f"{r}x{c}": ReplayBuffer(max_size=config.buffer_size)
            for r, c in config.board_sizes
        }

        # External data ingestion API
        self.data_api = ExternalDataIngestion(
            global_buffer=self.global_buffer,
            board_buffers=self.replay_buffers,
            env_configs=self.env_configs,
        )

        # Create pmap functions per board size
        self.pmap_play_fns = {}
        self.pmap_train_fns = {}

        for rows, cols in config.board_sizes:
            key = f"{rows}x{cols}"

            # Self-play function for this board size
            # Note: For Chimera, we need a wrapper network for this board size
            self.pmap_play_fns[key] = make_pmap_play_fn(
                network=self.network,  # Will need board-specific wrapper
                env_config=self.env_configs[key],
                max_turns=self.max_moves[key],
                max_moves=self.max_moves[key],
                num_simulations=config.num_simulations,
                temperature=config.temperature,
                temp_threshold=config.temp_threshold,
                temp_final=config.temp_final,
            )

        # Stats tracking
        self.total_games = {k: 0 for k in self.env_configs.keys()}
        self.metrics_history = []

    def get_curriculum_ratio(self) -> float:
        """Compute current curriculum ratio based on iteration."""
        if not self.config.curriculum_enabled:
            return 0.0

        progress = min(1.0, self.iteration / self.config.curriculum_decay_iterations)
        ratio = self.config.curriculum_initial_ratio - progress * (
            self.config.curriculum_initial_ratio - self.config.curriculum_final_ratio
        )
        return max(0.0, ratio)

    def run_self_play_for_board(self, board_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Run distributed self-play for a specific board size.

        Returns (states, policies, values, stats).
        """
        devices = self.device_allocation[board_key]
        num_devices = len(devices)
        games_per_device = self.config.device_config.games_per_device

        self.rng, play_rng = jax.random.split(self.rng)

        # Get params for this board (Chimera uses shared backbone + specific head)
        params = {
            'network_params': self.params,
            'batch_stats': self.batch_stats,
        }

        # Run distributed self-play
        trajectories = distributed_self_play(
            self.pmap_play_fns[board_key],
            params,
            play_rng,
            num_devices,
            games_per_device,
        )

        # Merge trajectories from all devices
        states, policies, values = merge_trajectories(trajectories)

        # Add to BOTH global and per-board buffers
        self.global_buffer.add(states, policies, values, board_key, source="self_play")
        self.replay_buffers[board_key].add(states, policies, values)

        # Compute stats
        stats = {
            'games': num_devices * games_per_device,
            'examples': len(states),
        }

        return states, policies, values, stats

    def train_step_distributed(
        self,
        states: jnp.ndarray,
        policies: jnp.ndarray,
        values: jnp.ndarray,
    ) -> Tuple[float, float]:
        """
        Distributed training step with gradient aggregation.
        """
        # Replicate across devices
        num_devices = self.num_devices
        batch_size = len(states)
        per_device = batch_size // num_devices

        # Reshape for pmap: (num_devices, per_device, ...)
        states = states[:num_devices * per_device].reshape(num_devices, per_device, *states.shape[1:])
        policies = policies[:num_devices * per_device].reshape(num_devices, per_device, *policies.shape[1:])
        values = values[:num_devices * per_device].reshape(num_devices, per_device)

        # Replicate params
        replicated_params = jax.device_put_replicated(self.params, jax.devices())
        replicated_batch_stats = jax.device_put_replicated(self.batch_stats, jax.devices())
        replicated_opt_state = jax.device_put_replicated(self.opt_state, jax.devices())

        # Create pmap train step if not exists
        if not hasattr(self, '_pmap_train_step'):
            self._pmap_train_step = make_pmap_train_step(self.network, self.optimizer)

        # Run distributed training step
        new_params, new_batch_stats, new_opt_state, loss, policy_loss, value_loss = \
            self._pmap_train_step(
                replicated_params,
                replicated_batch_stats,
                replicated_opt_state,
                states,
                policies,
                values,
            )

        # Extract from first device (all are synchronized)
        self.params = jax.tree_map(lambda x: x[0], new_params)
        self.batch_stats = jax.tree_map(lambda x: x[0], new_batch_stats)
        self.opt_state = jax.tree_map(lambda x: x[0], new_opt_state)

        return float(policy_loss[0]), float(value_loss[0])

    def train_iteration(self):
        """Run one training iteration across all board sizes."""
        start_time = time.time()

        iteration_stats = {}

        # Self-play for each board size (adds to both global and per-board buffers)
        for board_key in self.env_configs.keys():
            states, policies, values, stats = self.run_self_play_for_board(board_key)
            self.total_games[board_key] += stats['games']
            iteration_stats[board_key] = stats

        # Training steps
        curriculum_ratio = self.get_curriculum_ratio()
        global_ratio = self.config.global_buffer_ratio
        total_policy_loss = 0.0
        total_value_loss = 0.0

        total_batch_size = self.config.device_config.train_batch_per_device * self.num_devices

        for step_idx in range(self.config.train_steps_per_iteration):
            all_states = []
            all_policies = []
            all_values = []

            # Split batch: global recent, per-board historical, curriculum
            global_size = int(total_batch_size * global_ratio)
            remaining = total_batch_size - global_size
            curriculum_size = int(remaining * curriculum_ratio)
            per_board_size = remaining - curriculum_size
            per_board_each = per_board_size // len(self.env_configs)

            # 1. Sample from GLOBAL recent buffer (all boards, recent data)
            if global_size > 0 and len(self.global_buffer) >= global_size:
                g_states, g_policies, g_values, _ = self.global_buffer.sample(global_size)
                if g_states is not None:
                    all_states.append(g_states)
                    all_policies.append(g_policies)
                    all_values.append(g_values)

            # 2. Sample from per-board HISTORICAL buffers
            for board_key, env_config in self.env_configs.items():
                buffer = self.replay_buffers[board_key]

                if len(buffer) < self.config.min_buffer_size:
                    continue

                if per_board_each > 0 and len(buffer) >= per_board_each:
                    r_states, r_policies, r_values = buffer.sample(per_board_each)
                    all_states.append(r_states)
                    all_policies.append(r_policies)
                    all_values.append(r_values)

            # 3. Generate CURRICULUM examples per board
            for board_key, env_config in self.env_configs.items():
                curr_per_board = curriculum_size // len(self.env_configs)
                if curr_per_board > 0:
                    self.rng, curr_rng = jax.random.split(self.rng)
                    c_states, c_policies, c_values = generate_curriculum_batch(
                        curr_rng, env_config, curr_per_board,
                        jump_distribution=list(self.config.curriculum_jump_distribution),
                    )
                    all_states.append(c_states)
                    all_policies.append(c_policies)
                    all_values.append(c_values)

            if not all_states:
                continue

            # Combine and train
            batch_states = np.concatenate(all_states, axis=0)
            batch_policies = np.concatenate(all_policies, axis=0)
            batch_values = np.concatenate(all_values, axis=0)

            # Shuffle
            perm = np.random.permutation(len(batch_states))
            batch_states = batch_states[perm]
            batch_policies = batch_policies[perm]
            batch_values = batch_values[perm]

            policy_loss, value_loss = self.train_step_distributed(
                jnp.array(batch_states),
                jnp.array(batch_policies),
                jnp.array(batch_values),
            )

            total_policy_loss += policy_loss
            total_value_loss += value_loss

        # Average losses
        num_steps = self.config.train_steps_per_iteration
        avg_policy_loss = total_policy_loss / num_steps if num_steps > 0 else 0.0
        avg_value_loss = total_value_loss / num_steps if num_steps > 0 else 0.0

        elapsed = time.time() - start_time

        # Log
        print(f"\nIteration {self.iteration + 1}/{self.config.num_iterations}")
        print(f"  Time: {elapsed:.1f}s | Policy loss: {avg_policy_loss:.4f} | Value loss: {avg_value_loss:.4f}")
        print(f"  Curriculum ratio: {curriculum_ratio:.1%}")
        for board_key, stats in iteration_stats.items():
            buffer_size = len(self.replay_buffers[board_key])
            print(f"  {board_key}: {stats['games']} games, {stats['examples']} examples, buffer={buffer_size}")

        self.iteration += 1

        # Save checkpoint
        if self.iteration % self.config.checkpoint_every == 0:
            self.save_checkpoint()

        return {
            'iteration': self.iteration,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'elapsed': elapsed,
            'stats': iteration_stats,
        }

    def train(self):
        """Run full training loop."""
        print("=" * 60)
        print("Distributed Chimera Training")
        print("=" * 60)
        print(f"Board sizes: {self.config.board_sizes}")
        print(f"Devices: {self.num_devices}")
        print(f"Device allocation: {self.device_allocation}")
        print(f"Iterations: {self.config.num_iterations}")
        print("=" * 60)

        for _ in range(self.config.num_iterations - self.iteration):
            metrics = self.train_iteration()
            self.metrics_history.append(metrics)

        print("\nTraining complete!")

    def save_checkpoint(self):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.iteration:06d}.pkl"
        )

        checkpoint = {
            'iteration': self.iteration,
            'params': self.params,
            'batch_stats': self.batch_stats,
            'opt_state': self.opt_state,
            'rng': self.rng,
            'total_games': self.total_games,
            'config': self.config,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.iteration = checkpoint['iteration']
        self.params = checkpoint['params']
        self.batch_stats = checkpoint['batch_stats']
        self.opt_state = checkpoint['opt_state']
        self.rng = checkpoint['rng']
        self.total_games = checkpoint['total_games']

        print(f"Loaded checkpoint from iteration {self.iteration}")


# =============================================================================
# Convenience functions
# =============================================================================

def make_distributed_config(
    board_sizes: Tuple[Tuple[int, int], ...] = ((11, 9), (15, 11), (21, 15)),
    strategy: str = "uniform",
    games_per_device: int = 64,
    train_batch_per_device: int = 128,
    num_channels: int = 128,
    num_res_blocks: int = 10,
    num_simulations: int = 50,
    learning_rate: float = 0.001,
    num_iterations: int = 500,
    **kwargs,
) -> DistributedChimeraConfig:
    """Create a distributed training configuration."""

    device_config = DeviceConfig(
        strategy=strategy,
        games_per_device=games_per_device,
        train_batch_per_device=train_batch_per_device,
    )

    return DistributedChimeraConfig(
        board_sizes=board_sizes,
        device_config=device_config,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_simulations=num_simulations,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    if jax.device_count() < 2:
        print("\nWarning: Less than 2 devices available. Distributed training works best with multiple devices.")

    # Create config
    config = make_distributed_config(
        board_sizes=((9, 7),),  # Small board for testing
        games_per_device=8,
        train_batch_per_device=16,
        num_channels=32,
        num_res_blocks=2,
        num_simulations=8,
        num_iterations=2,
    )

    print(f"\nConfig created: {config.board_sizes}")
    print(f"Device config: {config.device_config}")
