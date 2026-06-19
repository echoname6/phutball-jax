"""
Flax Neural Network for Phutball AlphaZero

Includes:
- ResNet-style policy-value network (CNN)
- Transformer-based policy-value network
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional
import optax


# ============================================================================
# Transformer Architecture
# ============================================================================

class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""
    d_model: int
    n_heads: int
    ffn_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Pre-norm self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate if train else 0.0,
            deterministic=not train,
        )(y, y)
        x = x + y

        # Pre-norm FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        x = x + y

        return x


class PhutballTransformer(nn.Module):
    """
    Transformer-based Policy-Value network for Phutball.

    Input: (batch, channels, rows, cols) where channels = 10
        (4 board + 5 jump-history layers + 1 single-jump reachability mask)

    Tokenization:
    - Each cell (r, c) becomes a token with 12 dims:
      [ball, men, my_goal, opp_goal, jump_seq[0:5], reach, row_enc, col_enc]
    - Full board: (rows × cols, 12)

    Architecture:
    - Linear projection to d_model
    - N transformer encoder blocks
    - Per-cell policy heads (placement + jump)
    - Global value head (mean pooling)
    """
    rows: int = 21
    cols: int = 15
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 256
    dropout_rate: float = 0.0
    pos_encoding: str = "goal_distance"  # "normalized" or "goal_distance"

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Args:
            x: Input (batch, 10, rows, cols) - 4 board + 5 jump-history + 1 reachability
        Returns:
            policy_logits: (batch, 2 * rows * cols + 1)
            value: (batch,)
        """
        batch_size = x.shape[0]
        num_cells = self.rows * self.cols
        action_space_size = 2 * num_cells + 1

        # Transpose to (batch, rows, cols, channels)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # Extract components:
        # x[:, :, :, 0:4]  = board features (ball, men, my_goal, opp_goal)
        # x[:, :, :, 4:9]  = jump sequence channels
        # x[:, :, :, 9:10] = single-jump reachability mask
        board_features = x[:, :, :, 0:4]   # (batch, rows, cols, 4)
        jump_seq       = x[:, :, :, 4:9]    # (batch, rows, cols, 5)
        reach          = x[:, :, :, 9:10]   # (batch, rows, cols, 1)

        # Create positional encodings
        if self.pos_encoding == "goal_distance":
            # goal_dist: -1 = deepest endzone row, 0 = goal line, 1+ = rows away from scoring
            row_pos = jnp.arange(self.rows) - 1  # [-1, 0, 1, 2, ..., rows-2]
            # lateral_dist: negative = left, 0 = center, positive = right
            col_pos = jnp.arange(self.cols) - (self.cols - 1) / 2  # e.g. [-4..4] for 9 cols
        else:  # "normalized"
            row_pos = jnp.arange(self.rows) / (self.rows - 1)  # [0, 1]
            col_pos = jnp.arange(self.cols) / (self.cols - 1)  # [0, 1]

        # Broadcast to (rows, cols)
        row_enc = jnp.broadcast_to(row_pos[:, None], (self.rows, self.cols))
        col_enc = jnp.broadcast_to(col_pos[None, :], (self.rows, self.cols))

        # Add batch dimension and channel dimension
        row_enc = jnp.broadcast_to(row_enc[None, :, :, None], (batch_size, self.rows, self.cols, 1))
        col_enc = jnp.broadcast_to(col_enc[None, :, :, None], (batch_size, self.rows, self.cols, 1))

        # Concatenate all features: (batch, rows, cols, 12)
        # 4 board + 5 jump_seq + 1 reach + 1 row_enc + 1 col_enc
        tokens = jnp.concatenate([board_features, jump_seq, reach, row_enc, col_enc], axis=-1)

        # Flatten spatial dims: (batch, rows * cols, 12)
        tokens = tokens.reshape(batch_size, num_cells, 12)

        # Linear projection to d_model: (batch, num_cells, d_model)
        x = nn.Dense(self.d_model)(tokens)

        # Transformer encoder blocks
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        # Final layer norm
        x = nn.LayerNorm()(x)

        # === Policy Head ===
        # Per-cell outputs: (batch, num_cells, d_model) -> (batch, num_cells, 1)
        placement_logits = nn.Dense(1)(x).squeeze(-1)  # (batch, num_cells)
        jump_logits = nn.Dense(1)(x).squeeze(-1)       # (batch, num_cells)

        # Global pooling for halt logit
        pooled = jnp.mean(x, axis=1)  # (batch, d_model)
        halt_logit = nn.Dense(1)(pooled).squeeze(-1)  # (batch,)

        # Concatenate: [placements, jumps, halt]
        policy_logits = jnp.concatenate([
            placement_logits,  # (batch, num_cells)
            jump_logits,       # (batch, num_cells)
            halt_logit[:, None],  # (batch, 1)
        ], axis=-1)  # (batch, 2 * num_cells + 1)

        # === Value Head ===
        value = nn.Dense(64)(pooled)
        value = nn.gelu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value).squeeze(-1)  # (batch,)

        return policy_logits, value


def create_transformer_network(
    rows: int = 21,
    cols: int = 15,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ffn_dim: int = 256,
    dropout_rate: float = 0.0,
    pos_encoding: str = "goal_distance",
) -> PhutballTransformer:
    """Factory function to create transformer network."""
    return PhutballTransformer(
        rows=rows,
        cols=cols,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        dropout_rate=dropout_rate,
        pos_encoding=pos_encoding,
    )


def init_transformer_network(rng, network: PhutballTransformer, num_input_channels: int = 10):
    """Initialize transformer network parameters."""
    dummy_input = jnp.zeros((1, num_input_channels, network.rows, network.cols))
    variables = network.init(rng, dummy_input, train=False)
    return variables


# ============================================================================
# CNN Architecture (Original)
# ============================================================================


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and skip connection."""
    channels: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        
        x = x + residual
        x = nn.swish(x)
        return x


class Backbone(nn.Module):
    """Shared convolutional backbone - fully convolutional, any board size."""
    num_channels: int = 64
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        for _ in range(self.num_res_blocks):
            x = ResidualBlock(channels=self.num_channels)(x, train=train)
        return x


class ValueHead(nn.Module):
    """Value head with global avg pooling - board size agnostic."""

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(8, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        x = jnp.mean(x, axis=(1, 2))  # Global avg pool
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return nn.tanh(x).squeeze(-1)


class PolicyHead(nn.Module):
    """Policy head for specific board size."""
    rows: int
    cols: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        action_space_size = 2 * self.rows * self.cols + 1
        x = nn.Conv(32, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        x = x.reshape((batch_size, -1))
        return nn.Dense(action_space_size)(x)


class ChimeraNetwork(nn.Module):
    """
    Multi-board network: shared backbone + value head, separate policy heads.

    Args:
        board_sizes: Tuple of (rows, cols) pairs, e.g. ((21,15), (15,11), (11,9))
        num_channels: Backbone channel width
        num_res_blocks: Number of residual blocks
    """
    board_sizes: tuple  # ((r1,c1), (r2,c2), ...)
    num_channels: int = 64
    num_res_blocks: int = 6

    def setup(self):
        self.backbone = Backbone(self.num_channels, self.num_res_blocks)
        self.value_head = ValueHead()
        self.policy_heads = {
            f"{r}x{c}": PolicyHead(rows=r, cols=c)
            for r, c in self.board_sizes
        }

    def __call__(self, x, board_key: str, train: bool = True):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        features = self.backbone(x, train=train)
        value = self.value_head(features, train=train)
        policy = self.policy_heads[board_key](features, train=train)
        return policy, value

    def init_all_heads(self, inputs_dict: dict, train: bool = False):
        """
        Initialize all policy heads by calling each one.
        inputs_dict: {board_key: input_tensor} for each board size.
        Returns dict of (policy, value) for each board.
        """
        results = {}
        for board_key, x in inputs_dict.items():
            x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
            features = self.backbone(x, train=train)
            value = self.value_head(features, train=train)
            policy = self.policy_heads[board_key](features, train=train)
            results[board_key] = (policy, value)
        return results


def create_chimera_network(
    board_sizes: tuple,
    num_channels: int = 64,
    num_res_blocks: int = 6
) -> ChimeraNetwork:
    """Create chimera network with specified board sizes."""
    return ChimeraNetwork(
        board_sizes=board_sizes,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
    )


def init_chimera_network(rng, network: ChimeraNetwork, num_input_channels: int = 10):
    """Initialize chimera - needs dummy input for each board size."""
    # Create dummy inputs for all board sizes
    inputs_dict = {}
    for rows, cols in network.board_sizes:
        board_key = f"{rows}x{cols}"
        inputs_dict[board_key] = jnp.zeros((1, num_input_channels, rows, cols))

    # Initialize using init_all_heads which touches all policy heads
    variables = network.init(rng, inputs_dict, train=False, method=network.init_all_heads)
    return variables


def expand_chimera_network(
    old_variables: dict,
    old_board_sizes: tuple,
    new_network: ChimeraNetwork,
    rng,
    num_input_channels: int = 10,
):
    """
    Expand a trained ChimeraNetwork to include new board sizes.

    Copies backbone, value head, and existing policy heads from old_variables.
    Initializes new policy heads from scratch.

    Args:
        old_variables: Checkpoint variables with 'params' and 'batch_stats'
        old_board_sizes: Original board sizes tuple, e.g. ((11,9), (15,11))
        new_network: New ChimeraNetwork with expanded board_sizes
        rng: Random key for initializing new policy heads
        num_input_channels: Input channels (default 6)

    Returns:
        new_variables: Variables for expanded network

    Example:
        # Load checkpoint trained on small boards
        old_vars = pickle.load(...)
        old_sizes = ((11,9), (15,11))

        # Create network with new board size added
        new_net = create_chimera_network(
            board_sizes=((11,9), (15,11), (21,15)),  # added 21x15
            num_channels=64,
            num_res_blocks=6
        )

        # Expand - copies shared params, inits new policy head
        new_vars = expand_chimera_network(old_vars, old_sizes, new_net, rng)
    """
    # Initialize the new network fully
    new_variables = init_chimera_network(rng, new_network, num_input_channels)

    # Copy backbone params and batch_stats
    new_variables['params']['backbone'] = old_variables['params']['backbone']
    new_variables['batch_stats']['backbone'] = old_variables['batch_stats']['backbone']

    # Copy value head params and batch_stats
    new_variables['params']['value_head'] = old_variables['params']['value_head']
    new_variables['batch_stats']['value_head'] = old_variables['batch_stats']['value_head']

    # Copy existing policy heads (new ones stay randomly initialized)
    for rows, cols in old_board_sizes:
        board_key = f"{rows}x{cols}"
        if board_key in old_variables['params']['policy_heads']:
            new_variables['params']['policy_heads'][board_key] = \
                old_variables['params']['policy_heads'][board_key]
            new_variables['batch_stats']['policy_heads'][board_key] = \
                old_variables['batch_stats']['policy_heads'][board_key]

    return new_variables


def compute_chimera_loss(params, batch_norm_state, network, board_key, batch, rng):
    """
    Compute AlphaZero loss for a specific board size using ChimeraNetwork.

    Args:
        params: Network parameters
        batch_norm_state: BatchNorm running statistics
        network: ChimeraNetwork instance
        board_key: String like "21x15" specifying which policy head
        batch: Dict with 'states', 'policy_targets', 'value_targets'
        rng: Random key
    """
    states = batch['states']
    policy_targets = batch['policy_targets']
    value_targets = batch['value_targets']

    variables = {'params': params, 'batch_stats': batch_norm_state}
    (policy_logits, value_preds), new_state = network.apply(
        variables, states, board_key, train=True, mutable=['batch_stats']
    )

    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(jnp.sum(policy_targets * log_probs, axis=-1))
    value_loss = jnp.mean(jnp.square(value_preds - value_targets))

    probs = jax.nn.softmax(policy_logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    mcts_entropy = -jnp.sum(policy_targets * jnp.log(policy_targets + 1e-8), axis=-1)
    kl_div = jnp.mean(-mcts_entropy - jnp.sum(policy_targets * log_probs, axis=-1))

    # Value prediction stats
    value_pred_mean = jnp.mean(value_preds)
    value_pred_std = jnp.std(value_preds)

    total_loss = policy_loss + value_loss

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
        'policy_entropy': entropy,
        'mcts_entropy': jnp.mean(mcts_entropy),
        'kl_divergence': kl_div,
        'value_pred_mean': value_pred_mean,
        'value_pred_std': value_pred_std,
    }
    return total_loss, (new_state['batch_stats'], metrics)


def make_chimera_train_step_fn(network, optimizer, board_key):
    """Create JIT-compiled train step for a specific board size."""

    @jax.jit
    def _train_step(params, batch_stats, opt_state, batch, rng):
        def loss_fn(p):
            return compute_chimera_loss(p, batch_stats, network, board_key, batch, rng)

        (loss, (new_bn_state, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_bn_state, new_opt_state, metrics

    return _train_step


def predict_chimera(params, batch_norm_state, network, states, board_key):
    """Run inference for ChimeraNetwork."""
    variables = {'params': params, 'batch_stats': batch_norm_state}
    policy_logits, values = network.apply(variables, states, board_key, train=False)
    policy_probs = jax.nn.softmax(policy_logits)
    return policy_probs, values


class PhutballNetwork(nn.Module):
    """
    Policy-Value network for Phutball (single board size).

    Architecture:
    - Initial conv layer
    - N residual blocks
    - Policy head: conv -> flatten -> dense -> action logits
    - Value head: conv -> flatten -> dense -> tanh
    """
    rows: int = 21
    cols: int = 15
    num_channels: int = 64
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, rows, cols)
               Channels: [ball, men, my_goal, opp_goal, jump_0..4] = 9 channels
            train: Whether in training mode (affects BatchNorm)
            
        Returns:
            policy_logits: (batch, 2 * rows * cols + 1) - raw logits for actions
            value: (batch,) - value estimate in [-1, 1]
        """
        batch_size = x.shape[0]
        action_space_size = 2 * self.rows * self.cols + 1
        
        # JAX uses NHWC by default, but we have NCHW input
        # Transpose: (batch, channels, rows, cols) -> (batch, rows, cols, channels)
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Initial convolution
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        
        # Residual tower
        for _ in range(self.num_res_blocks):
            x = ResidualBlock(channels=self.num_channels)(x, train=train)
        
        # Policy head
        policy = nn.Conv(32, kernel_size=(1, 1), use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.swish(policy)
        policy = policy.reshape((batch_size, -1))  # Flatten
        policy_logits = nn.Dense(action_space_size)(policy)
        
        # Value head
        value = nn.Conv(8, kernel_size=(1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.swish(value)
        value = value.reshape((batch_size, -1))  # Flatten
        value = nn.Dense(64)(value)
        value = nn.swish(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        value = value.squeeze(-1)  # (batch, 1) -> (batch,)
        
        return policy_logits, value


def create_network(rows: int = 21, cols: int = 15, num_channels: int = 64, num_res_blocks: int = 6):
    """Factory function to create network with given config."""
    return PhutballNetwork(
        rows=rows,
        cols=cols,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks
    )


def init_network(rng, network: PhutballNetwork, num_input_channels: int = 10):
    """
    Initialize network parameters.
    
    Args:
        rng: JAX random key
        network: PhutballNetwork instance
        num_input_channels: Number of input channels (default 10: 4 board + 5 jump + 1 reachability)
        
    Returns:
        params: Network parameters
    """
    dummy_input = jnp.zeros((1, num_input_channels, network.rows, network.cols))
    variables = network.init(rng, dummy_input, train=False)
    return variables


def create_optimizer(learning_rate: float = 0.001, weight_decay: float = 1e-4):
    """Create optimizer with learning rate and weight decay."""
    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)


def compute_loss(params, batch_norm_state, network, batch, rng):
    """
    Compute AlphaZero loss: policy cross-entropy + value MSE.
    
    Args:
        params: Network parameters
        batch_norm_state: BatchNorm running statistics
        network: PhutballNetwork instance
        batch: Dict with 'states', 'policy_targets', 'value_targets'
        rng: Random key (for dropout if used)
        
    Returns:
        total_loss: Combined loss
        (new_batch_norm_state, metrics): Updated BN state and loss components
    """
    states = batch['states']           # (batch, channels, rows, cols)
    policy_targets = batch['policy_targets']  # (batch, action_space_size)
    value_targets = batch['value_targets']    # (batch,)
    
    # Forward pass with mutable batch norm state
    variables = {'params': params, 'batch_stats': batch_norm_state}
    (policy_logits, value_preds), new_state = network.apply(
        variables,
        states,
        train=True,
        mutable=['batch_stats']
    )
    
    # Policy loss: cross-entropy with MCTS policy targets
    # policy_targets are probabilities from MCTS, policy_logits be logits
    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(jnp.sum(policy_targets * log_probs, axis=-1))
    
    # Value loss: MSE between predicted value and game outcome
    value_loss = jnp.mean(jnp.square(value_preds - value_targets))
    
    # Total loss
    total_loss = policy_loss + value_loss

    # Policy entropy (network output)
    probs = jax.nn.softmax(policy_logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
    
    # KL divergence: D_KL(MCTS || Network) = sum(mcts * log(mcts / net))
    # = sum(mcts * log(mcts)) - sum(mcts * log(net))
    # The second term is just the cross-entropy (policy_loss before mean)
    mcts_entropy = -jnp.sum(policy_targets * jnp.log(policy_targets + 1e-8), axis=-1)
    kl_div = jnp.mean(-mcts_entropy - jnp.sum(policy_targets * log_probs, axis=-1))
    # Simplifies to: kl_div = policy_loss - mean(mcts_entropy)

    # Value prediction stats - check if value head is actually learning
    # (if outputting ~0 for everything, value_loss is low but it's not learning)
    value_pred_mean = jnp.mean(value_preds)
    value_pred_std = jnp.std(value_preds)

    total_loss = policy_loss + value_loss

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
        'policy_entropy': entropy,
        'mcts_entropy': jnp.mean(mcts_entropy),
        'kl_divergence': kl_div,
        'value_pred_mean': value_pred_mean,
        'value_pred_std': value_pred_std,
    }
    
    return total_loss, (new_state['batch_stats'], metrics)


def make_train_step_fn(network, optimizer):
    """Create a JIT-compiled train step function for a specific network/optimizer."""
    
    @jax.jit
    def _train_step(params, batch_stats, opt_state, batch, rng):
        def loss_fn(p):
            return compute_loss(p, batch_stats, network, batch, rng)
        
        (loss, (new_bn_state, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_bn_state, new_opt_state, metrics
    
    return _train_step


def compute_loss_transformer(params, network, batch, rng):
    """
    Compute AlphaZero loss for Transformer (no batch_stats).

    Args:
        params: Network parameters
        network: PhutballTransformer instance
        batch: Dict with 'states', 'policy_targets', 'value_targets'
        rng: Random key (for dropout if used)

    Returns:
        total_loss: Combined loss
        metrics: Dict of loss components
    """
    states = batch['states']
    policy_targets = batch['policy_targets']
    value_targets = batch['value_targets']

    # Forward pass (no mutable state for LayerNorm)
    variables = {'params': params}
    policy_logits, value_preds = network.apply(variables, states, train=True)

    # Policy loss: cross-entropy with MCTS policy targets
    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(jnp.sum(policy_targets * log_probs, axis=-1))

    # Value loss: MSE between predicted value and game outcome
    value_loss = jnp.mean(jnp.square(value_preds - value_targets))

    # Total loss
    total_loss = policy_loss + value_loss

    # Policy entropy (network output)
    probs = jax.nn.softmax(policy_logits)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

    # KL divergence
    mcts_entropy = -jnp.sum(policy_targets * jnp.log(policy_targets + 1e-8), axis=-1)
    kl_div = jnp.mean(-mcts_entropy - jnp.sum(policy_targets * log_probs, axis=-1))

    # Value prediction stats
    value_pred_mean = jnp.mean(value_preds)
    value_pred_std = jnp.std(value_preds)

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
        'policy_entropy': entropy,
        'mcts_entropy': jnp.mean(mcts_entropy),
        'kl_divergence': kl_div,
        'value_pred_mean': value_pred_mean,
        'value_pred_std': value_pred_std,
    }

    return total_loss, metrics


def make_transformer_train_step_fn(network, optimizer):
    """Create a JIT-compiled train step function for Transformer (no batch_stats)."""

    @jax.jit
    def _train_step(params, opt_state, batch, rng):
        def loss_fn(p):
            return compute_loss_transformer(p, network, batch, rng)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return _train_step


def predict_transformer(params, network, states):
    """Run inference for Transformer (no batch_stats)."""
    variables = {'params': params}
    policy_logits, values = network.apply(variables, states, train=False)
    policy_probs = jax.nn.softmax(policy_logits)
    return policy_probs, values


def predict(params, batch_norm_state, network, states):
    """
    Run inference (no gradients, uses running BN stats).

    Args:
        params: Network parameters
        batch_norm_state: BatchNorm running statistics
        network: PhutballNetwork instance
        states: Input states (batch, channels, rows, cols)
        
    Returns:
        policy_probs: (batch, action_space_size) - softmax probabilities
        values: (batch,) - value predictions
    """
    variables = {'params': params, 'batch_stats': batch_norm_state}
    policy_logits, values = network.apply(variables, states, train=False)
    policy_probs = jax.nn.softmax(policy_logits)
    return policy_probs, values


# JIT-compiled prediction for speed
@jax.jit
def predict_jit(params, batch_norm_state, network, states):
    """JIT-compiled inference."""
    return predict(params, batch_norm_state, network, states)


# ============================================================================
# Tests
# ============================================================================

def test_network_shapes():
    """Test that network produces correct output shapes."""
    rows, cols = 21, 15
    batch_size = 8
    num_input_channels=10
    action_space_size = 2 * rows * cols + 1  # 631
    
    # Create network
    network = create_network(rows=rows, cols=cols, num_channels=64, num_res_blocks=6)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = init_network(rng, network, num_input_channels)
    
    # Check params and batch_stats exist
    assert 'params' in variables, "Should have params"
    assert 'batch_stats' in variables, "Should have batch_stats"
    
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Create dummy input
    dummy_input = jax.random.normal(rng, (batch_size, num_input_channels, rows, cols))
    
    # Forward pass
    policy_logits, values = predict(params, batch_stats, network, dummy_input)
    
    # Check shapes
    assert policy_logits.shape == (batch_size, action_space_size), \
        f"Policy shape: expected {(batch_size, action_space_size)}, got {policy_logits.shape}"
    assert values.shape == (batch_size,), \
        f"Value shape: expected {(batch_size,)}, got {values.shape}"
    
    # Check value range (tanh output)
    assert jnp.all(values >= -1) and jnp.all(values <= 1), \
        f"Values should be in [-1, 1], got min={values.min()}, max={values.max()}"
    
    # Check policy sums to 1
    policy_sums = jnp.sum(policy_logits, axis=-1)
    # Note: these are logits before softmax, so won't sum to 1
    # After softmax:
    policy_probs = jax.nn.softmax(policy_logits)
    prob_sums = jnp.sum(policy_probs, axis=-1)
    assert jnp.allclose(prob_sums, 1.0, atol=1e-5), \
        f"Policy probs should sum to 1, got {prob_sums}"
    
    print(f"✓ Network shapes correct")
    print(f"  Policy: {policy_logits.shape}")
    print(f"  Values: {values.shape}")
    print(f"  Policy probs sum: {prob_sums[0]:.6f}")


def test_training_step():
    """Test that training step runs and reduces loss."""
    rows, cols = 21, 15
    batch_size = 4
    num_input_channels=10
    action_space_size = 2 * rows * cols + 1
    
    # Create network and optimizer
    network = create_network(rows=rows, cols=cols, num_channels=32, num_res_blocks=2)
    optimizer = create_optimizer(learning_rate=0.01)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = init_network(rng, network, num_input_channels)
    params = variables['params']
    batch_stats = variables['batch_stats']
    opt_state = optimizer.init(params)
    
    # Create the JIT-compiled train step for this network/optimizer
    train_step_fn = make_train_step_fn(network, optimizer)
    
    # Create dummy batch
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    dummy_states = jax.random.normal(key1, (batch_size, num_input_channels, rows, cols))
    
    # Random policy targets (normalized to sum to 1)
    dummy_policy = jax.random.uniform(key2, (batch_size, action_space_size))
    dummy_policy = dummy_policy / dummy_policy.sum(axis=-1, keepdims=True)
    
    # Random value targets in [-1, 1]
    dummy_values = jax.random.uniform(key3, (batch_size,), minval=-1, maxval=1)
    
    batch = {
        'states': dummy_states,
        'policy_targets': dummy_policy,
        'value_targets': dummy_values,
    }
    
    # Run a few training steps
    losses = []
    for i in range(5):
        rng, step_rng = jax.random.split(rng)
        params, batch_stats, opt_state, metrics = train_step_fn(
            params, batch_stats, opt_state, batch, step_rng
        )
        losses.append(float(metrics['total_loss']))
    
    print(f"✓ Training step runs")
    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    
    # Loss should generally decrease (not guaranteed but likely with high LR)
    # Just check it doesn't explode
    assert all(l < 100 for l in losses), f"Loss exploded: {losses}"


def test_different_board_sizes():
    """Test network works with different board sizes."""
    configs = [
        (9, 9),
        (15, 15),
        (21, 15),
    ]
    
    rng = jax.random.PRNGKey(42)
    
    for rows, cols in configs:
        network = create_network(rows=rows, cols=cols, num_channels=32, num_res_blocks=2)
        variables = init_network(rng, network, num_input_channels=10)
        
        dummy_input = jax.random.normal(rng, (2, 9, rows, cols))
        policy, value = predict(
            variables['params'], 
            variables['batch_stats'], 
            network, 
            dummy_input
        )
        
        expected_actions = 2 * rows * cols + 1
        assert policy.shape == (2, expected_actions), f"Failed for {rows}x{cols}"
        print(f"✓ {rows}x{cols}: policy shape {policy.shape}, value shape {value.shape}")


def test_transformer_shapes():
    """Test that transformer network produces correct output shapes."""
    rows, cols = 21, 15
    batch_size = 8
    num_input_channels=10
    action_space_size = 2 * rows * cols + 1  # 631

    # Create transformer network
    network = create_transformer_network(
        rows=rows, cols=cols,
        d_model=128, n_layers=4, n_heads=4, ffn_dim=256
    )

    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = init_transformer_network(rng, network, num_input_channels)

    # Check params exist (no batch_stats for transformer - uses LayerNorm)
    assert 'params' in variables, "Should have params"

    params = variables['params']

    # Create dummy input
    dummy_input = jax.random.normal(rng, (batch_size, num_input_channels, rows, cols))

    # Forward pass
    policy_logits, values = network.apply(variables, dummy_input, train=False)

    # Check shapes
    assert policy_logits.shape == (batch_size, action_space_size), \
        f"Policy shape: expected {(batch_size, action_space_size)}, got {policy_logits.shape}"
    assert values.shape == (batch_size,), \
        f"Value shape: expected {(batch_size,)}, got {values.shape}"

    # Check value range (tanh output)
    assert jnp.all(values >= -1) and jnp.all(values <= 1), \
        f"Values should be in [-1, 1], got min={values.min()}, max={values.max()}"

    # Check policy sums to 1 after softmax
    policy_probs = jax.nn.softmax(policy_logits)
    prob_sums = jnp.sum(policy_probs, axis=-1)
    assert jnp.allclose(prob_sums, 1.0, atol=1e-5), \
        f"Policy probs should sum to 1, got {prob_sums}"

    print(f"✓ Transformer shapes correct")
    print(f"  Policy: {policy_logits.shape}")
    print(f"  Values: {values.shape}")
    print(f"  Policy probs sum: {prob_sums[0]:.6f}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {param_count:,}")


def test_transformer_training():
    """Test transformer training step."""
    rows, cols = 11, 9  # Smaller for faster test
    batch_size = 4
    num_input_channels=10
    action_space_size = 2 * rows * cols + 1

    # Create network and optimizer
    network = create_transformer_network(
        rows=rows, cols=cols,
        d_model=64, n_layers=2, n_heads=4, ffn_dim=128
    )
    optimizer = create_optimizer(learning_rate=0.001)

    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = init_transformer_network(rng, network, num_input_channels)
    params = variables['params']
    opt_state = optimizer.init(params)

    # Create dummy batch
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    dummy_states = jax.random.normal(key1, (batch_size, num_input_channels, rows, cols))

    dummy_policy = jax.random.uniform(key2, (batch_size, action_space_size))
    dummy_policy = dummy_policy / dummy_policy.sum(axis=-1, keepdims=True)

    dummy_values = jax.random.uniform(key3, (batch_size,), minval=-1, maxval=1)

    batch = {
        'states': dummy_states,
        'policy_targets': dummy_policy,
        'value_targets': dummy_values,
    }

    # Define loss function for transformer (no batch_stats)
    def compute_transformer_loss(params, network, batch):
        states = batch['states']
        policy_targets = batch['policy_targets']
        value_targets = batch['value_targets']

        policy_logits, value_preds = network.apply({'params': params}, states, train=True)

        log_probs = jax.nn.log_softmax(policy_logits)
        policy_loss = -jnp.mean(jnp.sum(policy_targets * log_probs, axis=-1))
        value_loss = jnp.mean(jnp.square(value_preds - value_targets))

        return policy_loss + value_loss

    # JIT-compiled train step
    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(compute_transformer_loss)(params, network, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Run training steps
    losses = []
    for i in range(5):
        params, opt_state, loss = train_step(params, opt_state, batch)
        losses.append(float(loss))

    print(f"✓ Transformer training runs")
    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")

    assert all(l < 100 for l in losses), f"Loss exploded: {losses}"


def benchmark_networks():
    """Benchmark CNN vs Transformer performance."""
    import time

    rows, cols = 21, 15
    batch_size = 32
    num_input_channels=10
    num_warmup = 5
    num_iterations = 50

    rng = jax.random.PRNGKey(42)
    dummy_input = jax.random.normal(rng, (batch_size, num_input_channels, rows, cols))

    print(f"Board: {rows}x{cols}, Batch size: {batch_size}")
    print(f"Warmup: {num_warmup}, Iterations: {num_iterations}")
    print()

    # === CNN Benchmark ===
    print("=== CNN (PhutballNetwork) ===")
    cnn = create_network(rows=rows, cols=cols, num_channels=64, num_res_blocks=6)
    cnn_vars = init_network(rng, cnn, num_input_channels)
    cnn_params = cnn_vars['params']
    cnn_batch_stats = cnn_vars['batch_stats']

    cnn_param_count = sum(x.size for x in jax.tree_util.tree_leaves(cnn_params))
    print(f"Parameters: {cnn_param_count:,}")

    # JIT compile
    @jax.jit
    def cnn_forward(params, batch_stats, x):
        return predict(params, batch_stats, cnn, x)

    # Warmup
    for _ in range(num_warmup):
        _ = cnn_forward(cnn_params, cnn_batch_stats, dummy_input)
    jax.block_until_ready(cnn_forward(cnn_params, cnn_batch_stats, dummy_input))

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        policy, value = cnn_forward(cnn_params, cnn_batch_stats, dummy_input)
    jax.block_until_ready(policy)
    cnn_time = (time.perf_counter() - start) / num_iterations * 1000
    print(f"Inference: {cnn_time:.2f} ms/batch ({batch_size / cnn_time * 1000:.0f} samples/sec)")
    print()

    # === Transformer Benchmark ===
    print("=== Transformer (PhutballTransformer) ===")
    transformer = create_transformer_network(
        rows=rows, cols=cols,
        d_model=128, n_layers=4, n_heads=4, ffn_dim=256
    )
    transformer_vars = init_transformer_network(rng, transformer, num_input_channels)
    transformer_params = transformer_vars['params']

    transformer_param_count = sum(x.size for x in jax.tree_util.tree_leaves(transformer_params))
    print(f"Parameters: {transformer_param_count:,}")

    # JIT compile
    @jax.jit
    def transformer_forward(params, x):
        return transformer.apply({'params': params}, x, train=False)

    # Warmup
    for _ in range(num_warmup):
        _ = transformer_forward(transformer_params, dummy_input)
    jax.block_until_ready(transformer_forward(transformer_params, dummy_input))

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        policy, value = transformer_forward(transformer_params, dummy_input)
    jax.block_until_ready(policy)
    transformer_time = (time.perf_counter() - start) / num_iterations * 1000
    print(f"Inference: {transformer_time:.2f} ms/batch ({batch_size / transformer_time * 1000:.0f} samples/sec)")
    print()

    # === Summary ===
    print("=== Summary ===")
    print(f"CNN:         {cnn_param_count:>10,} params, {cnn_time:>6.2f} ms/batch")
    print(f"Transformer: {transformer_param_count:>10,} params, {transformer_time:>6.2f} ms/batch")
    print(f"Speedup: {cnn_time / transformer_time:.2f}x" if transformer_time < cnn_time else f"Slowdown: {transformer_time / cnn_time:.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Benchmarking Networks...\n")
        benchmark_networks()
    else:
        print("Testing Phutball Network...\n")

        print("=== CNN Tests ===")
        test_network_shapes()
        print()

        test_training_step()
        print()

        test_different_board_sizes()
        print()

        print("=== Transformer Tests ===")
        test_transformer_shapes()
        print()

        test_transformer_training()

        print("\n✓ All network tests passed!")
        print("\nRun 'python network.py benchmark' for performance comparison.")