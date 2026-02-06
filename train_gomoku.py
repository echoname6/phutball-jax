"""
AlphaZero Training for Gomoku.

Demonstrates the AlphaZero pipeline (policy-value network + MCTS) on
Gomoku, a much simpler game than Phutball.  Intended as a sanity check
that the framework learns.

Usage:
    python train_gomoku.py                           # 9x9, CNN, fast defaults
    python train_gomoku.py --network transformer     # use transformer
    python train_gomoku.py --board-size 15           # standard 15x15
    python train_gomoku.py --wandb                   # log to W&B
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import time
import os
import argparse
import pickle

from gomoku_env_jax import (
    GomokuConfig, GomokuState,
    reset, step, get_legal_actions, state_to_network_input,
    EMPTY, BLACK, WHITE, render_board,
)
from network import TransformerBlock
from alphazero_core import (
    make_recurrent_fn, run_mcts, play_games, make_training_examples,
    ReplayBuffer, make_train_step_cnn, make_train_step_transformer,
    evaluate_vs_random,
)


# ============================================================================
# Neural Networks
# ============================================================================

class GomokuCNN(nn.Module):
    """ResNet-style policy-value CNN for Gomoku."""
    board_size: int = 9
    num_channels: int = 64
    num_res_blocks: int = 6

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        action_space = self.board_size ** 2

        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC

        # Initial convolution
        x = nn.Conv(self.num_channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)

        # Residual tower
        for _ in range(self.num_res_blocks):
            residual = x
            x = nn.Conv(self.num_channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.swish(x)
            x = nn.Conv(self.num_channels, (3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = x + residual
            x = nn.swish(x)

        # Policy head
        p = nn.Conv(32, (1, 1), use_bias=False)(x)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.swish(p)
        p = p.reshape(batch_size, -1)
        policy_logits = nn.Dense(action_space)(p)

        # Value head
        v = nn.Conv(8, (1, 1), use_bias=False)(x)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.swish(v)
        v = jnp.mean(v, axis=(1, 2))  # global avg pool
        v = nn.Dense(64)(v)
        v = nn.swish(v)
        v = nn.Dense(1)(v)
        value = nn.tanh(v).squeeze(-1)

        return policy_logits, value


class GomokuTransformer(nn.Module):
    """Transformer-based policy-value network for Gomoku."""
    board_size: int = 9
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 256

    @nn.compact
    def __call__(self, x, train: bool = True):
        batch_size = x.shape[0]
        bs = self.board_size
        num_cells = bs * bs

        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, H, W, C)

        # Positional encoding (normalised row/col)
        row_pos = jnp.linspace(0, 1, bs)
        col_pos = jnp.linspace(0, 1, bs)
        row_enc = jnp.broadcast_to(row_pos[:, None], (bs, bs))
        col_enc = jnp.broadcast_to(col_pos[None, :], (bs, bs))
        row_enc = jnp.broadcast_to(
            row_enc[None, :, :, None], (batch_size, bs, bs, 1))
        col_enc = jnp.broadcast_to(
            col_enc[None, :, :, None], (batch_size, bs, bs, 1))

        tokens = jnp.concatenate([x, row_enc, col_enc], axis=-1)
        tokens = tokens.reshape(batch_size, num_cells, -1)

        x = nn.Dense(self.d_model)(tokens)

        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ffn_dim=self.ffn_dim,
            )(x, train=train)

        x = nn.LayerNorm()(x)

        # Policy head: per-cell logit
        policy_logits = nn.Dense(1)(x).squeeze(-1)  # (B, num_cells)

        # Value head: global pool
        pooled = jnp.mean(x, axis=1)
        v = nn.Dense(64)(pooled)
        v = nn.gelu(v)
        v = nn.Dense(1)(v)
        value = nn.tanh(v).squeeze(-1)

        return policy_logits, value


# ============================================================================
# Data Augmentation (Gomoku-specific: 8-fold symmetry)
# ============================================================================

def augment_data(states, policies, values, board_size):
    """
    8-fold symmetry augmentation (4 rotations x 2 reflections).
    Multiplies the data by 8.
    """
    all_s, all_p, all_v = [states], [policies], [values]

    pol_2d = policies.reshape(-1, board_size, board_size)

    for k in [1, 2, 3]:
        rs = np.rot90(states, k=k, axes=(2, 3))
        rp = np.rot90(pol_2d, k=k, axes=(1, 2)).reshape(-1, board_size ** 2)
        all_s.append(rs)
        all_p.append(rp)
        all_v.append(values)

    # Horizontal flip
    fs = np.flip(states, axis=3).copy()
    fp = np.flip(pol_2d, axis=2).reshape(-1, board_size ** 2).copy()
    all_s.append(fs)
    all_p.append(fp)
    all_v.append(values)

    fp_2d = fp.reshape(-1, board_size, board_size)
    for k in [1, 2, 3]:
        rs = np.rot90(fs, k=k, axes=(2, 3))
        rp = np.rot90(fp_2d, k=k, axes=(1, 2)).reshape(-1, board_size ** 2)
        all_s.append(rs)
        all_p.append(rp)
        all_v.append(values)

    return (np.concatenate(all_s),
            np.concatenate(all_p),
            np.concatenate(all_v))


# ============================================================================
# Sample Game Display (Gomoku-specific)
# ============================================================================

def show_sample_game(params, network, env_config, use_bn):
    """Play one game and print the board at each step."""
    rng = jax.random.PRNGKey(999)
    state = reset(env_config)
    action_space = env_config.board_size ** 2

    print("\n--- Sample game (network vs random) ---")
    network_player = BLACK

    for move in range(action_space):
        if state.terminated:
            break
        rng, move_rng = jax.random.split(rng)

        if int(state.current_player) == network_player:
            batch_state = jax.tree.map(lambda x: x[None], state)
            net_in = state_to_network_input(state, env_config)[None]
            if use_bn:
                variables = {
                    'params': params['network_params'],
                    'batch_stats': params['batch_stats'],
                }
            else:
                variables = {'params': params['network_params']}
            logits, val = network.apply(variables, net_in, train=False)
            legal = get_legal_actions(state, env_config)
            masked = jnp.where(legal == 1, logits[0], -1e9)
            action = jnp.argmax(masked)
            r, c = int(action) // env_config.board_size, int(action) % env_config.board_size
            print(f"  Network plays ({r},{c})  value={float(val[0]):.3f}")
        else:
            legal = get_legal_actions(state, env_config)
            probs = legal.astype(jnp.float32)
            probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
            action = jax.random.choice(move_rng, action_space, p=probs)
            r, c = int(action) // env_config.board_size, int(action) % env_config.board_size
            print(f"  Random  plays ({r},{c})")

        state = step(state, action, env_config)

    print(render_board(state))
    print()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero training for Gomoku')
    # Environment
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--win-length', type=int, default=5)
    # Network
    parser.add_argument('--network', choices=['cnn', 'transformer'],
                        default='cnn')
    parser.add_argument('--num-channels', type=int, default=64,
                        help='CNN channel width')
    parser.add_argument('--num-res-blocks', type=int, default=6)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--ffn-dim', type=int, default=256)
    # Training
    parser.add_argument('--num-iterations', type=int, default=50)
    parser.add_argument('--games-per-iter', type=int, default=32)
    parser.add_argument('--num-simulations', type=int, default=25)
    parser.add_argument('--train-batch-size', type=int, default=256)
    parser.add_argument('--train-steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Exploration temperature for early moves')
    parser.add_argument('--temp-threshold', type=int, default=15,
                        help='After this many moves, drop to temp=0.1')
    # Evaluation
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--eval-games', type=int, default=50)
    parser.add_argument('--eval-sims', type=int, default=0,
                        help='MCTS sims during eval (0 = raw policy)')
    # Data
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment',
                        action='store_false')
    parser.add_argument('--buffer-size', type=int, default=200_000)
    # Misc
    parser.add_argument('--checkpoint-dir', default='checkpoints_gomoku')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show-game-every', type=int, default=0,
                        help='Show a sample game every N iters (0=off)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    env_config = GomokuConfig(
        board_size=args.board_size,
        win_length=args.win_length,
        max_turns=args.board_size ** 2,
    )
    use_bn = args.network == 'cnn'
    rng = jax.random.PRNGKey(args.seed)

    # --- Bind env functions ---
    env_reset_fn = lambda: reset(env_config)
    env_step_fn = lambda s, a: step(s, a, env_config)
    env_legal_fn = lambda s: get_legal_actions(s, env_config)
    env_obs_fn = lambda s: state_to_network_input(s, env_config)

    action_space = args.board_size ** 2
    num_channels = 3
    obs_shape = (num_channels, args.board_size, args.board_size)
    max_moves = args.board_size ** 2

    # --- Network ---
    if args.network == 'cnn':
        network = GomokuCNN(
            board_size=args.board_size,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
        )
    else:
        network = GomokuTransformer(
            board_size=args.board_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
        )

    rng, init_rng = jax.random.split(rng)
    dummy = jnp.zeros((1, num_channels, args.board_size, args.board_size))
    variables = network.init(init_rng, dummy, train=False)

    if use_bn:
        params = {
            'network_params': variables['params'],
            'batch_stats': variables['batch_stats'],
        }
    else:
        params = {'network_params': variables['params']}

    # --- Optimizer ---
    optimizer = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params['network_params'])

    # --- Train step ---
    if use_bn:
        train_step_fn = make_train_step_cnn(network, optimizer)
    else:
        train_step_fn = make_train_step_transformer(network, optimizer)

    # --- Replay buffer ---
    buffer = ReplayBuffer(max_size=args.buffer_size)

    # --- Wandb ---
    if args.wandb:
        try:
            import wandb
            wandb.init(project='gomoku-alphazero', config=vars(args))
        except ImportError:
            print("wandb not installed, disabling")
            args.wandb = False

    # --- Checkpoints ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Count params ---
    param_count = sum(x.size for x in jax.tree.leaves(params['network_params']))

    print("=" * 60)
    print("  Gomoku AlphaZero Training")
    print("=" * 60)
    print(f"  Board        : {args.board_size}x{args.board_size}  "
          f"(win={args.win_length})")
    print(f"  Network      : {args.network}  "
          f"({param_count:,} params)")
    print(f"  Games/iter   : {args.games_per_iter}")
    print(f"  MCTS sims    : {args.num_simulations}")
    print(f"  Augmentation : {args.augment}")
    print(f"  Device       : {jax.devices()[0]}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for iteration in range(1, args.num_iterations + 1):
        iter_t0 = time.time()

        # ===== Self-play =====
        rng, play_rng = jax.random.split(rng)
        sp_t0 = time.time()

        s_data, p_data, pl_data, v_data, w_data = play_games(
            params, play_rng, network,
            env_reset_fn, env_step_fn, env_legal_fn, env_obs_fn,
            action_space=action_space,
            obs_shape=obs_shape,
            batch_size=args.games_per_iter,
            max_moves=max_moves,
            num_sims=args.num_simulations,
            use_bn=use_bn,
            temp=args.temp,
            temp_threshold=args.temp_threshold,
        )
        sp_time = time.time() - sp_t0

        p1w = int(np.sum(w_data == 1))
        p2w = int(np.sum(w_data == 2))
        dr = int(np.sum(w_data == 0))
        avg_len = float(np.mean(v_data.sum(axis=1)))

        # ===== Training examples =====
        tr_states, tr_policies, tr_values = make_training_examples(
            s_data, p_data, pl_data, v_data, w_data)

        if args.augment and len(tr_states) > 0:
            tr_states, tr_policies, tr_values = augment_data(
                tr_states, tr_policies, tr_values, args.board_size)

        buffer.add(tr_states, tr_policies, tr_values)

        # ===== Gradient steps =====
        tr_t0 = time.time()
        metrics_sum = None

        if buffer.size >= args.train_batch_size:
            for _ in range(args.train_steps):
                batch = buffer.sample(args.train_batch_size)

                if use_bn:
                    new_p, new_bn, opt_state, metrics = train_step_fn(
                        params['network_params'], params['batch_stats'],
                        opt_state, batch)
                    params = {
                        'network_params': new_p,
                        'batch_stats': new_bn,
                    }
                else:
                    new_p, opt_state, metrics = train_step_fn(
                        params['network_params'], opt_state, batch)
                    params = {'network_params': new_p}

                if metrics_sum is None:
                    metrics_sum = {k: float(v) for k, v in metrics.items()}
                else:
                    for k, v in metrics.items():
                        metrics_sum[k] += float(v)

        tr_time = time.time() - tr_t0
        total_time = time.time() - iter_t0

        avg_m = {}
        if metrics_sum:
            avg_m = {k: v / args.train_steps for k, v in metrics_sum.items()}

        print(
            f"Iter {iteration:3d}/{args.num_iterations} | "
            f"SP {sp_time:5.1f}s  Train {tr_time:4.1f}s | "
            f"B={p1w} W={p2w} D={dr} len={avg_len:.0f} | "
            f"Buf {buffer.size:>7,} | "
            f"Loss {avg_m.get('total_loss', 0):.4f} "
            f"(P={avg_m.get('policy_loss', 0):.4f} "
            f"V={avg_m.get('value_loss', 0):.4f})")

        # ===== Evaluation vs random =====
        if iteration % args.eval_every == 0 or iteration == 1:
            ev_t0 = time.time()
            wins, ev_draws, losses = evaluate_vs_random(
                params, network,
                env_reset_fn, env_step_fn, env_legal_fn, env_obs_fn,
                action_space, args.eval_games,
                use_bn, num_sims=args.eval_sims)
            ev_time = time.time() - ev_t0
            wr = wins / args.eval_games
            print(f"  >> Eval vs Random: {wins}W {ev_draws}D {losses}L  "
                  f"(win rate {wr:.1%})  [{ev_time:.1f}s]")

            if args.wandb:
                import wandb
                wandb.log({
                    'eval/win_rate': wr,
                    'eval/wins': wins,
                    'eval/draws': ev_draws,
                    'eval/losses': losses,
                }, step=iteration)

        # ===== Show a sample game =====
        if args.show_game_every > 0 and iteration % args.show_game_every == 0:
            show_sample_game(params, network, env_config, use_bn)

        # ===== Wandb =====
        if args.wandb and avg_m:
            import wandb
            wandb.log({
                'train/total_loss': avg_m['total_loss'],
                'train/policy_loss': avg_m['policy_loss'],
                'train/value_loss': avg_m['value_loss'],
                'self_play/black_wins': p1w,
                'self_play/white_wins': p2w,
                'self_play/draws': dr,
                'self_play/avg_length': avg_len,
                'buffer_size': buffer.size,
                'time/self_play': sp_time,
                'time/training': tr_time,
                'time/total': total_time,
            }, step=iteration)

        # ===== Checkpoint =====
        if iteration % args.save_every == 0:
            path = os.path.join(
                args.checkpoint_dir, f'gomoku_iter_{iteration:04d}.pkl')
            with open(path, 'wb') as f:
                pickle.dump({
                    'params': jax.tree.map(np.array, params),
                    'opt_state': jax.tree.map(np.array, opt_state),
                    'iteration': iteration,
                    'args': vars(args),
                }, f)
            print(f"  >> Saved {path}")

    print("\nTraining complete!")

    # Final eval with more games
    print("\nFinal evaluation (200 games)...")
    wins, draws, losses = evaluate_vs_random(
        params, network,
        env_reset_fn, env_step_fn, env_legal_fn, env_obs_fn,
        action_space, 200, use_bn, num_sims=args.eval_sims)
    print(f"  {wins}W {draws}D {losses}L  "
          f"(win rate {wins/200:.1%})")

    # Show a game
    show_sample_game(params, network, env_config, use_bn)


if __name__ == '__main__':
    main()
