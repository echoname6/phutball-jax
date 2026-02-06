"""
Game-agnostic AlphaZero infrastructure.

Provides MCTS, self-play, training, and evaluation functions that work
with any game environment conforming to the state contract:

    State fields: terminated, current_player (1 or 2), winner (0/1/2)

    Env functions (config pre-bound):
        env_reset() -> state
        env_step(state, action) -> state
        env_legal(state) -> (action_space,) int8
        env_obs(state) -> (channels, H, W) float32
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import mctx


# ============================================================================
# MCTS
# ============================================================================

def make_recurrent_fn(network, env_step, env_legal, env_obs,
                      use_bn, policy_post_fn=None):
    """Create the recurrent function for mctx tree search.

    Args:
        network: Flax module with (policy_logits, value) = network.apply(...)
        env_step: state, action -> state
        env_legal: state -> (action_space,) int8
        env_obs: state -> (channels, H, W) float32
        use_bn: True if network uses BatchNorm (needs batch_stats in params)
        policy_post_fn: optional (logits, states) -> logits transform
            (e.g. P2 rotation for Phutball)
    """

    def recurrent_fn(params, rng, action, embedding):
        next_states = jax.vmap(env_step)(embedding, action)

        net_in = jax.vmap(env_obs)(next_states)

        if use_bn:
            variables = {
                'params': params['network_params'],
                'batch_stats': params['batch_stats'],
            }
        else:
            variables = {'params': params['network_params']}
        policy_logits, values = network.apply(variables, net_in, train=False)

        if policy_post_fn is not None:
            policy_logits = policy_post_fn(policy_logits, next_states)

        legal = jax.vmap(env_legal)(next_states)
        masked = jnp.where(legal == 1, policy_logits, -1e9)

        terminated = next_states.terminated
        discount = jnp.where(terminated, 0.0, 1.0)

        terminal_value = jnp.where(
            next_states.winner == next_states.current_player,
            1.0,
            jnp.where(next_states.winner == 0, 0.0, -1.0),
        )
        values = jnp.where(terminated, terminal_value, values)

        return mctx.RecurrentFnOutput(
            reward=jnp.zeros_like(values),
            discount=discount,
            prior_logits=masked,
            value=values,
        ), next_states

    return recurrent_fn


def run_mcts(params, states, rng, network, action_space,
             env_legal, env_obs, num_sims, temperature, use_bn,
             recurrent_fn, dirichlet_alpha=None, policy_post_fn=None):
    """Get MCTS-improved policy for a batch of states.

    Args:
        params: dict with 'network_params' (and optionally 'batch_stats')
        states: batched game states
        rng: JAX random key
        network: Flax module
        action_space: int, total number of actions
        env_legal: state -> (action_space,) int8
        env_obs: state -> (channels, H, W) float32
        num_sims: number of MCTS simulations
        temperature: sampling temperature
        use_bn: True if network uses BatchNorm
        recurrent_fn: pre-created recurrent function
        dirichlet_alpha: Dirichlet noise param (default: 10/action_space)
        policy_post_fn: optional (logits, states) -> logits transform
    """
    # Infer batch_size from first leaf of states
    first_leaf = jax.tree.leaves(states)[0]
    batch_size = first_leaf.shape[0]

    net_in = jax.vmap(env_obs)(states)

    if use_bn:
        variables = {
            'params': params['network_params'],
            'batch_stats': params['batch_stats'],
        }
    else:
        variables = {'params': params['network_params']}

    policy_logits, values = network.apply(variables, net_in, train=False)

    if policy_post_fn is not None:
        policy_logits = policy_post_fn(policy_logits, states)

    legal = jax.vmap(env_legal)(states)
    masked = jnp.where(legal == 1, policy_logits, -1e9)

    # Dirichlet noise at the root for exploration
    rng, noise_rng = jax.random.split(rng)
    priors = jax.nn.softmax(masked, axis=-1)
    alpha = dirichlet_alpha if dirichlet_alpha is not None else 10.0 / action_space
    noise_rngs = jax.random.split(noise_rng, batch_size)
    noise = jax.vmap(
        lambda r: jax.random.dirichlet(r, jnp.full(action_space, alpha))
    )(noise_rngs)
    noisy = 0.75 * priors + 0.25 * noise
    noisy = jnp.where(legal == 1, noisy, 0.0)
    noisy = noisy / (noisy.sum(axis=-1, keepdims=True) + 1e-8)
    noisy_logits = jnp.log(noisy + 1e-8)
    noisy_logits = jnp.where(legal == 1, noisy_logits, -1e9)

    root = mctx.RootFnOutput(
        prior_logits=noisy_logits,
        value=values,
        embedding=states,
    )

    rng, mcts_rng = jax.random.split(rng)
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=mcts_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_sims,
        max_num_considered_actions=min(16, action_space),
        gumbel_scale=1.0,
    )

    mcts_policy = policy_output.action_weights
    root_values = policy_output.search_tree.node_values[:, 0]

    # Sample or pick greedily depending on temperature
    rng, sample_rng = jax.random.split(rng)
    greedy = jnp.argmax(mcts_policy, axis=-1)
    safe_temp = jnp.maximum(temperature, 1e-8)
    logits = jnp.log(mcts_policy + 1e-8) / safe_temp
    sample_rngs = jax.random.split(sample_rng, batch_size)
    sampled = jax.vmap(
        lambda r, l: jax.random.categorical(r, l))(sample_rngs, logits)
    actions = jnp.where(temperature < 0.01, greedy, sampled)

    return actions, mcts_policy, root_values


# ============================================================================
# Self-Play
# ============================================================================

def play_games(params, rng, network, env_reset, env_step, env_legal, env_obs,
               action_space, obs_shape, batch_size, max_moves,
               num_sims, use_bn, temp=1.0, temp_threshold=15,
               policy_post_fn=None):
    """Play a batch of games with MCTS, collecting training data.

    Args:
        params: network parameters dict
        rng: JAX random key
        network: Flax module
        env_reset: () -> state
        env_step: (state, action) -> state
        env_legal: state -> (action_space,) int8
        env_obs: state -> (channels, H, W) float32
        action_space: int
        obs_shape: tuple (channels, H, W)
        batch_size: number of parallel games
        max_moves: max moves per game
        num_sims: MCTS simulations per move
        use_bn: True if network uses BatchNorm
        temp: exploration temperature for early moves
        temp_threshold: after this many moves, drop to temp=0.1
        policy_post_fn: optional (logits, states) -> logits transform

    Returns:
        (states, policies, players, valid_mask, winners) as numpy arrays
    """
    # Initialise batch of games
    single = env_reset()
    states = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape).copy(),
        single,
    )

    # Numpy storage for trajectories
    all_states = np.zeros(
        (batch_size, max_moves) + obs_shape, dtype=np.float32)
    all_policies = np.zeros(
        (batch_size, max_moves, action_space), dtype=np.float32)
    all_players = np.zeros((batch_size, max_moves), dtype=np.int32)
    valid_mask = np.zeros((batch_size, max_moves), dtype=bool)
    move_count = np.zeros(batch_size, dtype=np.int32)

    # Pre-create recurrent fn once
    recurrent_fn = make_recurrent_fn(
        network, env_step, env_legal, env_obs, use_bn, policy_post_fn)

    for step_idx in range(max_moves):
        terminated = np.array(states.terminated)
        if np.all(terminated):
            break

        active = ~terminated

        # Temperature schedule
        temp_eff = temp if step_idx < temp_threshold else 0.1

        # Current observations & players
        obs = jax.vmap(env_obs)(states)
        players = np.array(states.current_player)

        # MCTS
        rng, step_rng = jax.random.split(rng)
        actions, policies, _values = run_mcts(
            params, states, step_rng, network, action_space,
            env_legal, env_obs, num_sims, temp_eff, use_bn,
            recurrent_fn, policy_post_fn=policy_post_fn,
        )

        # Store trajectories (only for active games)
        obs_np = np.array(obs)
        pol_np = np.array(policies)
        batch_idx = np.arange(batch_size)
        mc = move_count

        all_states[batch_idx[active], mc[active]] = obs_np[active]
        all_policies[batch_idx[active], mc[active]] = pol_np[active]
        all_players[batch_idx[active], mc[active]] = players[active]
        valid_mask[batch_idx[active], mc[active]] = True
        move_count[active] += 1

        # Step environments
        states = jax.vmap(env_step)(states, actions)

    winners = np.array(states.winner)
    return all_states, all_policies, all_players, valid_mask, winners


# ============================================================================
# Training Data
# ============================================================================

def make_training_examples(states, policies, players, valid_mask, winners,
                           p2_policy_transform=None):
    """Convert game trajectories to (state, policy_target, value_target).

    Args:
        states: (batch, max_moves, C, H, W)
        policies: (batch, max_moves, action_space)
        players: (batch, max_moves)
        valid_mask: (batch, max_moves) bool
        winners: (batch,)
        p2_policy_transform: optional fn(policy) -> policy for P2 examples
            (e.g. un-rotating Phutball P2 policies to match rotated obs)
    """
    batch_size, max_moves = valid_mask.shape

    all_s, all_p, all_v = [], [], []

    for g in range(batch_size):
        winner = winners[g]
        for m in range(max_moves):
            if not valid_mask[g, m]:
                continue

            player = players[g, m]
            if winner == 0:
                v = 0.0
            elif winner == player:
                v = 1.0
            else:
                v = -1.0

            policy = policies[g, m]
            if p2_policy_transform is not None and player == 2:
                policy = p2_policy_transform(policy)

            all_s.append(states[g, m])
            all_p.append(policy)
            all_v.append(v)

    if not all_s:
        ch, h, w = states.shape[2:]
        act = policies.shape[2]
        return (np.zeros((0, ch, h, w), np.float32),
                np.zeros((0, act), np.float32),
                np.zeros((0,), np.float32))

    return np.stack(all_s), np.stack(all_p), np.array(all_v, np.float32)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.states = None
        self.policies = None
        self.values = None
        self.size = 0
        self.idx = 0

    def add(self, states, policies, values):
        n = len(states)
        if n == 0:
            return
        if self.states is None:
            self.states = np.zeros(
                (self.max_size,) + states.shape[1:], np.float32)
            self.policies = np.zeros(
                (self.max_size,) + policies.shape[1:], np.float32)
            self.values = np.zeros(self.max_size, np.float32)

        if self.idx + n <= self.max_size:
            self.states[self.idx:self.idx + n] = states
            self.policies[self.idx:self.idx + n] = policies
            self.values[self.idx:self.idx + n] = values
            self.idx += n
        else:
            first = self.max_size - self.idx
            if first > 0:
                self.states[self.idx:] = states[:first]
                self.policies[self.idx:] = policies[:first]
                self.values[self.idx:] = values[:first]
            remaining = n - first
            if remaining > 0:
                self.states[:remaining] = states[first:first + remaining]
                self.policies[:remaining] = policies[first:first + remaining]
                self.values[:remaining] = values[first:first + remaining]
            self.idx = (self.idx + n) % self.max_size

        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size,
                                   replace=batch_size > self.size)
        return {
            'states': jnp.array(self.states[indices]),
            'policy_targets': jnp.array(self.policies[indices]),
            'value_targets': jnp.array(self.values[indices]),
        }


# ============================================================================
# Loss & Training Step
# ============================================================================

def compute_loss_cnn(params, batch_stats, network, batch):
    """Compute AlphaZero loss for CNN (with BatchNorm)."""
    variables = {'params': params, 'batch_stats': batch_stats}
    (policy_logits, value_preds), new_vars = network.apply(
        variables, batch['states'], train=True, mutable=['batch_stats'])

    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(
        jnp.sum(batch['policy_targets'] * log_probs, axis=-1))
    value_loss = jnp.mean(
        jnp.square(value_preds - batch['value_targets']))
    total = policy_loss + value_loss

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total,
    }
    return total, (new_vars['batch_stats'], metrics)


def compute_loss_transformer(params, network, batch):
    """Compute AlphaZero loss for Transformer (no BatchNorm)."""
    variables = {'params': params}
    policy_logits, value_preds = network.apply(
        variables, batch['states'], train=True)

    log_probs = jax.nn.log_softmax(policy_logits)
    policy_loss = -jnp.mean(
        jnp.sum(batch['policy_targets'] * log_probs, axis=-1))
    value_loss = jnp.mean(
        jnp.square(value_preds - batch['value_targets']))
    total = policy_loss + value_loss

    metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total,
    }
    return total, metrics


def make_train_step_cnn(network, optimizer):
    """Create JIT-compiled train step for CNN (with BatchNorm)."""
    @jax.jit
    def train_step(params, batch_stats, opt_state, batch):
        def loss_fn(p):
            return compute_loss_cnn(p, batch_stats, network, batch)
        (loss, (new_bn, metrics)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_bn, new_opt, metrics
    return train_step


def make_train_step_transformer(network, optimizer):
    """Create JIT-compiled train step for Transformer (no BatchNorm)."""
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            return compute_loss_transformer(p, network, batch)
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, metrics
    return train_step


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_vs_random(params, network, env_reset, env_step, env_legal,
                       env_obs, action_space, num_games, use_bn,
                       num_sims=0):
    """Play the network against a uniform-random opponent.

    Returns (wins, draws, losses) from the network's perspective.
    """
    rng = jax.random.PRNGKey(0)

    wins = draws = losses = 0

    for game_idx in range(num_games):
        rng, game_rng = jax.random.split(rng)
        state = env_reset()
        network_player = 1 if game_idx % 2 == 0 else 2

        for _ in range(action_space):
            if state.terminated:
                break
            rng, move_rng = jax.random.split(rng)

            if int(state.current_player) == network_player:
                batch_state = jax.tree.map(lambda x: x[None], state)
                if num_sims > 0:
                    recurrent_fn = make_recurrent_fn(
                        network, env_step, env_legal, env_obs, use_bn)
                    actions, _, _ = run_mcts(
                        params, batch_state, move_rng, network,
                        action_space, env_legal, env_obs,
                        num_sims, 0.0, use_bn, recurrent_fn)
                else:
                    net_in = env_obs(state)[None]
                    if use_bn:
                        variables = {
                            'params': params['network_params'],
                            'batch_stats': params['batch_stats'],
                        }
                    else:
                        variables = {'params': params['network_params']}
                    logits, _ = network.apply(
                        variables, net_in, train=False)
                    legal = env_legal(state)
                    masked = jnp.where(legal == 1, logits[0], -1e9)
                    actions = jnp.argmax(masked)[None]
                action = actions[0]
            else:
                # Random move
                legal = env_legal(state)
                probs = legal.astype(jnp.float32)
                probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
                action = jax.random.choice(move_rng, action_space, p=probs)

            state = env_step(state, action)

        w = int(state.winner)
        if w == network_player:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses
