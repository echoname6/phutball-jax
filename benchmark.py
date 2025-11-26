"""
Benchmark: JAX vs Reference Phutball env speed
"""

import numpy as np
import jax
import jax.numpy as jnp
import time

from phutball_ref import PhutballEnvRef
from phutball_env_jax import (
    reset, step, get_legal_actions,
    EnvConfig, PhutballState
)


def benchmark_ref_env(num_games=10, max_moves_per_game=200):
    """Benchmark reference (Python) environment."""
    total_moves = 0
    start_time = time.time()
    
    for game in range(num_games):
        env = PhutballEnvRef(rows=21, cols=15)
        np.random.seed(game)
        
        for move in range(max_moves_per_game):
            legal = env.get_legal_actions()
            legal_indices = np.where(legal == 1)[0]
            if len(legal_indices) == 0:
                break
            
            action = np.random.choice(legal_indices)
            _, _, terminated, truncated, _ = env.step(int(action))
            total_moves += 1
            
            if terminated or truncated:
                break
    
    elapsed = time.time() - start_time
    return total_moves, elapsed


def benchmark_jax_env_no_jit(num_games=10, max_moves_per_game=200):
    """Benchmark JAX environment without JIT."""
    config = EnvConfig(rows=21, cols=15)
    total_moves = 0
    start_time = time.time()
    
    for game in range(num_games):
        state = reset(config)
        np.random.seed(game)
        
        for move in range(max_moves_per_game):
            legal = get_legal_actions(state, config)
            legal_np = np.array(legal)
            legal_indices = np.where(legal_np == 1)[0]
            if len(legal_indices) == 0:
                break
            
            action = np.random.choice(legal_indices)
            state = step(state, jnp.array(action, dtype=jnp.int32), config)
            total_moves += 1
            
            if bool(state.terminated):
                break
    
    elapsed = time.time() - start_time
    return total_moves, elapsed


def benchmark_jax_env_jit(num_games=10, max_moves_per_game=200):
    """Benchmark JAX environment with JIT."""
    config = EnvConfig(rows=21, cols=15)
    
    # JIT compile the functions with static config
    # Note: config must be static (not traced) for this to work
    @jax.jit
    def jit_step(state, action):
        return step(state, action, config)
    
    @jax.jit
    def jit_get_legal(state):
        return get_legal_actions(state, config)
    
    # Warm-up JIT compilation
    state = reset(config)
    _ = jit_get_legal(state)
    _ = jit_step(state, jnp.array(0, dtype=jnp.int32))
    
    total_moves = 0
    start_time = time.time()
    
    for game in range(num_games):
        state = reset(config)
        np.random.seed(game)
        
        for move in range(max_moves_per_game):
            legal = jit_get_legal(state)
            legal_np = np.array(legal)
            legal_indices = np.where(legal_np == 1)[0]
            if len(legal_indices) == 0:
                break
            
            action = np.random.choice(legal_indices)
            state = jit_step(state, jnp.array(action, dtype=jnp.int32))
            total_moves += 1
            
            if bool(state.terminated):
                break
    
    elapsed = time.time() - start_time
    return total_moves, elapsed


def benchmark_jax_batched(num_envs=64, steps_per_env=100):
    """Benchmark batched JAX environment using vmap."""
    config = EnvConfig(rows=21, cols=15)
    
    # Create batched versions
    batched_reset = jax.vmap(lambda _: reset(config))
    
    @jax.jit
    def batched_step(states, actions):
        return jax.vmap(lambda s, a: step(s, a, config))(states, actions)
    
    @jax.jit
    def batched_get_legal(states):
        return jax.vmap(lambda s: get_legal_actions(s, config))(states)
    
    # Initialize batch of states
    key = jax.random.PRNGKey(42)
    states = batched_reset(jnp.arange(num_envs))
    
    # Warm-up
    _ = batched_get_legal(states)
    dummy_actions = jnp.zeros(num_envs, dtype=jnp.int32)
    _ = batched_step(states, dummy_actions)
    
    total_steps = 0
    start_time = time.time()
    
    for step_num in range(steps_per_env):
        # Get legal actions for all envs
        legal_batch = batched_get_legal(states)
        legal_np = np.array(legal_batch)
        
        # Sample random legal action for each env
        actions = []
        for i in range(num_envs):
            legal_indices = np.where(legal_np[i] == 1)[0]
            if len(legal_indices) > 0:
                actions.append(np.random.choice(legal_indices))
            else:
                actions.append(0)  # Fallback (game ended)
        
        actions = jnp.array(actions, dtype=jnp.int32)
        states = batched_step(states, actions)
        total_steps += num_envs
    
    elapsed = time.time() - start_time
    return total_steps, elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Phutball Environment Benchmark")
    print("=" * 60)
    
    # Benchmark reference env
    print("\n1. Reference (Python) Environment")
    print("-" * 40)
    moves, elapsed = benchmark_ref_env(num_games=5, max_moves_per_game=100)
    print(f"   Total moves: {moves}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Speed: {moves/elapsed:.1f} moves/sec")
    
    # Skip non-JIT (too slow)
    print("\n2. JAX Environment (no JIT) - SKIPPED (too slow)")
    
    # Benchmark JAX with JIT
    print("\n3. JAX Environment (with JIT)")
    print("-" * 40)
    moves, elapsed = benchmark_jax_env_jit(num_games=5, max_moves_per_game=100)
    print(f"   Total moves: {moves}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Speed: {moves/elapsed:.1f} moves/sec")
    
    # Benchmark batched JAX
    print("\n4. JAX Batched (64 parallel envs)")
    print("-" * 40)
    total_steps, elapsed = benchmark_jax_batched(num_envs=64, steps_per_env=50)
    print(f"   Total steps: {total_steps}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Speed: {total_steps/elapsed:.1f} steps/sec")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")