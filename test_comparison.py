"""
Comparison tests: JAX Phutball env vs Reference Phutball env

This ensures the JAX implementation matches the original exactly.
"""

import numpy as np
import jax.numpy as jnp

# Import both environments
from phutball_ref import PhutballEnvRef as RefEnv
from phutball_env_jax import (
    reset, step, get_legal_actions, render_board,
    EnvConfig, PhutballState, EMPTY, BALL, MAN
)


def jax_state_to_numpy(state: PhutballState):
    """Convert JAX state to numpy for comparison."""
    return {
        'board': np.array(state.board),
        'ball_pos': tuple(np.array(state.ball_pos)),
        'current_player': int(state.current_player),
        'is_jumping': bool(state.is_jumping),
        'terminated': bool(state.terminated),
        'winner': int(state.winner),
        'num_turns': int(state.num_turns),
    }


def ref_state_snapshot(env: RefEnv):
    """Get comparable state from Reference env."""
    return {
        'board': env.board.copy(),
        'ball_pos': env._phutball_pos,
        'current_player': env._current_player,
        'is_jumping': bool(env.is_jump_seq_underway()),
        'terminated': env._winner != 0,
        'winner': env._winner,
        'num_turns': env._num_turns,
    }


def compare_states(jax_state, pytorch_state, context=""):
    """Compare JAX and Reference states, raise on mismatch."""
    jax_np = jax_state_to_numpy(jax_state)
    
    # Compare boards
    if not np.array_equal(jax_np['board'], pytorch_state['board']):
        print(f"\n{context} Board mismatch!")
        print("JAX board:")
        print(jax_np['board'])
        print("Reference board:")
        print(pytorch_state['board'])
        raise AssertionError(f"{context} Board mismatch")
    
    # Compare ball position
    if jax_np['ball_pos'] != pytorch_state['ball_pos']:
        raise AssertionError(f"{context} Ball position mismatch: JAX={jax_np['ball_pos']} vs PT={pytorch_state['ball_pos']}")
    
    # Compare current player
    if jax_np['current_player'] != pytorch_state['current_player']:
        raise AssertionError(f"{context} Current player mismatch: JAX={jax_np['current_player']} vs PT={pytorch_state['current_player']}")
    
    # Compare is_jumping
    if jax_np['is_jumping'] != pytorch_state['is_jumping']:
        raise AssertionError(f"{context} is_jumping mismatch: JAX={jax_np['is_jumping']} vs PT={pytorch_state['is_jumping']}")


def compare_legal_actions(jax_legal, pytorch_legal, context=""):
    """Compare legal action masks."""
    jax_np = np.array(jax_legal)
    pytorch_np = np.array(pytorch_legal)
    
    if not np.array_equal(jax_np, pytorch_np):
        # Find differences
        diff_indices = np.where(jax_np != pytorch_np)[0]
        print(f"\n{context} Legal action mismatch at {len(diff_indices)} positions:")
        for idx in diff_indices[:10]:  # Show first 10
            print(f"  Action {idx}: JAX={jax_np[idx]} vs PT={pytorch_np[idx]}")
        if len(diff_indices) > 10:
            print(f"  ... and {len(diff_indices) - 10} more")
        raise AssertionError(f"{context} Legal action mismatch")


def test_initial_state():
    """Test that initial states match."""
    # Reference env
    ref_env = RefEnv(rows=21, cols=15)
    ref_state = ref_state_snapshot(ref_env)
    
    # JAX env
    config = EnvConfig(rows=21, cols=15)
    jax_state = reset(config)
    
    compare_states(jax_state, ref_state, "Initial state:")
    print("✓ test_initial_state passed")


def test_legal_actions_initial():
    """Test that initial legal actions match."""
    # Reference
    ref_env = RefEnv(rows=21, cols=15)
    ref_legal = ref_env.get_legal_actions()
    
    # JAX
    config = EnvConfig(rows=21, cols=15)
    jax_state = reset(config)
    jax_legal = get_legal_actions(jax_state, config)
    
    compare_legal_actions(jax_legal, ref_legal, "Initial legal actions:")
    print("✓ test_legal_actions_initial passed")


def test_placement_sequence():
    """Test a sequence of placements."""
    config = EnvConfig(rows=21, cols=15)
    
    # Both envs start fresh
    ref_env = RefEnv(rows=21, cols=15)
    jax_state = reset(config)
    
    # Place men at several positions
    placements = [
        (9, 7),   # Above ball
        (11, 7),  # Below ball
        (10, 6),  # Left of ball
        (10, 8),  # Right of ball
    ]
    
    for i, (row, col) in enumerate(placements):
        action = row * 15 + col  # Placement action
        
        # Reference step
        ref_env.step(action)
        ref_state = ref_state_snapshot(ref_env)
        
        # JAX step
        jax_state = step(jax_state, jnp.array(action, dtype=jnp.int32), config)
        
        compare_states(jax_state, ref_state, f"After placement {i+1}:")
    
    print("✓ test_placement_sequence passed")


def test_jump_sequence():
    """Test a jump and halt sequence."""
    config = EnvConfig(rows=21, cols=15)
    total_positions = config.rows * config.cols
    
    # Start fresh
    ref_env = RefEnv(rows=21, cols=15)
    jax_state = reset(config)
    
    # Place a man above the ball (at 9, 7)
    place_action = 9 * 15 + 7
    ref_env.step(place_action)
    jax_state = step(jax_state, jnp.array(place_action, dtype=jnp.int32), config)
    
    compare_states(jax_state, ref_state_snapshot(ref_env), "After placement:")
    
    # Compare legal actions after placement
    ref_legal = ref_env.get_legal_actions()
    jax_legal = get_legal_actions(jax_state, config)
    compare_legal_actions(jax_legal, ref_legal, "Legal actions after placement:")
    
    # Now jump (player 2 is playing, jumping to 8,7)
    jump_action = total_positions + 8 * 15 + 7
    ref_env.step(jump_action)
    jax_state = step(jax_state, jnp.array(jump_action, dtype=jnp.int32), config)
    
    compare_states(jax_state, ref_state_snapshot(ref_env), "After jump:")
    
    # Compare legal actions during jump sequence
    ref_legal = ref_env.get_legal_actions()
    jax_legal = get_legal_actions(jax_state, config)
    compare_legal_actions(jax_legal, ref_legal, "Legal actions during jump:")
    
    # Halt
    halt_action = 2 * total_positions
    ref_env.step(halt_action)
    jax_state = step(jax_state, jnp.array(halt_action, dtype=jnp.int32), config)
    
    compare_states(jax_state, ref_state_snapshot(ref_env), "After halt:")
    
    print("✓ test_jump_sequence passed")


def test_random_game(seed=42, num_moves=50):
    """Play a random game in both envs and compare at each step."""
    np.random.seed(seed)
    config = EnvConfig(rows=21, cols=15)
    
    ref_env = RefEnv(rows=21, cols=15)
    jax_state = reset(config)
    
    for move_num in range(num_moves):
        # Get legal actions
        ref_legal = ref_env.get_legal_actions()
        jax_legal = get_legal_actions(jax_state, config)
        
        compare_legal_actions(jax_legal, ref_legal, f"Move {move_num} legal actions:")
        
        # Pick a random legal action
        legal_indices = np.where(ref_legal == 1)[0]
        if len(legal_indices) == 0:
            print(f"No legal actions at move {move_num}")
            break
        
        action = np.random.choice(legal_indices)
        
        # Execute in both
        ref_obs, ref_reward, ref_term, ref_trunc, ref_info = ref_env.step(int(action))
        jax_state = step(jax_state, jnp.array(action, dtype=jnp.int32), config)
        
        ref_state = ref_state_snapshot(ref_env)
        compare_states(jax_state, ref_state, f"Move {move_num} after action {action}:")
        
        # Check termination
        if ref_term or ref_trunc:
            print(f"Game ended at move {move_num}, winner: {ref_env._winner}")
            break
    
    print(f"✓ test_random_game passed ({move_num + 1} moves)")


if __name__ == "__main__":
    print("Running comparison tests: JAX vs Reference Phutball env\n")
    
    test_initial_state()
    test_legal_actions_initial()
    test_placement_sequence()
    test_jump_sequence()
    test_random_game(seed=42, num_moves=100)
    test_random_game(seed=123, num_moves=100)
    test_random_game(seed=999, num_moves=100)
    
    print("\n✓ All comparison tests passed!")