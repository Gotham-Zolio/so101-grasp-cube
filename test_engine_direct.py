#!/usr/bin/env python3
"""
Direct test of inference engine to debug action output shape
"""

import sys
import pathlib
import numpy as np

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scripts.inference_engine import DiffusionPolicyInferenceEngine

def test_engine():
    print("=" * 70)
    print("Testing DiffusionPolicyInferenceEngine")
    print("=" * 70)
    
    # Initialize engine
    print("\n[TEST] Initializing engine...")
    try:
        engine = DiffusionPolicyInferenceEngine(
            model_path="checkpoints/lift_real/checkpoint-best",
            device="cuda",
            verbose=True
        )
        print("[TEST] ✓ Engine initialized successfully")
    except Exception as e:
        print(f"[TEST] ✗ Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check attributes
    print(f"\n[TEST] Engine attributes:")
    print(f"  horizon: {engine.horizon}")
    print(f"  action_dim: {engine.action_dim}")
    print(f"  state_dim: {engine.state_dim}")
    
    # Create dummy input
    print(f"\n[TEST] Creating dummy inputs...")
    image = np.random.rand(3, 84, 84).astype(np.float32)
    state = np.random.rand(engine.state_dim).astype(np.float32)
    print(f"  Image shape: {image.shape}")
    print(f"  State shape: {state.shape}")
    
    # Run inference
    print(f"\n[TEST] Running inference...")
    try:
        actions = engine.predict(image, state)
        print(f"[TEST] ✓ Inference completed")
        print(f"  Output type: {type(actions)}")
        print(f"  Output dtype: {actions.dtype}")
        print(f"  Output shape: {actions.shape}")
        print(f"  Output ndim: {actions.ndim}")
        print(f"  Expected shape: ({engine.horizon}, {engine.action_dim})")
        
        if actions.shape == (engine.horizon, engine.action_dim):
            print("[TEST] ✓ Output shape is correct!")
        else:
            print(f"[TEST] ✗ Output shape mismatch!")
            print(f"  Expected: ({engine.horizon}, {engine.action_dim})")
            print(f"  Got: {actions.shape}")
            
            # Try to reshape it
            if actions.size == engine.horizon * engine.action_dim:
                print(f"[TEST] Size matches, trying reshape...")
                actions_reshaped = actions.reshape(engine.horizon, engine.action_dim)
                print(f"  Reshaped: {actions_reshaped.shape}")
            
    except Exception as e:
        print(f"[TEST] ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test tolist()
    print(f"\n[TEST] Testing tolist() serialization...")
    try:
        actions_list = actions.tolist()
        print(f"  tolist() type: {type(actions_list)}")
        print(f"  tolist() len: {len(actions_list)}")
        if isinstance(actions_list[0], list):
            print(f"  tolist()[0] type: {type(actions_list[0])}")
            print(f"  tolist()[0] len: {len(actions_list[0])}")
        
        # Convert back
        actions_back = np.array(actions_list)
        print(f"  Back to numpy shape: {actions_back.shape}")
        
        if np.allclose(actions, actions_back):
            print("[TEST] ✓ Serialization round-trip successful!")
        else:
            print("[TEST] ✗ Values differ after round-trip")
            
    except Exception as e:
        print(f"[TEST] ✗ Serialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
