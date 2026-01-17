#!/usr/bin/env python3
"""推理引擎快速测试 v3"""
import sys, os
sys.path.insert(0, os.getcwd())

print("START_TEST", flush=True)

try:
    import torch, numpy as np
    from scripts.inference_engine import DiffusionPolicyInferenceEngine
    
    print("LOAD_ENGINE", flush=True)
    engine = DiffusionPolicyInferenceEngine('checkpoints/lift_real/checkpoint-best', device='cuda', verbose=False)
    
    print("PREPARE_DATA", flush=True)
    image = np.ones((3, 480, 640), dtype=np.float32) * 0.5  
    state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    
    print("PREDICT", flush=True)
    actions = engine.predict(image, state)
    
    print(f"SUCCESS: actions shape {actions.shape}", flush=True)
    print(f"ACTIONS: {actions[0]}", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
