#!/usr/bin/env python3
"""直接测试推理引擎修复"""
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.getcwd())

print(f"Python: {sys.executable}")
print(f"CWD: {os.getcwd()}")

# 测试导入
try:
    print("\n1. Importing modules...")
    import torch
    print(f"   ✓ torch: {torch.__version__}")
    
    import numpy as np
    print(f"   ✓ numpy: {np.__version__}")
    
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    print(f"   ✓ LeRobot DiffusionPolicy")
    
    from scripts.inference_engine import DiffusionPolicyInferenceEngine
    print(f"   ✓ DiffusionPolicyInferenceEngine")
    
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# 测试加载模型
try:
    print("\n2. Loading model...")
    engine = DiffusionPolicyInferenceEngine(
        'checkpoints/lift_real/checkpoint-best',
        device='cuda',
        verbose=True
    )
    print(f"   ✓ Model loaded")
    print(f"     - State dim: {engine.state_dim}")
    print(f"     - Action dim: {engine.action_dim}")
    print(f"     - Horizon: {engine.horizon}")
    
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试推理
try:
    print("\n3. Testing inference...")
    
    # 创建简单的输入
    image = np.ones((3, 480, 640), dtype=np.float32) * 0.5  # 使用480x640，推理引擎会自动调整到84x84
    state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    
    print(f"   Input image shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"   Input state shape: {state.shape}")
    
    import time
    start = time.time()
    actions = engine.predict(image, state)
    elapsed = time.time() - start
    
    print(f"   ✓ Inference successful!")
    print(f"     - Output shape: {actions.shape}")
    print(f"     - Output range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"     - First action: {actions[0]}")
    print(f"     - Inference time: {elapsed*1000:.2f} ms")
    
except Exception as e:
    print(f"   ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
