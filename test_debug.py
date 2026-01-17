#!/usr/bin/env python3
"""快速推理引擎测试 - 带详细调试"""
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.getcwd())

print(f"Python: {sys.executable}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

# 测试导入
print("\n1. Importing modules...", flush=True)
import torch
print(f"   ✓ torch: {torch.__version__}", flush=True)

import numpy as np
print(f"   ✓ numpy: {np.__version__}", flush=True)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
print(f"   ✓ LeRobot DiffusionPolicy", flush=True)

from scripts.inference_engine import DiffusionPolicyInferenceEngine
print(f"   ✓ DiffusionPolicyInferenceEngine", flush=True)

# 测试加载模型
print("\n2. Loading model...", flush=True)
engine = DiffusionPolicyInferenceEngine(
    'checkpoints/lift_real/checkpoint-best',
    device='cuda',
    verbose=True
)
print(f"   ✓ Model loaded", flush=True)
print(f"     - State dim: {engine.state_dim}", flush=True)
print(f"     - Action dim: {engine.action_dim}", flush=True)
print(f"     - Horizon: {engine.horizon}", flush=True)

# 测试推理
print("\n3. Testing inference...", flush=True)

# 创建简单的输入
image = np.ones((3, 480, 640), dtype=np.float32) * 0.5  
state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)

print(f"   Input image shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]", flush=True)
print(f"   Input state shape: {state.shape}", flush=True)

import time
start = time.time()

try:
    print("   Calling engine.predict()...", flush=True)
    actions = engine.predict(image, state)
    elapsed = time.time() - start
    
    print(f"   ✓ Inference successful!", flush=True)
    print(f"     - Output shape: {actions.shape}", flush=True)
    print(f"     - Output range: [{actions.min():.4f}, {actions.max():.4f}]", flush=True)
    print(f"     - First action: {actions[0]}", flush=True)
    print(f"     - Inference time: {elapsed*1000:.2f} ms", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ All tests passed!", flush=True)
    print("="*70, flush=True)
    
except Exception as e:
    elapsed = time.time() - start
    print(f"   ✗ Inference failed after {elapsed*1000:.2f}ms: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
