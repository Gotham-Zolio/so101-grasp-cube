#!/usr/bin/env python3
"""快速测试推理引擎"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from scripts.inference_engine import DiffusionPolicyInferenceEngine

print("=" * 60)
print("快速推理引擎测试")
print("=" * 60)

try:
    print("\n1. 加载模型...")
    engine = DiffusionPolicyInferenceEngine('checkpoints/lift_real/checkpoint-best', device='cuda', verbose=True)
    print(f"   State dim: {engine.state_dim}")
    print(f"   Action dim: {engine.action_dim}")
    print(f"   Horizon: {engine.horizon}")
    
    print("\n2. 测试推理...")
    image = np.random.rand(3, 480, 640).astype(np.float32) * 0.5 + 0.25
    state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    
    actions = engine.predict(image, state)
    print(f"   ✓ 推理成功!")
    print(f"   Actions shape: {actions.shape}")
    print(f"   First action: {actions[0]}")
    print(f"   Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print("\n✓ 所有测试通过!")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
