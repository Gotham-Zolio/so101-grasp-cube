#!/usr/bin/env python3
import numpy as np
from scripts.inference_engine import DiffusionPolicyInferenceEngine

# 加载SORT模型
print("=== Loading SORT model ===")
engine_sort = DiffusionPolicyInferenceEngine("checkpoints/sort_real/checkpoint-best", verbose=True)
print(f"State dim: {engine_sort.state_dim}")
print(f"Action dim: {engine_sort.action_dim}")

# 检查stats
print(f"\nStats keys: {list(engine_sort.stats.keys())}")
if "observation.state" in engine_sort.stats:
    print(f"observation.state stats: {engine_sort.stats['observation.state']}")

# 创建输入
image = np.random.rand(3, 480, 640).astype(np.float32)
state = np.zeros(engine_sort.state_dim, dtype=np.float32)

print(f"\nInput state shape: {state.shape}")
print(f"Stats observation.state mean length: {len(engine_sort.stats.get('observation.state', {}).get('mean', []))}")

# 调试推理
print("\n=== Starting inference ===")
try:
    actions = engine_sort.predict(image, state)
    print(f"Success! Actions shape: {actions.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
