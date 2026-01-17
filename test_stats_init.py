#!/usr/bin/env python3
"""Minimal test to check if ACTPolicy can be initialized with stats"""

import torch
import json
import numpy as np
from pathlib import Path

# Load real stats
stats_path = Path("real_data/lift/meta/stats.json")
with open(stats_path) as f:
    loaded_stats = json.load(f)

print(f"Loaded stats keys: {list(loaded_stats.keys())}")

# Convert to tensors
dataset_stats = {}
for key, stat_dict in loaded_stats.items():
    if isinstance(stat_dict, dict) and any(k in stat_dict for k in ["mean", "std"]):
        tensor_dict = {}
        for stat_key in ["mean", "std"]:
            if stat_key in stat_dict:
                val = stat_dict[stat_key]
                if isinstance(val, list):
                    tensor_dict[stat_key] = torch.from_numpy(np.array(val, dtype=np.float32))
        if tensor_dict:
            dataset_stats[key] = tensor_dict

print(f"\nConverted stats keys: {list(dataset_stats.keys())}")
print(f"Sample - observation.state:")
if "observation.state" in dataset_stats:
    for k, v in dataset_stats["observation.state"].items():
        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")

print(f"\nTrying to create ACTPolicy...")
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    
    config = ACTConfig(
        n_obs_steps=1,
        n_action_steps=8,
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(6,),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,),
            ),
        },
    )
    
    # Try with dataset_stats
    print("Trying with dataset_stats...")
    model = ACTPolicy(config, dataset_stats=dataset_stats)
    print("✓ Success with dataset_stats!")
    
except TypeError as e:
    print(f"✗ dataset_stats failed: {e}")
    try:
        print("Trying with stats...")
        model = ACTPolicy(config, stats=dataset_stats)
        print("✓ Success with stats!")
    except TypeError as e2:
        print(f"✗ stats failed: {e2}")
        print("Creating without stats...")
        model = ACTPolicy(config)
        print("✓ Created without stats")
        
        # Check if we can call normalize_inputs
        print(f"\nChecking normalize_inputs...")
        if hasattr(model, 'normalize_inputs'):
            print(f"  Has normalize_inputs: {type(model.normalize_inputs)}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
