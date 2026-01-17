#!/usr/bin/env python3
"""Minimal test to debug ACT model input format"""

import torch
import pathlib
try:
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

# Create a simple config
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

# Create model
model = ACTPolicy(config)
model = model.to("cuda")
model.train()
batch_size = 2
batch = {
    "observation.images.front": torch.randn(batch_size, 1, 3, 480, 640, dtype=torch.float32).cuda(),
    "observation.state": torch.randn(batch_size, 1, 6, dtype=torch.float32).cuda(),
    "action": torch.randn(batch_size, 8, 6, dtype=torch.float32).cuda(),
}

print(f"Batch shapes:")
for k, v in batch.items():
    print(f"  {k}: {v.shape}")

# Try forward pass
try:
    print("\nCalling model.forward()...")
    output = model(batch)
    print(f"Success! Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output[0] shape: {output[0].shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
