#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import torch
from scripts.train_act_real_data import (
    RealDataACTDataset, collate_fn, create_act_config,
    ACTPolicy
)
import pathlib

# Quick test
dataset = RealDataACTDataset(
    task_dir=pathlib.Path("real_data/lift"),
    horizon=16,
    n_obs_steps=1,
    load_images=True,
)

print(f"Dataset size: {len(dataset)}")

# Create model
config = create_act_config(action_dim=6, state_dim=6)
model = ACTPolicy(config)
model = model.cuda()
model.train()

# Get a single batch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
batch = next(iter(loader))

print(f"\nBatch shapes:")
print(f"  images: {batch['observation']['images'].shape}")
print(f"  states: {batch['observation']['states'].shape}")
print(f"  actions: {batch['action'].shape}")

# Try with different input formats
print("\n=== Testing with original shapes ===")
batch_input = {
    "observation.images.front": batch['observation']['images'].cuda(),
    "observation.state": batch['observation']['states'].cuda(),
    "action": batch['action'].cuda(),
}

print(f"Input shapes:")
for k, v in batch_input.items():
    print(f"  {k}: {v.shape}")

try:
    output = model(batch_input)
    print("✓ Success with original shapes!")
except Exception as e:
    print(f"✗ Error: {e}")
    
    # Try squeezed version
    print("\n=== Testing with squeezed n_obs_steps ===")
    batch_input["observation.images.front"] = batch['observation']['images'].squeeze(1).cuda()
    batch_input["observation.state"] = batch['observation']['states'].squeeze(1).cuda()
    
    print(f"Input shapes after squeeze:")
    for k, v in batch_input.items():
        print(f"  {k}: {v.shape}")
    
    try:
        output = model(batch_input)
        print("✓ Success with squeezed shapes!")
        if isinstance(output, tuple):
            print(f"  Output[0] shape: {output[0].shape}")
    except Exception as e2:
        print(f"✗ Still error: {e2}")
