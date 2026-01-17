#!/usr/bin/env python3
"""Test dataset creation and batch shapes"""

import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import pathlib
from torch.utils.data import DataLoader
import torch
import numpy as np

# Try importing the dataset class from the training script
exec(open('scripts/train_act_real_data.py').read(), globals())

# Create dataset
task_dir = pathlib.Path("real_data/lift")
dataset = RealDataACTDataset(
    task_dir=task_dir,
    horizon=16,
    n_obs_steps=1,
    load_images=True,
    state_normalization=False,
    action_normalization=False,
)

print(f"Dataset size: {len(dataset)}")

# Get one sample
print("\n=== Single Sample ===")
sample = dataset[0]
print(f"observation type: {type(sample['observation'])}")
for key, val in sample['observation'].items():
    if isinstance(val, np.ndarray):
        print(f"  {key}: shape {val.shape}, dtype {val.dtype}")
    else:
        print(f"  {key}: {type(val)}")
print(f"action shape: {sample['action'].shape}")

# Create batch
print("\n=== Batch via DataLoader ===")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    for key, val in batch['observation'].items():
        if isinstance(val, torch.Tensor):
            print(f"  observation.{key}: shape {val.shape}")
    print(f"  action: shape {batch['action'].shape}")
    
    if batch_idx == 0:
        break
