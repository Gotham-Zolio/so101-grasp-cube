"""
Debug script to inspect actual batch data shapes and content
"""
import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import pathlib
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import the dataset class
from scripts.train_real_data_lerobot import ParquetRealDataset

# Load lift task data
task_dir = pathlib.Path("/home/gotham/shared/so101-grasp-cube/real_data/lift")

print(f"Loading dataset from {task_dir}")
dataset = ParquetRealDataset(task_dir, horizon=16, n_obs_steps=1)

print(f"Dataset size: {len(dataset)}")

# Get one sample
print("\n" + "="*60)
print("SINGLE SAMPLE FROM DATASET")
print("="*60)
sample = dataset[0]
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"{key:30s}: shape={value.shape}, dtype={value.dtype}, min={value.min():.4f}, max={value.max():.4f}")
    else:
        print(f"{key:30s}: {type(value)} = {value}")

# Get a batch
print("\n" + "="*60)
print("BATCH FROM DATALOADER")
print("="*60)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(dataloader))

for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"{key:30s}: shape={value.shape}, dtype={value.dtype}, min={value.min():.4f}, max={value.max():.4f}")
    else:
        print(f"{key:30s}: {type(value)}")

# Print details about image tensor
print("\n" + "="*60)
print("IMAGE TENSOR DETAILS")
print("="*60)
if "observation.images.front" in batch:
    img = batch["observation.images.front"]
    print(f"Image shape: {img.shape}")
    print(f"Expected shape for model: (batch_size, 3, height, width)")
    print(f"Actual: (batch_size={img.shape[0]}, channels={img.shape[1]}, height={img.shape[2]}, width={img.shape[3]})")
    
    # Check first sample
    print(f"\nFirst sample in batch:")
    print(f"  shape: {img[0].shape}")
    print(f"  value range: [{img[0].min():.4f}, {img[0].max():.4f}]")

# Print details about state tensor
print("\n" + "="*60)
print("STATE TENSOR DETAILS")
print("="*60)
if "observation.state" in batch:
    state = batch["observation.state"]
    print(f"State shape: {state.shape}")
    print(f"Expected: (batch_size, n_obs_steps*state_dim) or (batch_size, state_dim)")
    print(f"Actual: (batch_size={state.shape[0]}, dim={state.shape[1]})")

# Print details about action tensor
print("\n" + "="*60)
print("ACTION TENSOR DETAILS")
print("="*60)
if "action" in batch:
    action = batch["action"]
    print(f"Action shape: {action.shape}")
    print(f"Expected: (batch_size, horizon, action_dim)")
    print(f"Actual: (batch_size={action.shape[0]}, ...)") 
    print(f"Full shape: {action.shape}")
