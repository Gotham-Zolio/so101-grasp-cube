#!/usr/bin/env python3
"""
Quick test to verify image batch shape after fix
"""
import sys
import pathlib
import torch
from torch.utils.data import DataLoader

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_real_data_lerobot import ParquetRealDataset

task_dir = project_root / "real_data" / "lift"
dataset = ParquetRealDataset(task_dir, load_images=True)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

batch = next(iter(dataloader))
print("Batch shapes:")
for key, val in batch.items():
    print(f"  {key:30s}: {val.shape}")

print(f"\nExpected image shape: (batch_size=2, n_obs_steps=1, 3, 480, 640)")
print(f"Actual image shape:   {batch['observation.images.front'].shape}")

if batch['observation.images.front'].shape == torch.Size([2, 1, 3, 480, 640]):
    print("✓ Image shape is correct!")
else:
    print("✗ Image shape mismatch!")
