#!/usr/bin/env python3
"""
Debug script to check batch contents during training
"""
import sys
import pathlib
import torch
from torch.utils.data import DataLoader

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_batch_contents():
    """Check what's in the batch from the dataloader"""
    from scripts.train_real_data_lerobot import ParquetRealDataset
    
    task_dir = project_root / "real_data" / "lift"
    
    print("Creating dataset with load_images=True...")
    dataset = ParquetRealDataset(task_dir, horizon=16, n_obs_steps=1, load_images=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Dataloader created with batch_size=4")
    
    # Get first batch
    print("\nGetting first batch...")
    batch = next(iter(dataloader))
    
    print(f"\nBatch keys: {batch.keys()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:30s}: shape={str(value.shape):20s} dtype={value.dtype}")
        else:
            print(f"  {key:30s}: {type(value)}")
    
    # Check if image is present
    if "observation.images.front" in batch:
        img = batch["observation.images.front"]
        print(f"\n✓ Image is in batch!")
        print(f"  Shape: {img.shape}")
        print(f"  Dtype: {img.dtype}")
        print(f"  Min: {img.min():.4f}, Max: {img.max():.4f}")
        return True
    else:
        print(f"\n✗ Image NOT in batch!")
        print(f"Available keys: {list(batch.keys())}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Batch Contents Debug")
    print("="*60 + "\n")
    
    success = check_batch_contents()
    
    print("\n" + "="*60)
    if success:
        print("✓ Image is properly in batch")
    else:
        print("✗ Image is missing from batch")
    print("="*60 + "\n")
    
    sys.exit(0 if success else 1)
