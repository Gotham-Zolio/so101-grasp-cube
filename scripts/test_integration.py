#!/usr/bin/env python3
"""
Test script to verify VideoFrameLoader (ffmpeg) integration with ParquetRealDataset
"""
import sys
import pathlib
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_real_data_lerobot import ParquetRealDataset

def test_dataset_single_sample():
    """Test loading a single sample from the dataset"""
    print("\n" + "="*60)
    print("Test 1: Loading single sample from lift task")
    print("="*60)
    
    task_dir = project_root / "real_data" / "lift"
    
    try:
        # Create dataset with image loading enabled
        dataset = ParquetRealDataset(task_dir, horizon=16, n_obs_steps=1, load_images=True)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Get first sample
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        
        # Check each key
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {key:30s} shape: {tensor.shape:20s} dtype: {tensor.dtype}")
                if 'image' in key.lower():
                    print(f"    min: {tensor.min():.4f}, max: {tensor.max():.4f}")
            else:
                print(f"  {key:30s} type: {type(tensor)}")
        
        # Verify required keys
        required_keys = ["observation.state", "action", "action_is_pad"]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        print(f"\n✓ All required keys present: {required_keys}")
        
        # Check for image key
        if "observation.images.front" in sample:
            print(f"✓ Image loaded successfully: {sample['observation.images.front'].shape}")
        else:
            print(f"⚠ No image in sample (VideoFrameLoader may not be available)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_batch():
    """Test creating a batch from dataloader"""
    print("\n" + "="*60)
    print("Test 2: Creating batch from dataloader")
    print("="*60)
    
    task_dir = project_root / "real_data" / "lift"
    
    try:
        # Create dataset
        dataset = ParquetRealDataset(task_dir, horizon=16, n_obs_steps=1, load_images=True)
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        print(f"✓ Dataloader created")
        
        # Get first batch
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {batch.keys()}")
        
        for key, tensor in batch.items():
            print(f"  {key:30s} shape: {tensor.shape}")
            if 'image' in key.lower():
                print(f"    min: {tensor.min():.4f}, max: {tensor.max():.4f}")
        
        # Verify batch shapes
        assert batch["observation.state"].shape == (4, 6), \
            f"Wrong state shape: {batch['observation.state'].shape}"
        assert batch["action"].shape == (4, 16, 6), \
            f"Wrong action shape: {batch['action'].shape}"
        assert batch["action_is_pad"].shape == (4, 16), \
            f"Wrong action_is_pad shape: {batch['action_is_pad'].shape}"
        
        print(f"\n✓ All batch shapes correct")
        
        if "observation.images.front" in batch:
            assert batch["observation.images.front"].shape == (4, 3, 480, 640), \
                f"Wrong image shape: {batch['observation.images.front'].shape}"
            print(f"✓ Image batch shape correct: {batch['observation.images.front'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_tasks():
    """Test that dataset works for different tasks"""
    print("\n" + "="*60)
    print("Test 3: Testing different tasks")
    print("="*60)
    
    tasks = ["lift", "sort", "stack"]
    base_path = project_root / "real_data"
    
    results = {}
    for task in tasks:
        task_dir = base_path / task
        try:
            if not task_dir.exists():
                print(f"⚠ {task} directory not found, skipping")
                results[task] = "skipped"
                continue
                
            dataset = ParquetRealDataset(task_dir, load_images=True)
            sample = dataset[0]
            
            # Check if image is present
            has_image = "observation.images.front" in sample
            status = "✓ images" if has_image else "⚠ no images"
            print(f"  {task:10s}: {len(dataset):5d} samples, {status}")
            results[task] = "success"
            
        except Exception as e:
            print(f"  {task:10s}: ❌ {e}")
            results[task] = "failed"
    
    return all(v != "failed" for v in results.values())


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VideoFrameLoader Integration Tests")
    print("="*60)
    
    all_passed = True
    
    # Test 1
    if not test_dataset_single_sample():
        all_passed = False
    
    # Test 2
    if not test_dataloader_batch():
        all_passed = False
    
    # Test 3
    if not test_multiple_tasks():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)
