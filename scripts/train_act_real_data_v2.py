#!/usr/bin/env python3
"""
ACT 模型训练脚本 - 使用 LeRobotDataset (推荐方式)

这个脚本使用官方的 LeRobotDataset 来加载数据，避免了自定义数据集的形状问题。
LeRobotDataset 自动处理视频加载、数据归一化等问题。

用法：
    python scripts/train_act_real_data_v2.py --task lift --epochs 200
    python scripts/train_act_real_data_v2.py --task sort --epochs 200
    python scripts/train_act_real_data_v2.py --task stack --epochs 200
"""

import argparse
import pathlib
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import dataclasses
from typing import Dict, Optional
import time

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    print(f"Error: LeRobot not installed. Please install it: pip install lerobot")
    exit(1)


def create_act_config(action_dim: int, state_dim: int, n_action_steps: int = 8) -> ACTConfig:
    """Create ACT configuration"""
    config = ACTConfig(
        n_obs_steps=1,
        n_action_steps=n_action_steps,
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(state_dim,),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_dim,),
            ),
        },
    )
    return config


def train_act_model(
    task_name: str,
    dataset_repo_id: Optional[str] = None,
    output_dir: Optional[pathlib.Path] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    num_workers: int = 4,
    n_action_steps: int = 8,
    log_interval: int = 10,
):
    """
    Train ACT model using LeRobotDataset
    """
    
    if output_dir is None:
        output_dir = pathlib.Path(f"checkpoints/{task_name}_act_v2")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training ACT Model for Task: {task_name}")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    # Load dataset using LeRobotDataset
    print("\nLoading dataset...")
    if dataset_repo_id is None:
        # Use local dataset format
        # LeRobotDataset can load from local directories
        dataset_repo_id = f"lerobot/{task_name}_so101"  # Try this ID first
        print(f"Attempting to load: {dataset_repo_id}")
    
    try:
        dataset = LeRobotDataset(dataset_repo_id)
        print(f"✓ Loaded dataset: {dataset_repo_id}")
        print(f"  Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"⚠ Could not load from Hub: {e}")
        print("Please ensure the dataset is available on Hugging Face Hub")
        print("Or use the original train_act_real_data.py with local Parquet files")
        return
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Determine action and state dimensions from dataset
    sample = dataset[0]
    action_dim = sample["action"].shape[-1]
    state_dim = sample["observation.state"].shape[-1] if "observation.state" in sample else 6
    
    print(f"Action dimension: {action_dim}")
    print(f"State dimension: {state_dim}")
    
    # Create model configuration
    print("\nCreating ACT model...")
    config = create_act_config(
        action_dim=action_dim,
        state_dim=state_dim,
        n_action_steps=n_action_steps,
    )
    
    # Create model - LeRobotDataset provides stats automatically
    model = ACTPolicy(config)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # LeRobotDataset should provide batches in the correct format
                loss, info = model.forward(batch, training=True)
            except TypeError:
                # Try without training parameter
                try:
                    output = model(batch)
                    if isinstance(output, tuple):
                        predicted_actions = output[0]
                        actions = batch["action"]
                        loss = loss_fn(predicted_actions, actions)
                    else:
                        loss = output
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    raise
            
            # Backward pass
            if isinstance(loss, torch.Tensor):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch results
        avg_loss = epoch_loss / max(len(dataloader), 1)
        if (epoch + 1) % log_interval == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_dir / "model_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 50 == 0:
            checkpoint_path = output_dir / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
    
    print(f"\n✓ Training complete! Best loss: {best_loss:.4f}")
    print(f"✓ Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ACT model on robot data")
    parser.add_argument("--task", type=str, required=True, choices=["lift", "sort", "stack"],
                        help="Task to train on")
    parser.add_argument("--dataset-id", type=str, default=None,
                        help="Hugging Face Hub dataset ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")
    
    args = parser.parse_args()
    
    train_act_model(
        task_name=args.task,
        dataset_repo_id=args.dataset_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
