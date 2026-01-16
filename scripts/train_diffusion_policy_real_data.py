"""
改进的训练脚本：直接加载 LeRobot 真机数据进行训练

相比原始版本的改进：
1. 直接使用 LeRobotDataset 加载真机数据（Parquet 格式）
2. 自动从 stats.json 读取数据统计信息用于归一化
3. 支持多任务（pick, lift, stack, sort）
4. 使用 LeRobot 官方的 preprocessor 和 postprocessor
"""

import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from typing import Dict, Any, Optional
import time

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot not installed. Please install it: pip install lerobot")


class RealDatasetLoader:
    """
    加载真机数据的便利类
    
    真机数据结构：
    - real_data/
      - lift/
        - data/chunk-000/file-*.parquet
        - meta/info.json, stats.json
      - sort/
        - ...
    """
    
    def __init__(self, dataset_path: pathlib.Path, task_name: str):
        self.dataset_path = pathlib.Path(dataset_path)
        self.task_name = task_name  # 'lift', 'sort', 'stack', etc.
        
        # Load metadata
        info_path = self.dataset_path / self.task_name / "meta" / "info.json"
        stats_path = self.dataset_path / self.task_name / "meta" / "stats.json"
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        print(f"Loaded metadata for task '{task_name}':")
        print(f"  - Total episodes: {self.info['total_episodes']}")
        print(f"  - Total frames: {self.info['total_frames']}")
        print(f"  - Robot type: {self.info['robot_type']}")
    
    def load_dataset(self) -> LeRobotDataset:
        """
        Load LeRobot dataset directly from parquet files
        """
        dataset_dir = self.dataset_path / self.task_name
        
        # LeRobotDataset expects a repo_id or local_files_dir
        dataset = LeRobotDataset(
            repo_id="",  # Not using huggingface hub
            root=dataset_dir,
        )
        return dataset, self.stats
    
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """
        Extract normalization statistics from stats.json
        
        Returns:
            Dict with keys like 'action', 'observation.state' containing
            mean and std for normalization
        """
        norm_stats = {}
        for key in ['action', 'observation.state']:
            if key in self.stats:
                stats = self.stats[key]
                norm_stats[key] = {
                    'mean': np.array(stats.get('mean', [0.0])),
                    'std': np.array(stats.get('std', [1.0])),
                    'min': np.array(stats.get('min', [-1.0])),
                    'max': np.array(stats.get('max', [1.0])),
                }
        return norm_stats


def train(
    dataset_path: pathlib.Path,
    task_name: str,
    output_dir: pathlib.Path,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_freq: int = 10,
    num_workers: int = 4,
):
    """
    Train Diffusion Policy on real robot data
    
    Args:
        dataset_path: Path to real_data folder (e.g., 'real_data')
        task_name: Task name (e.g., 'lift', 'sort')
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cuda or cpu)
        save_freq: Save checkpoint every N epochs
        num_workers: Number of data loading workers
    """
    
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot required. Install with: pip install lerobot")
    
    device = torch.device(device)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/5] Loading real robot data for task: {task_name}")
    loader = RealDatasetLoader(dataset_path, task_name)
    dataset, stats = loader.load_dataset()
    
    print(f"\n[2/5] Creating model and processors")
    # Create a default diffusion config
    # Adjust these hyperparameters based on your task complexity
    policy_config = DiffusionConfig(
        n_obs_steps=1,
        horizon=16,
        n_action_steps=8,
        n_latent_steps=4,
        input_shapes={
            "observation.state": [6],  # Single arm
            "observation.images.front": [480, 640, 3],
        },
        output_shapes={
            "action": [6]
        },
        # Task encoding
        condition_shapes={
            "task": [64]  # Text embedding dimension
        }
    )
    
    # Create model
    model = DiffusionPolicy(policy_config)
    model = model.to(device)
    
    # Create preprocessor and postprocessor using LeRobot's official factory
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=None,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        }
    )
    
    print(f"\n[3/5] Creating data loaders")
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Batches per epoch: {len(dataloader)}")
    
    print(f"\n[4/5] Setting up optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Training loop
    print(f"\n[5/5] Starting training for {num_epochs} epochs")
    print("=" * 60)
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Preprocess batch
            # LeRobot datasets return dicts with properly formatted observations
            batch_processed = preprocessor(batch)
            
            # Forward pass
            # DiffusionPolicy.compute_loss expects the batch in a specific format
            with torch.no_grad():
                pass  # Preprocessor handles data movement to device
            
            # Predict action
            loss = model.compute_loss(batch_processed)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix(loss=loss.item())
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch+1} - Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_dir = output_path / f"checkpoint-{epoch+1:03d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))
            
            # Save training info
            serializable_config = {}
            for key, value in policy_config.__dict__.items():
                if isinstance(value, (PolicyFeature,)):
                    serializable_config[key] = {
                        "type": value.type.name if hasattr(value.type, 'name') else str(value.type),
                        "shape": list(value.shape),
                    }
                elif isinstance(value, (NormalizationMode,)):
                    serializable_config[key] = value.name if hasattr(value, 'name') else str(value)
                else:
                    serializable_config[key] = value
            
            training_info = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": current_lr,
                "global_step": global_step,
                "model_config": serializable_config,
            }
            with open(checkpoint_dir / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            print(f"  Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_dir = output_path / "checkpoint-best"
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_checkpoint_dir))
            print(f"  Saved best model (loss: {best_loss:.6f}) to {best_checkpoint_dir}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"All checkpoints saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy on real robot data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on lift task
  python train_diffusion_policy_real_data.py \\
    --dataset-path real_data \\
    --task-name lift \\
    --output-dir checkpoints/lift_real

  # Train on sort task with more epochs
  python train_diffusion_policy_real_data.py \\
    --dataset-path real_data \\
    --task-name sort \\
    --output-dir checkpoints/sort_real \\
    --num-epochs 300
        """
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to real_data directory containing task folders (lift, sort, stack, etc.)"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        choices=['lift', 'sort', 'stack', 'pick'],
        help="Task name (folder name in dataset-path)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    train(
        dataset_path=args.dataset_path,
        task_name=args.task_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
