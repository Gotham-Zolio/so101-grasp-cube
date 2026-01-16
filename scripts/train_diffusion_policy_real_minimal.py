"""
最小化改动方案：只替换数据加载，其他逻辑保持不变

将原来的 NPZDataset 替换为 LeRobotDataset，
但保留现有的训练循环和优化器逻辑
"""

import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from typing import Dict, Any, Optional, List

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot not installed. Please install it: pip install lerobot")


class LeRobotRealDataAdapter(Dataset):
    """
    Adapter to make LeRobotDataset compatible with existing training loop.
    
    LeRobotDataset returns dicts with keys like:
    - action
    - observation.state
    - observation.images.front
    - observation.images.wrist
    - task
    
    We need to convert these to the format expected by the existing training code.
    """
    
    def __init__(self, lerobot_dataset: LeRobotDataset, horizon: int = 16, n_obs_steps: int = 1):
        self.dataset = lerobot_dataset
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.episode_boundaries = self._find_episode_boundaries()
    
    def _find_episode_boundaries(self) -> List[tuple]:
        """Find start and end indices for each episode"""
        boundaries = []
        current_episode = None
        start_idx = 0
        
        for i in range(len(self.dataset)):
            episode_idx = self.dataset[i].get('episode_index', 0)
            
            if current_episode is None:
                current_episode = episode_idx
            elif episode_idx != current_episode:
                boundaries.append((start_idx, i))
                start_idx = i
                current_episode = episode_idx
        
        if start_idx < len(self.dataset):
            boundaries.append((start_idx, len(self.dataset)))
        
        return boundaries
    
    def __len__(self) -> int:
        return len(self.dataset) - self.horizon
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which episode this index belongs to
        episode_start, episode_end = None, None
        for start, end in self.episode_boundaries:
            if start <= idx < end - self.horizon:
                episode_start, episode_end = start, end
                break
        
        if episode_start is None:
            # Fallback to simple slicing
            episode_start = max(0, idx - self.horizon)
            episode_end = min(len(self.dataset), idx + self.horizon)
        
        # Sample starting frame
        max_start = min(idx, episode_end - self.horizon - 1)
        start_frame = np.random.randint(max(episode_start, idx - self.horizon), max_start + 1)
        
        # Get observation frames
        obs_start = max(episode_start, start_frame - self.n_obs_steps + 1)
        obs_end = start_frame + 1
        
        # Collect observations
        state_sequence = []
        image_front_sequence = []
        image_wrist_sequence = []
        
        for frame_idx in range(obs_start, obs_end):
            sample = self.dataset[frame_idx]
            state_sequence.append(sample['observation.state'])
            image_front_sequence.append(sample['observation.images.front'])
            image_wrist_sequence.append(sample['observation.images.wrist'])
        
        # Pad if necessary
        while len(state_sequence) < self.n_obs_steps:
            state_sequence.insert(0, state_sequence[0])
            image_front_sequence.insert(0, image_front_sequence[0])
            image_wrist_sequence.insert(0, image_wrist_sequence[0])
        
        state_sequence = torch.stack([torch.from_numpy(s).float() for s in state_sequence])
        image_front_sequence = torch.stack(image_front_sequence)
        image_wrist_sequence = torch.stack(image_wrist_sequence)
        
        # Collect actions
        action_sequence = []
        for frame_idx in range(start_frame, min(start_frame + self.horizon, episode_end)):
            sample = self.dataset[frame_idx]
            action_sequence.append(sample['action'])
        
        # Pad actions if episode ends early
        while len(action_sequence) < self.horizon:
            action_sequence.append(action_sequence[-1])
        
        action_sequence = torch.stack([torch.from_numpy(a).float() for a in action_sequence])
        
        return {
            "state": state_sequence,
            "front_image": image_front_sequence,
            "wrist_image": image_wrist_sequence,
            "action": action_sequence,
        }


def train(
    dataset_path: str,
    task_name: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_freq: int = 10,
    num_workers: int = 4,
):
    device = torch.device(device)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Loading LeRobot real data for task: {task_name}")
    
    # Load dataset
    dataset_dir = pathlib.Path(dataset_path) / task_name
    lerobot_dataset = LeRobotDataset(repo_id="", root=dataset_dir)
    dataset = LeRobotRealDataAdapter(lerobot_dataset, horizon=16, n_obs_steps=1)
    
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - LeRobot raw size: {len(lerobot_dataset)}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    
    # Create model
    print("\n>>> Creating Diffusion Policy model")
    policy_config = DiffusionConfig(
        n_obs_steps=1,
        horizon=16,
    )
    model = DiffusionPolicy.from_config(policy_config)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print("\n>>> Starting training")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass - compute loss
            # Adjust this based on your actual DiffusionPolicy API
            loss = model.compute_loss(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            ckpt_dir = output_path / f"checkpoint-{epoch+1:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"  Saved checkpoint: {ckpt_dir}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = output_path / "checkpoint-best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on real robot data")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to real_data")
    parser.add_argument("--task-name", type=str, required=True, help="Task: lift, sort, stack, pick")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
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
