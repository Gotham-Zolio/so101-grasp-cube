"""
直接使用 Parquet 文件和自定义训练循环训练真机数据
不依赖 LeRobotDataset（避免网络调用）
"""

import pandas as pd
import pathlib
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional, Dict, Any
import glob

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    import pyarrow.parquet as pq
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot or PyArrow not installed.")

# Import video frame loader
try:
    from scripts.video_frame_loader_ffmpeg import VideoFrameLoader
    VIDEO_LOADER_AVAILABLE = True
except ImportError:
    VIDEO_LOADER_AVAILABLE = False


class ParquetRealDataset(Dataset):
    """
    直接从 Parquet 文件加载真机数据
    避免使用 LeRobotDataset（会产生网络调用）
    """
    
    def __init__(self, task_dir: pathlib.Path, horizon: int = 16, n_obs_steps: int = 1, load_images: bool = True):
        self.task_dir = pathlib.Path(task_dir)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.load_images = load_images
        
        # 加载所有 parquet 文件
        parquet_files = sorted(glob.glob(str(self.task_dir / "data" / "chunk-*" / "*.parquet")))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.task_dir}/data/")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # 加载所有数据到内存（如果文件太大可以改为逐个加载）
        self.data = []
        for pq_file in parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            self.data.append(df)
            print(f"  Loaded {len(df)} frames from {pathlib.Path(pq_file).name}")
        
        # 合并所有数据
        self.df = pd.concat(self.data, ignore_index=True)
        print(f"Total frames: {len(self.df)}")
        
        # 初始化 VideoFrameLoader 用于加载视频帧
        self.video_loader = None
        if self.load_images and VIDEO_LOADER_AVAILABLE:
            try:
                self.video_loader = VideoFrameLoader(self.task_dir)
                print("✓ VideoFrameLoader initialized for image loading")
            except Exception as e:
                print(f"⚠ Failed to initialize VideoFrameLoader: {e}")
                print("  Continuing with state-only training")
                self.load_images = False
        
        # 找出 episode 边界
        self.episode_boundaries = self._find_episode_boundaries()
        print(f"Found {len(self.episode_boundaries)} episodes")
    
    def _find_episode_boundaries(self):
        """找出每个 episode 的开始和结束索引"""
        boundaries = []
        if 'episode_index' not in self.df.columns:
            # 如果没有 episode_index，把整个数据集当作一个 episode
            boundaries.append((0, len(self.df)))
            return boundaries
        
        current_episode = None
        start_idx = 0
        
        for i, episode_idx in enumerate(self.df['episode_index'].values):
            if current_episode is None:
                current_episode = episode_idx
            elif episode_idx != current_episode:
                boundaries.append((start_idx, i))
                start_idx = i
                current_episode = episode_idx
        
        if start_idx < len(self.df):
            boundaries.append((start_idx, len(self.df)))
        
        return boundaries
    
    def __len__(self):
        return len(self.df) - self.horizon
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 找到该索引所在的 episode
        episode_start, episode_end = None, None
        for start, end in self.episode_boundaries:
            if start <= idx < end - self.horizon:
                episode_start, episode_end = start, end
                break
        
        if episode_start is None:
            episode_start = max(0, idx - self.horizon)
            episode_end = min(len(self.df), idx + self.horizon)
        
        # 采样起始帧
        max_start = min(idx, episode_end - self.horizon - 1)
        start_frame = np.random.randint(max(episode_start, idx - self.horizon), max_start + 1)
        
        # 收集观测帧
        obs_start = max(episode_start, start_frame - self.n_obs_steps + 1)
        obs_end = start_frame + 1
        
        # 获取动作序列
        action_sequence = []
        action_is_pad = []  # Track which actions are padding
        for frame_idx in range(start_frame, min(start_frame + self.horizon, episode_end)):
            row = self.df.iloc[frame_idx]
            action = self._extract_action(row)
            action_sequence.append(action)
            action_is_pad.append(False)  # Real action, not padding
        
        # 补齐 action
        num_real_actions = len(action_sequence)
        while len(action_sequence) < self.horizon:
            action_sequence.append(action_sequence[-1])  # Repeat last action
            action_is_pad.append(True)  # This is a padding action
        
        action_sequence = torch.stack([torch.from_numpy(a).float() for a in action_sequence])
        action_is_pad = torch.tensor(action_is_pad, dtype=torch.bool)
        
        # 获取观测（仅状态，无图像）
        obs_dict = {}
        for frame_idx in range(obs_start, obs_end):
            row = self.df.iloc[frame_idx]
            state = self._extract_state(row)
            obs_dict.setdefault('state', []).append(state)
        
        # 补齐观测
        while len(obs_dict.get('state', [])) < self.n_obs_steps:
            obs_dict['state'].insert(0, obs_dict['state'][0])
        
        state_tensor = torch.stack([torch.from_numpy(s).float() for s in obs_dict['state']])
        # Keep shape (n_obs_steps, 6) - LeRobot expects time dimension to be present
        
        # 加载图像序列（对应观测窗口的所有帧）
        batch = {
            "observation.state": state_tensor,
            "action": action_sequence,
            "action_is_pad": action_is_pad,
        }

        # Load images for all observation steps (same range as states: obs_start to obs_end)
        images = []
        if self.load_images and self.video_loader is not None:
            for frame_idx in range(obs_start, obs_end):
                try:
                    image = self.video_loader.get_frame(frame_idx)
                    if image is not None:
                        images.append(torch.from_numpy(image).float())
                    else:
                        images.append(torch.zeros((3, 480, 640), dtype=torch.float32))
                except Exception as e:
                    print(f"Warning: Failed to load image for frame {frame_idx}: {e}")
                    images.append(torch.zeros((3, 480, 640), dtype=torch.float32))
        
        # Pad image sequence if needed
        while len(images) < self.n_obs_steps:
            if images:
                images.insert(0, images[0].clone())  # Repeat first image
            else:
                images.insert(0, torch.zeros((3, 480, 640), dtype=torch.float32))
        
        # Stack images to shape (n_obs_steps, 3, H, W)
        image_tensor = torch.stack(images)
        batch["observation.images.front"] = image_tensor
        
        return batch
    
    def _extract_action(self, row) -> np.ndarray:
        """从行数据中提取动作"""
        if 'action' in row:
            return np.array(row['action'], dtype=np.float32)
        # 如果没有 action 列，尝试从 observation 构造
        return np.zeros(6, dtype=np.float32)
    
    def _extract_state(self, row) -> np.ndarray:
        """从行数据中提取状态（关节角度）"""
        if 'observation.state' in row:
            state = row['observation.state']
            if isinstance(state, np.ndarray):
                return state.astype(np.float32)
            return np.array(state, dtype=np.float32)
        # 如果没有状态列，返回零向量
        return np.zeros(6, dtype=np.float32)
    
    def _extract_image(self, row) -> np.ndarray:
        """从行数据中提取图像（RGB，3x480x640）"""
        if 'observation.images.front' in row:
            img = row['observation.images.front']
            if isinstance(img, np.ndarray):
                # 确保是 (C, H, W) 格式且是 float32
                if img.shape != (3, 480, 640):
                    # 如果形状不对，尝试转换
                    if img.shape == (480, 640, 3):
                        img = np.transpose(img, (2, 0, 1))
                return img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
            return np.zeros((3, 480, 640), dtype=np.float32)
        # 如果没有图像列，返回零图像
        return np.zeros((3, 480, 640), dtype=np.float32)


def compute_dataset_stats(dataset: ParquetRealDataset) -> Dict[str, Any]:
    """
    Compute normalization statistics from the entire dataset
    """
    print("  Computing dataset statistics...")
    
    all_states = []
    all_images = []
    all_actions = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    for batch in dataloader:
        if "observation.state" in batch:
            all_states.append(batch["observation.state"].numpy().reshape(-1, 6))
        if "observation.images.front" in batch:
            all_images.append(batch["observation.images.front"].numpy().reshape(-1, 3, 480, 640))
        if "action" in batch:
            all_actions.append(batch["action"].numpy().reshape(-1, 6))
    
    stats = {}
    
    if all_states:
        states = np.concatenate(all_states, axis=0)
        stats["observation.state"] = {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        }
    
    if all_images:
        images = np.concatenate(all_images, axis=0)
        # Normalize per channel
        images_flat = images.reshape(len(images), 3, -1)
        stats["observation.images.front"] = {
            "mean": images_flat.mean(axis=(0, 2)).tolist(),
            "std": images_flat.std(axis=(0, 2)).tolist(),
        }
    
    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
        stats["action"] = {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
        }
    
    return stats


def normalize_batch(batch: Dict[str, torch.Tensor], stats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Normalize batch data using computed statistics
    Only normalize state and action, skip images (they're already [0, 1])
    """
    normalized = {}
    for key, tensor in batch.items():
        # Skip image keys - don't normalize them
        if 'image' in key.lower():
            normalized[key] = tensor
            continue
        
        # Only normalize if we have stats for this key
        if key not in stats or not isinstance(stats[key], dict):
            normalized[key] = tensor
            continue
        
        stat = stats[key]
        mean = stat.get('mean', 0)
        std = stat.get('std', 1)
        
        # Convert to tensors if needed
        if isinstance(mean, list):
            mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
        if isinstance(std, list):
            std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
        
        # Reshape mean/std for broadcasting
        if len(tensor.shape) == 2:  # (B, D) - state/action
            mean = mean.reshape(1, -1) if mean.dim() == 1 else mean
            std = std.reshape(1, -1) if std.dim() == 1 else std
        elif len(tensor.shape) == 1:  # (D,) - single sample
            pass  # Keep as is
        else:
            # For higher dim tensors, try to keep original shape
            normalized[key] = tensor
            continue
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)
        
        # Normalize
        normalized[key] = (tensor - mean) / std
    
    return normalized


class IdentityModule(torch.nn.Module):
    """A module that returns input unchanged (identity operation)"""
    def forward(self, x):
        return x


class DiffusionPolicyWrapper(torch.nn.Module):
    """
    Wrapper around DiffusionPolicy that applies manual normalization
    to avoid the infinity stats error
    """
    def __init__(self, policy: DiffusionPolicy, stats: Dict[str, Any], device: str = "cuda"):
        super().__init__()
        self.policy = policy
        self.stats = stats
        self.device = device
        
        # Replace all normalizers with identity modules to disable their normalization
        # We'll handle normalization manually before passing to the policy
        if hasattr(policy, 'normalize_inputs'):
            self.policy.normalize_inputs = IdentityModule()
        if hasattr(policy, 'normalize_outputs'):
            self.policy.normalize_outputs = IdentityModule()
        if hasattr(policy, 'normalize_targets'):
            self.policy.normalize_targets = IdentityModule()
        
        # Also try common aliases
        for attr_name in dir(policy):
            if 'norm' in attr_name.lower() and hasattr(getattr(policy, attr_name), 'forward'):
                try:
                    setattr(self.policy, attr_name, IdentityModule())
                except:
                    pass
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        # Debug: log batch keys
        if not hasattr(self, '_logged_keys'):
            print(f"DEBUG: Batch keys in DiffusionPolicyWrapper.forward: {batch.keys()}")
            self._logged_keys = True
        
        # Manually normalize the batch before passing to policy
        normalized_batch = normalize_batch(batch, self.stats)
        
        # Debug: log normalized batch keys
        if not hasattr(self, '_logged_norm_keys'):
            print(f"DEBUG: Normalized batch keys: {normalized_batch.keys()}")
            self._logged_norm_keys = True
        
        # Call the policy's forward with normalized data
        # All the policy's normalizers are now identity, so won't interfere
        loss, _ = self.policy(normalized_batch)
        return loss, None
    
    def parameters(self):
        return self.policy.parameters()
    
    def to(self, device):
        self.policy = self.policy.to(device)
        self.device = device
        return self
    
    def train(self):
        self.policy.train()
        return self
    
    def eval(self):
        self.policy.eval()
        return self
    
    def save_pretrained(self, path: str):
        """Save the underlying policy model"""
        self.policy.save_pretrained(path)



def train(
    task_name: str,
    dataset_path: str = "real_data",
    output_dir: str = "checkpoints",
    num_epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_freq: int = 10,
    num_workers: int = 0,  # 设置为 0 以避免多进程问题
):
    """
    Train Diffusion Policy directly on real robot parquet data
    """
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot or PyArrow required.")
    
    device = torch.device(device)
    task_dir = pathlib.Path(dataset_path) / task_name
    output_path = pathlib.Path(output_dir) / f"{task_name}_real"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training '{task_name}' task on real robot data")
    print(f"{'='*60}")
    print(f"Dataset: {task_dir.absolute()}")
    print(f"Output:  {output_path.absolute()}")
    print(f"Epochs:  {num_epochs}")
    print(f"Batch:   {batch_size}")
    print(f"LR:      {learning_rate}")
    print(f"Device:  {device}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("[1/5] Loading parquet data...")
    try:
        import pandas as pd
        dataset = ParquetRealDataset(task_dir, horizon=16, n_obs_steps=1, load_images=True)
        print(f"  ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        raise
    
    # Create dataloader
    print("[2/5] Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    print(f"  ✓ Dataloader ready ({len(dataloader)} batches)")
    
    # Create model
    print("[3/5] Creating Diffusion Policy model...")
    
    # Compute or load stats for normalization
    stats_file = task_dir / "data" / "stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"  ✓ Loaded normalization stats from {stats_file}")
    else:
        print(f"  ⚠ No stats.json found, computing from dataset...")
        stats = compute_dataset_stats(dataset)
        # Save stats for future use
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved stats to {stats_file}")
    
    # Create output checkpoint directory
    checkpoint_dir = output_path / "checkpoint-best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # DiffusionConfig with proper features
    # Vision + State policy (images from video files)
    policy_config = DiffusionConfig(
        n_obs_steps=1,
        horizon=16,
        # Input features: front camera image + joint state
        input_features={
            "observation.images.front": PolicyFeature(
                shape=(3, 480, 640),
                type=FeatureType.VISUAL,
            ),
            "observation.state": PolicyFeature(
                shape=(6,),
                type=FeatureType.STATE,
            ),
        },
        # Output feature: joint actions
        output_features={
            "action": PolicyFeature(
                shape=(6,),
                type=FeatureType.ACTION,
            ),
        },
    )
    
    # Create initial model
    print("  Creating and initializing model...")
    model = DiffusionPolicy(policy_config)
    model = model.to(device)
    
    # Wrap the model to handle normalization manually
    model = DiffusionPolicyWrapper(model, stats, device)
    
    print(f"  ✓ Model created")
    
    # Optimizer
    print("[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print(f"  ✓ Optimizer ready")
    
    # Training loop
    print(f"[5/5] Starting training for {num_epochs} epochs\n")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass - the wrapper handles normalization
            # Returns (loss, None)
            loss, _ = model(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
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
            
            training_info = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": current_lr,
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
            print(f"  Saved best model (loss: {best_loss:.6f})")
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print(f"📁 Checkpoints saved to: {output_path}")
    print(f"   Best model: {output_path}/checkpoint-best/")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy on real robot data using LeRobot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train lift task with default settings
  python scripts/train_real_data_lerobot.py --task lift
  
  # Train sort task with custom epochs and batch size
  python scripts/train_real_data_lerobot.py \\
    --task sort \\
    --num-epochs 300 \\
    --batch-size 64
  
  # Train stack task on CPU
  python scripts/train_real_data_lerobot.py \\
    --task stack \\
    --device cpu
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["lift", "sort", "stack", "pick"],
        help="Task to train"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="real_data",
        help="Root path to real_data directory (default: real_data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints (default: checkpoints)"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cuda)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        train(
            task_name=args.task,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
