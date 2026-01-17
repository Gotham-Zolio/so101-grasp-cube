"""
使用 LeRobot ACT 模型在真机数据上训练

相比 DiffusionPolicy 的优点：
1. 推理速度更快（一步输出完整序列，不需要扩散过程）
2. 内存占用更小
3. 架构更简洁（Transformer 基础）

支持三个任务：
- lift (6维动作)
- sort (12维动作)  
- stack (6维动作)

用法：
    python scripts/train_act_real_data.py --task lift --output-dir checkpoints/lift_act
    python scripts/train_act_real_data.py --task sort --output-dir checkpoints/sort_act
    python scripts/train_act_real_data.py --task stack --output-dir checkpoints/stack_act
"""

import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import dataclasses
from typing import Dict, Any, Optional, Tuple
import time
import glob

import pandas as pd

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    import pyarrow.parquet as pq
    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    print(f"Error: LeRobot not installed or import failed: {e}")

# Import video frame loader
try:
    from scripts.video_frame_loader_ffmpeg import VideoFrameLoader
    VIDEO_LOADER_AVAILABLE = True
except ImportError:
    VIDEO_LOADER_AVAILABLE = False


class IdentityModule(torch.nn.Module):
    """A module that returns input unchanged (identity operation)"""
    def forward(self, x):
        return x


class RealDataACTDataset(Dataset):
    """
    真机数据集，用于 ACT 模型训练
    从 Parquet 文件直接加载，避免网络调用
    """
    
    def __init__(
        self, 
        task_dir: pathlib.Path, 
        horizon: int = 16,  # 预测步数
        n_obs_steps: int = 1,  # 观测步数
        load_images: bool = True,
        state_normalization: bool = True,
        action_normalization: bool = True,
        stats: Optional[Dict] = None
    ):
        self.task_dir = pathlib.Path(task_dir)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.load_images = load_images
        self.state_normalization = state_normalization
        self.action_normalization = action_normalization
        
        # 加载所有 parquet 文件
        parquet_files = sorted(glob.glob(str(self.task_dir / "data" / "chunk-*" / "*.parquet")))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.task_dir}/data/")
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # 加载所有数据
        self.data = []
        for pq_file in parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()
            self.data.append(df)
            print(f"  Loaded {len(df)} frames from {pathlib.Path(pq_file).name}")
        
        # 合并所有数据
        self.df = pd.concat(self.data, ignore_index=True)
        print(f"Total frames: {len(self.df)}")
        
        # 自动检测状态和动作维度
        sample_state = self._extract_state(self.df.iloc[0])
        self.state_dim = len(sample_state)
        print(f"Detected state dimension: {self.state_dim}")
        
        sample_action = self._extract_action(self.df.iloc[0])
        self.action_dim = len(sample_action)
        print(f"Detected action dimension: {self.action_dim}")
        
        # 加载统计信息用于归一化
        self.stats = stats or self._load_stats()
        
        # 初始化 VideoFrameLoader
        self.video_loader = None
        if self.load_images and VIDEO_LOADER_AVAILABLE:
            try:
                self.video_loader = VideoFrameLoader(self.task_dir)
                print("✓ VideoFrameLoader initialized for image loading")
            except Exception as e:
                print(f"⚠ Failed to initialize VideoFrameLoader: {e}")
                self.load_images = False
        
        # 找出 episode 边界
        self.episode_boundaries = self._find_episode_boundaries()
        print(f"Found {len(self.episode_boundaries)} episodes")
    
    def _load_stats(self) -> Dict:
        """从 meta/stats.json 加载统计信息"""
        stats_path = self.task_dir / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            print(f"Loaded stats from {stats_path}")
            return stats
        else:
            print(f"⚠ No stats.json found, will compute on the fly")
            return {}
    
    def _find_episode_boundaries(self) -> list:
        """找出每个 episode 的开始和结束索引"""
        boundaries = []
        if 'episode_index' not in self.df.columns:
            boundaries.append((0, len(self.df)))
            return boundaries
        
        episode_indices = self.df['episode_index'].values
        current_ep = episode_indices[0]
        start = 0
        
        for i, ep in enumerate(episode_indices):
            if ep != current_ep:
                boundaries.append((start, i))
                start = i
                current_ep = ep
        boundaries.append((start, len(self.df)))
        return boundaries
    
    def _extract_state(self, row) -> np.ndarray:
        """提取状态向量"""
        # 首先检查是否有 'observation.state' 列（通常包含完整的状态数组）
        if 'observation.state' in row.index:
            state = row['observation.state']
            if isinstance(state, np.ndarray):
                return state.astype(np.float32)
            else:
                return np.array(state, dtype=np.float32)
        # 备选：查找以 'observation.state' 开头的列
        state_keys = [k for k in row.index if k.startswith('observation.state')]
        if state_keys:
            return np.array([row[k] for k in state_keys], dtype=np.float32)
        # 备选：使用关节角度
        joint_keys = [k for k in row.index if 'joint' in k.lower()]
        if joint_keys:
            return np.array([row[k] for k in joint_keys], dtype=np.float32)
        return np.zeros(6, dtype=np.float32)
    
    def _extract_action(self, row) -> np.ndarray:
        """提取动作向量"""
        # 首先检查是否有 'action' 列（通常包含完整的动作数组）
        if 'action' in row.index:
            action = row['action']
            if isinstance(action, np.ndarray):
                return action.astype(np.float32)
            else:
                return np.array(action, dtype=np.float32)
        # 备选：查找以 'action' 开头的列
        action_keys = [k for k in row.index if k.startswith('action')]
        if action_keys:
            return np.array([row[k] for k in action_keys], dtype=np.float32)
        return np.zeros(6, dtype=np.float32)
    
    def _load_image(self, timestamp: int) -> np.ndarray:
        """加载对应 timestamp 的图像"""
        if self.video_loader is None:
            return np.random.rand(3, 480, 640).astype(np.float32)
        try:
            image = self.video_loader.get_frame(timestamp)
            if image is None:
                return np.random.rand(3, 480, 640).astype(np.float32)
            return image
        except Exception as e:
            print(f"⚠ Failed to load image for timestamp {timestamp}: {e}")
            return np.random.rand(3, 480, 640).astype(np.float32)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """使用统计信息归一化状态"""
        if not self.state_normalization or "observation.state" not in self.stats:
            return state
        
        stat = self.stats.get("observation.state", {})
        if "mean" in stat and "std" in stat:
            mean = np.array(stat["mean"], dtype=np.float32)
            std = np.array(stat["std"], dtype=np.float32)
            return (state - mean) / (std + 1e-6)
        return state
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """使用统计信息归一化动作"""
        if not self.action_normalization or "action" not in self.stats:
            return action
        
        stat = self.stats.get("action", {})
        if "mean" in stat and "std" in stat:
            mean = np.array(stat["mean"], dtype=np.float32)
            std = np.array(stat["std"], dtype=np.float32)
            return (action - mean) / (std + 1e-6)
        return action
    
    def __len__(self) -> int:
        """返回能生成的有效样本数"""
        total_valid_samples = 0
        for start, end in self.episode_boundaries:
            episode_len = end - start
            # 每个 episode 能生成 (episode_len - horizon) 个样本
            valid_in_episode = max(0, episode_len - self.horizon)
            total_valid_samples += valid_in_episode
        return total_valid_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取一个样本"""
        # 找出这个 idx 属于哪个 episode
        sample_count = 0
        for start, end in self.episode_boundaries:
            episode_len = end - start
            valid_in_episode = max(0, episode_len - self.horizon)
            
            if sample_count + valid_in_episode > idx:
                # 在这个 episode 中
                local_idx = idx - sample_count
                global_idx = start + local_idx
                break
            sample_count += valid_in_episode
        
        # 获取观测
        obs_rows = [self.df.iloc[global_idx]]
        for i in range(1, self.n_obs_steps):
            if global_idx - i >= 0:
                obs_rows.insert(0, self.df.iloc[global_idx - i])
        
        # 加载观测数据
        observations = {
            "images": [],
            "states": []
        }
        
        for row in obs_rows:
            if self.load_images:
                timestamp = int(row.get('timestamp_us', 0)) // 1000  # 转换为ms
                image = self._load_image(timestamp)
                observations["images"].append(image)
            state = self._extract_state(row)
            state = self._normalize_state(state)
            observations["states"].append(state)
        
        # 栈化 (shape: (n_obs_steps, C, H, W) for images, (n_obs_steps, state_dim) for states)
        if self.load_images:
            observations["images"] = np.stack(observations["images"], axis=0)  # (n_obs_steps, 3, 480, 640)
        observations["states"] = np.stack(observations["states"], axis=0)  # (n_obs_steps, state_dim)
        
        # 获取动作序列（horizon 步）
        actions = []
        for i in range(self.horizon):
            if global_idx + i < len(self.df):
                action = self._extract_action(self.df.iloc[global_idx + i])
                action = self._normalize_action(action)
                actions.append(action)
            else:
                # 越界则填充零
                actions.append(np.zeros(self.action_dim, dtype=np.float32))
        actions = np.stack(actions, axis=0)
        
        return {
            "observation": observations,
            "action": actions,
        }


def collate_fn(batch):
    """自定义 collate 函数，正确处理观测和动作维度"""
    observations = {
        "images": [],
        "states": []
    }
    actions = []
    
    for item in batch:
        obs = item["observation"]
        if "images" in obs:
            observations["images"].append(obs["images"])
        observations["states"].append(obs["states"])
        actions.append(item["action"])
    
    result = {
        "observation": {},
        "action": torch.from_numpy(np.stack(actions, axis=0)).float()  # (B, horizon, action_dim)
    }
    
    if observations["images"]:
        # images: (B, n_obs_steps, 3, 480, 640)
        result["observation"]["images"] = torch.from_numpy(
            np.stack(observations["images"], axis=0)
        ).float()
    
    # states: (B, n_obs_steps, state_dim)
    result["observation"]["states"] = torch.from_numpy(
        np.stack(observations["states"], axis=0)
    ).float()
    
    return result


def create_act_config(action_dim: int, state_dim: int, n_action_steps: int = 8, n_obs_steps: int = 1, chunk_size: int = 16) -> ACTConfig:
    """
    创建 ACT 配置，使用 LeRobot 官方默认参数
    注意：ACT 只支持 n_obs_steps=1
    """
    # LeRobot ACT 配置 - 简化参数
    config = ACTConfig(
        # 时间步长参数
        n_obs_steps=n_obs_steps,  # 观测步数（ACT仅支持1）
        n_action_steps=n_action_steps,  # 预测的动作步数（默认 8）
        chunk_size=chunk_size,  # 动作预测块大小 (必须与 dataset horizon 一致)
        
        # 特征定义（无规范化参数）
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
    data_dir: pathlib.Path = pathlib.Path("real_data"),
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
    训练 ACT 模型
    """
    
    if output_dir is None:
        output_dir = pathlib.Path(f"checkpoints/{task_name}_act")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_dir = data_dir / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    print(f"\n{'='*70}")
    print(f"Training ACT Model for Task: {task_name}")
    print(f"{'='*70}")
    print(f"Data directory: {task_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    # 创建数据集
    print("\nLoading dataset...")
    dataset = RealDataACTDataset(
        task_dir=task_dir,
        horizon=16,
        n_obs_steps=1,  # ACT 仅支持 n_obs_steps=1
        load_images=True,
        state_normalization=True,
        action_normalization=True,
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,  # 使用自定义 collate 函数
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # 创建 ACT 配置和模型
    print("\nCreating ACT model...")
    config = create_act_config(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=1,   # ACT 只支持 n_obs_steps=1
        chunk_size=16,   # 必须与 dataset horizon (16) 一致
    )
    
    # 为模型准备统计数据（用于规范化）
    # 将 dataset.stats（JSON dict）转换为 PyTorch tensors，用于 ACTPolicy
    dataset_stats = {}
    if dataset.stats:
        print(f"\nPreparing stats for model...")
        print(f"Available stat keys: {list(dataset.stats.keys())}")
        for key, stat_dict in dataset.stats.items():
            if isinstance(stat_dict, dict) and any(k in stat_dict for k in ["mean", "std", "min", "max"]):
                tensor_dict = {}
                for stat_key in ["mean", "std", "min", "max"]:
                    if stat_key in stat_dict:
                        val = stat_dict[stat_key]
                        if isinstance(val, (list, np.ndarray)):
                            tensor_dict[stat_key] = torch.from_numpy(np.array(val, dtype=np.float32))
                        elif isinstance(val, torch.Tensor):
                            tensor_dict[stat_key] = val
                if tensor_dict:
                    dataset_stats[key] = tensor_dict
                    print(f"  {key}: {list(tensor_dict.keys())}")
    
    print(f"Dataset stats prepared: {list(dataset_stats.keys())}")
    
    # 使用 dataset_stats 创建模型（与 DiffusionPolicy 相同的参数名）
    try:
        print(f"Creating ACTPolicy with dataset_stats...")
        model = ACTPolicy(config, dataset_stats=dataset_stats)
    except TypeError as e:
        print(f"  dataset_stats not supported: {e}")
        # 如果 ACTPolicy 不支持 dataset_stats 参数，试试 stats
        try:
            print(f"Creating ACTPolicy with stats...")
            model = ACTPolicy(config, stats=dataset_stats)
        except TypeError as e2:
            print(f"  stats not supported: {e2}")
            # 如果都不支持，直接创建不带 stats 的模型
            print(f"Creating ACTPolicy without stats...")
            model = ACTPolicy(config)
    
    model = model.to(device)
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数
    loss_fn = nn.MSELoss()
    
    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        # 使用 dataclass 的 asdict 或直接保存配置
        config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    
    # 保存统计信息
    if dataset.stats:
        stats_path = output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(dataset.stats, f, indent=2)
        print(f"Saved stats to {stats_path}")
    
    # 训练循环
    print("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            # 移到设备
            observations = {
                k: v.to(device) for k, v in batch["observation"].items()
                if isinstance(v, torch.Tensor)
            }
            actions = batch["action"].to(device)  # (B, horizon, action_dim)
            
            # DEBUG: Print shapes
            if batch_idx == 0 and epoch == 0:
                print(f"\n=== BATCH DEBUG INFO (before moving to device) ===")
                print(f"Batch keys: {batch.keys()}")
                if "observation" in batch:
                    print(f"Observation keys: {batch['observation'].keys()}")
                    for k, v in batch['observation'].items():
                        if isinstance(v, torch.Tensor):
                            print(f"  observation[{k}]: shape {v.shape}, dtype {v.dtype}")
                print(f"Action: shape {batch['action'].shape if isinstance(batch['action'], torch.Tensor) else 'not tensor'}")
            
            # 移到设备
            observations = {
                k: v.to(device) for k, v in batch["observation"].items()
                if isinstance(v, torch.Tensor)
            }
            actions = batch["action"].to(device)  # (B, horizon, action_dim)
            
            # DEBUG: Print shapes after device move
            if batch_idx == 0 and epoch == 0:
                print(f"\n=== AFTER MOVING TO DEVICE ===")
                for k, v in observations.items():
                    print(f"  observation[{k}]: shape {v.shape}")
                print(f"  action: shape {actions.shape}")
                print("=======================\n")
            
            # 前向传播
            optimizer.zero_grad()
            
            # 构建输入字典，包含动作（训练模式需要）
            # ACT 期望输入格式：observation.state: (B, state_dim), observation.images.*: (B, C, H, W)
# 我们有n_obs_steps=1的数据，需要移除单例时间维度
            # 否则 VAE encoder 的 torch.cat 会失败 (3D vs 4D)
            # 并且 ResNet backbone 不支持 5D 输入
            images_raw = observations["images"]  # (B, 1, C, H, W)
            states_raw = observations["states"]  # (B, 1, state_dim)
            
            # Squeeze n_obs_steps=1
            images = images_raw.squeeze(1) if images_raw.shape[1] == 1 else images_raw
            states = states_raw.squeeze(1) if states_raw.shape[1] == 1 else states_raw
            
            # ACT 模型需要 action_is_pad 标记
            batch_size = actions.shape[0]
            horizon = actions.shape[1]
            action_is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
            
            batch_input = {
                "observation.images.front": images,  # (B, C, H, W)
                "observation.state": states,         # (B, state_dim)
                "action": actions,
                "action_is_pad": action_is_pad,  # ACT 需要这个掩码
            }
            
            # DEBUG: Print input shapes before model
            if batch_idx == 0 and epoch == 0:
                print(f"\n=== BATCH INPUT SHAPES TO MODEL ===")
                for k, v in batch_input.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape {v.shape}")
                print("===================================\n")
            
            # 直接调用模型的 forward 方法
            model.train()
            output = model(batch_input)
            
            # ACT 的输出格式：(actions_hat, (mu_hat, log_sigma_x2_hat))
            # 我们需要 actions_hat 用于损失计算
            if isinstance(output, tuple):
                predicted_actions = output[0]  # 提取 actions_hat
            else:
                predicted_actions = output
            
            # 确保输出形状正确 (B, horizon, action_dim)
            if predicted_actions.shape != actions.shape:
                # 如果形状不匹配，尝试调整
                if len(predicted_actions.shape) == 2 and len(actions.shape) == 3:
                    # 如果是 (B, action_dim)，需要扩展为 (B, horizon, action_dim)
                    predicted_actions = predicted_actions.unsqueeze(1).expand_as(actions)
            
            # 计算损失
            loss = loss_fn(predicted_actions, actions)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 平均损失
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        # 日志
        if (epoch + 1) % log_interval == 0:
            print(f"\nEpoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # 保存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir = output_dir / "checkpoint-best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(str(checkpoint_dir))
            print(f"✓ Saved best model to {checkpoint_dir} (loss: {best_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best model saved to: {output_dir / 'checkpoint-best'}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Train ACT model on real robot data")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["lift", "sort", "stack"],
        help="Task name",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("real_data"),
        help="Root directory of real data",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for checkpoints (default: checkpoints/{task}_act)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4, LeRobot official value)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Number of action steps to predict (default: 8, LeRobot official value)",
    )
    
    args = parser.parse_args()
    
    if not LEROBOT_AVAILABLE:
        raise RuntimeError("LeRobot is not installed. Please install it: pip install lerobot")
    
    train_act_model(
        task_name=args.task,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_workers=args.num_workers,
        n_action_steps=args.n_action_steps,
    )


if __name__ == "__main__":
    main()
