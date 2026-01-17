"""
使用 LeRobot ACT 模型在真机数据上训练（LeRobotDataset 版本）

这个脚本使用 LeRobot 官方的 LeRobotDataset，
相比直接加载 Parquet 的方式，提供更好的兼容性和性能优化。

用法：
    python scripts/train_act_real_data_lerobot_dataset.py --task lift
    python scripts/train_act_real_data_lerobot_dataset.py --task sort
    python scripts/train_act_real_data_lerobot_dataset.py --task stack
"""

import argparse
import pathlib
import json
import logging
from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_act_config_for_task(task_name: str) -> ACTConfig:
    """
    为特定任务创建 ACT 配置
    
    不同任务的动作维度：
    - lift: 6D (EE position + orientation delta)
    - sort: 12D (位置、姿态和抓手开度等)
    - stack: 6D (EE position + orientation delta)
    """
    
    action_dim_map = {
        "lift": 6,
        "sort": 12,
        "stack": 6,
    }
    action_dim = action_dim_map.get(task_name, 6)
    state_dim = 15  # 关节角度 + 其他状态信息
    
    logger.info(f"Creating ACT config for task '{task_name}'")
    logger.info(f"  Action dimension: {action_dim}")
    logger.info(f"  State dimension: {state_dim}")
    
    config = ACTConfig(
        # ============ 时间步长参数 ============
        n_obs_steps=1,        # 观测步数（当前 LeRobot 版本只支持 1）
        n_action_steps=8,     # 预测的动作步数（horizon）
        
        # ============ 特征定义 ============
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
        
        # ============ 训练相关参数 ============
        use_vit=False,        # 不使用 Vision Transformer
        pretrained_backbone_weights=None,  # 使用随机初始化
    )
    
    return config


def train_act_with_lerobot_dataset(
    task_name: str,
    data_dir: pathlib.Path = pathlib.Path("real_data"),
    output_dir: pathlib.Path = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    num_workers: int = 4,
    checkpoint_interval: int = 5,
):
    """
    使用 LeRobotDataset 训练 ACT 模型
    """
    
    if output_dir is None:
        output_dir = pathlib.Path(f"checkpoints/{task_name}_act")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training ACT Model with LeRobotDataset")
    logger.info(f"{'='*70}")
    logger.info(f"Task: {task_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # ============ 加载数据集 ============
    logger.info("\nLoading dataset with LeRobotDataset...")
    try:
        dataset = LeRobotDataset(
            repo_id=f"so101-grasp-cube/{task_name}",  # 或你的实际数据集名称
            local_files_only=False,
        )
        logger.info(f"✓ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.warning(f"⚠ Failed to load from repo: {e}")
        logger.info("  Falling back to local dataset construction...")
        # 如果无法从 repo 加载，构造本地数据集
        # 这里需要根据你的数据集结构调整
        raise NotImplementedError("Local dataset construction needs to be implemented")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Created dataloader with {len(dataloader)} batches")
    
    # ============ 创建模型 ============
    logger.info("\nCreating ACT model...")
    config = create_act_config_for_task(task_name)
    
    model = ACTPolicy(config)
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # ============ 优化器和调度器 ============
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # ============ 训练循环 ============
    logger.info("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            # 准备输入
            # batch 应该包含以下键：
            # - observation.images.front: (B, C, H, W)
            # - observation.state: (B, T, state_dim)
            # - action: (B, T, action_dim)
            
            # 移到设备
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # 前向传播
            optimizer.zero_grad()
            
            # 调用模型的前向方法
            # ACTPolicy 的 forward 接收：
            # - input_features: Dict[str, Tensor]
            # - output_features: Dict[str, Tensor]
            
            input_features = {
                "observation.images.front": batch_on_device.get("observation.images.front"),
                "observation.state": batch_on_device.get("observation.state"),
            }
            output_features = {
                "action": batch_on_device.get("action"),
            }
            
            # 过滤掉 None 值
            input_features = {k: v for k, v in input_features.items() if v is not None}
            output_features = {k: v for k, v in output_features.items() if v is not None}
            
            try:
                output = model(input_features)
                predicted_actions = output  # 或 output["action"] 取决于返回格式
                
                # 计算损失
                if "action" in output_features:
                    loss = loss_fn(predicted_actions, output_features["action"])
                else:
                    # 跳过此批次
                    continue
            except Exception as e:
                logger.warning(f"⚠ Error during forward pass: {e}")
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item():.4f})
        
        # 平均损失
        avg_loss = epoch_loss / max(len(dataloader), 1)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # 保存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir = output_dir / "checkpoint-best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))
            logger.info(f"✓ Saved best model (loss: {best_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))
            logger.info(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training completed!")
    logger.info(f"Best model: {output_dir / 'checkpoint-best'}")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train ACT model on real robot data using LeRobotDataset"
    )
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
        help="Real data root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    
    args = parser.parse_args()
    
    train_act_with_lerobot_dataset(
        task_name=args.task,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
