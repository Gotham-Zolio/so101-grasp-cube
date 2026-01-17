# LeRobot ACT VAE Encoder - 代码实现指南

## 📚 概览

本文档提供了 LeRobot ACT 模型 VAE encoder 输入处理的完整 Python 实现示例，以及常见问题的排查方法。

---

## 第一部分: 完整实现示例

### 1. 最小化测试（验证基础功能）

```python
#!/usr/bin/env python3
"""最小化测试：验证 VAE encoder 输入形状"""

import torch
import torch.nn as nn
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


def test_vae_encoder_shapes():
    """测试 VAE encoder 对输入形状的要求"""
    
    # ✅ 标准配置
    config = ACTConfig(
        n_obs_steps=1,           # ⭐ ACT 仅支持 1
        n_action_steps=8,        # 预测 8 步动作
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),  # 图像形状
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(15,),  # 状态维度
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(6,),  # 动作维度
            ),
        },
    )
    
    model = ACTPolicy(config)
    model = model.cuda()
    model.eval()
    
    # ✅ 创建输入批次
    batch_size = 4
    n_obs_steps = 1
    
    batch = {
        "observation.images.front": torch.randn(
            batch_size, n_obs_steps, 3, 480, 640, 
            dtype=torch.float32
        ).cuda(),  # (4, 1, 3, 480, 640)
        "observation.state": torch.randn(
            batch_size, n_obs_steps, 15, 
            dtype=torch.float32
        ).cuda(),  # (4, 1, 15)
        "action": torch.randn(
            batch_size, 8, 6, 
            dtype=torch.float32
        ).cuda(),  # (4, 8, 6)
    }
    
    print("=" * 70)
    print("Input Batch Shapes:")
    print("=" * 70)
    for key, value in batch.items():
        print(f"  {key:40s} {str(value.shape):20s}")
    
    # ✅ 测试 forward pass
    try:
        print("\nCalling model.forward()...")
        with torch.no_grad():
            output = model(batch)
        print("✅ Forward pass succeeded!")
        print(f"Output type: {type(output)}")
        if isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                print(f"  output[{i}]: {o.shape}")
        else:
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_vae_encoder_shapes()
    exit(0 if success else 1)
```

### 2. 完整的数据加载管道

```python
#!/usr/bin/env python3
"""完整的数据加载管道，正确处理 VAE encoder 输入"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
from typing import Dict, Optional


class ACTDatasetSimple(Dataset):
    """简化的 ACT 数据集，演示正确的形状处理"""
    
    def __init__(
        self,
        num_episodes: int = 10,
        horizon: int = 16,
        n_obs_steps: int = 1,
        episode_length: int = 100,
    ):
        """
        创建模拟数据集
        
        Args:
            num_episodes: 总 episode 数
            horizon: 动作预测步数
            n_obs_steps: 观测时间步数（ACT 仅支持 1）
            episode_length: 每个 episode 的长度
        """
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.episode_length = episode_length
        
        # 假设数据维度
        self.image_shape = (3, 480, 640)
        self.state_dim = 15
        self.action_dim = 6
    
    def __len__(self):
        """返回总样本数"""
        # 每个 episode 有 (episode_length - horizon) 个有效样本
        samples_per_episode = max(self.episode_length - self.horizon, 1)
        return self.num_episodes * samples_per_episode
    
    def __getitem__(self, idx):
        """
        返回单个数据样本
        
        Returns:
            {
                "observation": {
                    "images": (n_obs_steps, C, H, W),
                    "states": (n_obs_steps, state_dim),
                },
                "action": (horizon, action_dim),
            }
        """
        # ✅ 关键：保持 n_obs_steps 维度
        observation = {
            # 图像：(n_obs_steps, 3, 480, 640)
            "images": np.random.randn(
                self.n_obs_steps, *self.image_shape
            ).astype(np.float32),
            # 状态：(n_obs_steps, state_dim)
            "states": np.random.randn(
                self.n_obs_steps, self.state_dim
            ).astype(np.float32),
        }
        
        # 动作：(horizon, action_dim)
        action = np.random.randn(
            self.horizon, self.action_dim
        ).astype(np.float32)
        
        return {
            "observation": observation,
            "action": action,
        }


def custom_collate_fn(batch):
    """
    自定义 collate 函数
    
    ✅ 重要：正确处理形状
       - 输入中的图像：(T, C, H, W)
       - 输出中的图像：(B, T, C, H, W)
    """
    batch_images = []
    batch_states = []
    batch_actions = []
    
    for item in batch:
        observation = item["observation"]
        # 每个样本的图像：(n_obs_steps, C, H, W)
        batch_images.append(observation["images"])
        # 每个样本的状态：(n_obs_steps, state_dim)
        batch_states.append(observation["states"])
        # 每个样本的动作：(horizon, action_dim)
        batch_actions.append(item["action"])
    
    # ✅ Stack 以添加 batch 维度
    # 结果：(B, n_obs_steps, C, H, W)
    images = torch.from_numpy(np.stack(batch_images, axis=0)).float()
    # 结果：(B, n_obs_steps, state_dim)
    states = torch.from_numpy(np.stack(batch_states, axis=0)).float()
    # 结果：(B, horizon, action_dim)
    actions = torch.from_numpy(np.stack(batch_actions, axis=0)).float()
    
    return {
        "observation": {
            "images": images,
            "states": states,
        },
        "action": actions,
    }


def demonstrate_data_pipeline():
    """演示完整的数据加载管道"""
    
    print("=" * 70)
    print("ACT Data Pipeline Demonstration")
    print("=" * 70)
    
    # 1. 创建数据集
    print("\n1. Creating dataset...")
    dataset = ACTDatasetSimple(
        num_episodes=2,
        horizon=8,
        n_obs_steps=1,
        episode_length=50,
    )
    print(f"   Dataset size: {len(dataset)} samples")
    
    # 2. 获取单个样本
    print("\n2. Single sample shapes:")
    sample = dataset[0]
    print(f"   images: {sample['observation']['images'].shape}")  # (1, 3, 480, 640)
    print(f"   states: {sample['observation']['states'].shape}")  # (1, 15)
    print(f"   action: {sample['action'].shape}")                 # (8, 6)
    
    # 3. 创建 DataLoader
    print("\n3. Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,  # 设为 0 用于演示
    )
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # 4. 从 DataLoader 获取批次
    print("\n4. Batch shapes from DataLoader:")
    batch = next(iter(dataloader))
    images = batch["observation"]["images"]
    states = batch["observation"]["states"]
    actions = batch["action"]
    
    print(f"   images: {images.shape}")  # (4, 1, 3, 480, 640)
    print(f"   states: {states.shape}")  # (4, 1, 15)
    print(f"   action: {actions.shape}")  # (4, 8, 6)
    
    # 5. 展平图像用于 VAE encoder
    print("\n5. Flattening images for VAE encoder:")
    B, T, C, H, W = images.shape
    images_for_vae = images.reshape(B * T, C, H, W)
    print(f"   images for VAE: {images_for_vae.shape}")  # (4, 3, 480, 640)
    
    # 6. 展平状态
    print("\n6. Flattening states:")
    states_for_vae = states.reshape(B * T, -1)
    print(f"   states for VAE: {states_for_vae.shape}")  # (4, 15)
    
    # 7. 模拟 VAE encoder
    print("\n7. Simulating VAE encoder:")
    vae_encoder = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 480 * 640, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # latent_dim = 128
    )
    image_features = vae_encoder(images_for_vae)
    print(f"   image_features: {image_features.shape}")  # (4, 128)
    
    # 8. 恢复时间维度
    print("\n8. Restoring time dimension:")
    image_features_with_time = image_features.reshape(B, T, -1)
    print(f"   image_features (restored): {image_features_with_time.shape}")  # (4, 1, 128)
    
    # 9. 拼接图像特征和状态
    print("\n9. Concatenating features:")
    # 选项 A：在展平空间拼接
    combined_flat = torch.cat([image_features, states_for_vae], dim=-1)
    print(f"   combined (flat): {combined_flat.shape}")  # (4, 128+15=143)
    
    # 选项 B：保持时间维度拼接
    combined_with_time = torch.cat([image_features_with_time, states], dim=-1)
    print(f"   combined (with time): {combined_with_time.shape}")  # (4, 1, 143)
    
    print("\n" + "=" * 70)
    print("✅ Data pipeline demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_data_pipeline()
```

### 3. 与 ACTPolicy 集成的完整例子

```python
#!/usr/bin/env python3
"""与实际 ACTPolicy 集成的完整例子"""

import torch
from torch.utils.data import DataLoader
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType


class ACTTrainingLoop:
    """ACT 训练循环，正确处理 VAE encoder"""
    
    def __init__(self, device: str = "cuda"):
        """初始化"""
        self.device = torch.device(device)
        
        # 创建配置
        self.config = ACTConfig(
            n_obs_steps=1,
            n_action_steps=8,
            input_features={
                "observation.images.front": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 480, 640),
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(15,),
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(6,),
                ),
            },
        )
        
        # 创建模型
        self.model = ACTPolicy(self.config)
        self.model = self.model.to(self.device)
        self.model.train()
    
    def process_batch(self, batch: dict) -> dict:
        """
        处理批次，正确处理 VAE encoder 输入
        
        Args:
            batch: {
                "observation": {
                    "images": (B, n_obs_steps, 3, 480, 640),
                    "states": (B, n_obs_steps, state_dim),
                },
                "action": (B, horizon, action_dim),
            }
        
        Returns:
            处理后的批次，可直接输入模型
        """
        batch_processed = {}
        
        # ✅ 处理图像
        images = batch["observation"]["images"]  # (B, T, C, H, W)
        B, T, C, H, W = images.shape
        
        # 方法 1: 保持原始形状（让模型内部处理）
        batch_processed["observation.images.front"] = images.to(self.device)
        
        # 方法 2: 展平（如果模型期望展平输入）
        # images_flat = images.reshape(B * T, C, H, W)
        # batch_processed["observation.images.front"] = images_flat.to(self.device)
        
        # ✅ 处理状态
        states = batch["observation"]["states"]  # (B, T, state_dim)
        batch_processed["observation.state"] = states.to(self.device)
        
        # ✅ 处理动作
        actions = batch["action"]  # (B, horizon, action_dim)
        batch_processed["action"] = actions.to(self.device)
        
        return batch_processed
    
    def forward_pass(self, batch: dict):
        """
        执行前向传播
        
        Args:
            batch: 原始批次
        
        Returns:
            loss: 损失值
            output: 模型输出
        """
        # 处理批次
        batch_processed = self.process_batch(batch)
        
        # 前向传播
        with torch.autograd.detect_anomaly():
            output = self.model(batch_processed)
        
        return output
    
    @torch.no_grad()
    def inference(self, batch: dict) -> torch.Tensor:
        """
        推理模式
        
        Args:
            batch: 输入批次
        
        Returns:
            predictions: 预测的动作
        """
        batch_processed = self.process_batch(batch)
        
        self.model.eval()
        predictions = self.model.select_action(batch_processed)
        
        return predictions


def demonstrate_integration():
    """演示与 ACTPolicy 的集成"""
    
    print("=" * 70)
    print("ACT Training Loop Integration")
    print("=" * 70)
    
    # 创建训练循环
    trainer = ACTTrainingLoop(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模拟批次
    batch = {
        "observation": {
            "images": torch.randn(4, 1, 3, 480, 640),  # (B, T, C, H, W)
            "states": torch.randn(4, 1, 15),           # (B, T, state_dim)
        },
        "action": torch.randn(4, 8, 6),                # (B, horizon, action_dim)
    }
    
    print("\nInput batch shapes:")
    print(f"  images: {batch['observation']['images'].shape}")
    print(f"  states: {batch['observation']['states'].shape}")
    print(f"  action: {batch['action'].shape}")
    
    # 前向传播
    print("\nPerforming forward pass...")
    try:
        output = trainer.forward_pass(batch)
        print("✅ Forward pass succeeded!")
        print(f"Output type: {type(output)}")
        if isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                if hasattr(o, 'shape'):
                    print(f"  output[{i}]: {o.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_integration()
```

---

## 第二部分: 常见错误排查

### 错误 1: "Tensors must have same number of dimensions"

**症状**：
```python
RuntimeError: Tensors must have same number of dimensions: got 3 and 4
```

**诊断**：

```python
# ❌ 问题代码
images = torch.randn(32, 1, 3, 480, 640)  # 5D
states = torch.randn(32, 15)               # 2D (缺少时间维度)

# VAE encoder 可能输出 4D 或 3D
image_features = vae_encoder(images)  # 返回 4D (32, 1, 128, ...)? 或 3D?

# torch.cat([image_features, states], dim=-1)  # ❌ 维度不匹配
```

**调试步骤**：

```python
# 1. 检查所有输入的维度
print(f"images shape: {images.ndim}D")
print(f"states shape: {states.ndim}D")

# 2. 检查 VAE encoder 输出的维度
test_input = torch.randn(1, 3, 480, 640)
test_output = vae_encoder(test_input)
print(f"VAE output shape: {test_output.shape}")
print(f"VAE output ndim: {test_output.ndim}D")

# 3. 检查 torch.cat 的操作
if image_features.ndim != states.ndim:
    print(f"ERROR: Dimension mismatch!")
    print(f"  image_features: {image_features.shape}")
    print(f"  states: {states.shape}")
```

**解决方案**：

```python
# ✅ 正确的方法
images = torch.randn(B, T, 3, 480, 640)  # 5D
states = torch.randn(B, T, state_dim)     # 3D

# 展平
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)
states_flat = states.reshape(B * T, -1)

# 通过 VAE
image_features = vae_encoder(images_flat)  # (B*T, latent_dim)

# 拼接（现在维度一致）
combined = torch.cat([image_features, states_flat], dim=-1)  # ✅
```

### 错误 2: "Expected input size XXX, got YYY"

**症状**：
```python
RuntimeError: Expected input size (720, 3, 480, 640), got torch.Size([720, 1, 3, 480, 640])
```

**原因**：模型期望 4D 输入，但得到了 5D

**调试**：

```python
# 检查期望的输入形状
print(f"Model expects: (B, C, H, W)")
print(f"But got: {images.shape}")

# 如果得到 (B, T, C, H, W)，需要展平
if images.ndim == 5:
    B, T, C, H, W = images.shape
    images = images.reshape(B * T, C, H, W)
    print(f"Reshaped to: {images.shape}")
```

**解决方案**：

```python
# ✅ 在输入模型前展平
B, T, C, H, W = batch["observation.images.front"].shape
batch["observation.images.front"] = batch["observation.images.front"].reshape(B * T, C, H, W)

# 现在符合模型期望
```

### 错误 3: 形状恢复失败

**症状**：
```python
# 处理后的形状无法恢复到原始形状
image_features_flat = vae_encoder(images_flat)  # (64, 128)
# 如何恢复到 (32, 2, 128)?
```

**调试**：

```python
B, T = 32, 2
original_shape_info = (B, T)  # ✅ 保存原始形状信息

# 处理
image_features_flat = vae_encoder(images_flat)  # (64, 128)

# 恢复
B_saved, T_saved = original_shape_info
image_features = image_features_flat.reshape(B_saved, T_saved, -1)  # (32, 2, 128) ✅
```

**完整示例**：

```python
class VAEEncoderWrapper:
    """VAE Encoder 包装器，自动处理形状变换"""
    
    def __init__(self, vae_encoder):
        self.vae_encoder = vae_encoder
    
    def encode(self, images):
        """
        Args:
            images: (B, T, C, H, W) 或 (B, C, H, W)
        
        Returns:
            features: (B, T, latent_dim) 或 (B, latent_dim)
        """
        if images.ndim == 5:
            # 保存原始形状
            B, T, C, H, W = images.shape
            shape_info = (B, T)
            
            # 展平
            images_flat = images.reshape(B * T, C, H, W)
        else:
            shape_info = None
            images_flat = images
        
        # 通过 VAE encoder
        features_flat = self.vae_encoder(images_flat)
        
        # 恢复形状
        if shape_info is not None:
            B_saved, T_saved = shape_info
            features = features_flat.reshape(B_saved, T_saved, -1)
        else:
            features = features_flat
        
        return features


# 使用
vae_encoder = VAEEncoderWrapper(original_vae_encoder)
image_features = vae_encoder.encode(batch["observation.images.front"])
```

### 错误 4: DataLoader 的 collate_fn 问题

**症状**：
```python
# 从 DataLoader 得到的形状与预期不符
for batch in dataloader:
    print(batch["observation"]["images"].shape)  # 可能是 (B, C, H, W) 而不是 (B, T, C, H, W)
```

**调试**：

```python
# 检查单个样本的形状
sample = dataset[0]
print(f"Single sample images shape: {sample['observation']['images'].shape}")

# 检查 collate 后的形状
batch = custom_collate_fn([dataset[0], dataset[1], dataset[2]])
print(f"Batch images shape: {batch['observation']['images'].shape}")

# 是否丢失了时间维度？
if batch["observation"]["images"].ndim == 4:
    print("ERROR: Time dimension was lost!")
```

**解决方案**：

```python
# ✅ 正确的 collate_fn
def correct_collate_fn(batch):
    images_list = []
    for item in batch:
        # 每个 item 的 images 应该是 (T, C, H, W)
        images = item["observation"]["images"]
        images_list.append(images)
    
    # Stack 得到 (B, T, C, H, W)
    images = torch.stack(images_list, dim=0)
    
    # 检查
    assert images.ndim == 5, f"Expected 5D, got {images.ndim}D"
    
    return {"observation": {"images": images}, ...}
```

---

## 第三部分: 验证清单

### 部署前检查清单

```python
def verify_input_shapes(model, batch):
    """验证输入形状是否正确"""
    
    checks = {
        "images_is_5d": False,
        "states_is_3d": False,
        "images_ndim": None,
        "states_ndim": None,
        "shapes_match": False,
        "vae_encoder_compatible": False,
    }
    
    # 检查 1: 图像是否为 5D
    images = batch["observation"]["images"]
    checks["images_ndim"] = images.ndim
    checks["images_is_5d"] = images.ndim == 5
    if not checks["images_is_5d"]:
        print(f"⚠️  WARNING: Images should be 5D, got {images.ndim}D")
    
    # 检查 2: 状态是否为 3D
    states = batch["observation"]["states"]
    checks["states_ndim"] = states.ndim
    checks["states_is_3d"] = states.ndim == 3
    if not checks["states_is_3d"]:
        print(f"⚠️  WARNING: States should be 3D, got {states.ndim}D")
    
    # 检查 3: Batch size 是否一致
    if images.shape[0] == states.shape[0]:
        checks["shapes_match"] = True
    else:
        print(f"❌ ERROR: Batch sizes don't match: {images.shape[0]} vs {states.shape[0]}")
    
    # 检查 4: n_obs_steps 是否一致
    if images.shape[1] == states.shape[1]:
        checks["n_obs_steps_match"] = True
    else:
        print(f"❌ ERROR: n_obs_steps don't match: {images.shape[1]} vs {states.shape[1]}")
    
    # 检查 5: VAE encoder 兼容性
    B, T, C, H, W = images.shape
    images_flat = images.reshape(B * T, C, H, W)
    checks["vae_encoder_compatible"] = images_flat.ndim == 4
    if not checks["vae_encoder_compatible"]:
        print(f"❌ ERROR: Flattened images should be 4D, got {images_flat.ndim}D")
    
    # 总体检查
    all_passed = all(checks.values())
    status = "✅ PASS" if all_passed else "❌ FAIL"
    print(f"\n{status} - Shape verification summary:")
    for check, result in checks.items():
        symbol = "✅" if result else "❌" if isinstance(result, bool) else "ℹ️"
        print(f"  {symbol} {check}: {result}")
    
    return all_passed


# 使用
batch = next(iter(dataloader))
verify_input_shapes(model, batch)
```

---

## 快速参考卡

### ✅ 正确的形状变换流程

```
数据集输出:
  images: (T, C, H, W)
  states: (T, state_dim)
  
        ↓ collate (stack)
  
DataLoader 输出:
  images: (B, T, C, H, W)  ← ✅ 5D
  states: (B, T, state_dim)  ← ✅ 3D
  
        ↓ reshape (B*T)
  
VAE Encoder 输入:
  images: (B*T, C, H, W)  ← ✅ 4D
  states: (B*T, state_dim)  ← ✅ 2D
  
        ↓ encode + reshape
  
模型输入:
  image_features: (B, T, latent_dim)  ← ✅ 3D
  states: (B, T, state_dim)             ← ✅ 3D
  
        ↓ cat
  
Transformer 输入:
  combined: (B, T, latent_dim + state_dim)  ← ✅ 3D
```

---

**最后更新**：2026-01-17

