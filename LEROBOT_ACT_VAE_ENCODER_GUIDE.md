# LeRobot ACT 模型 VAE Encoder 输入形状详解

## 📌 概述

本指南详细说明了 LeRobot ACT 模型中 VAE encoder 的预期输入形状，以及如何正确处理 `n_obs_steps` 维度。这对于避免 `"Tensors must have same number of dimensions: got 3 and 4"` 错误至关重要。

---

## 1️⃣ VAE Encoder 的基本概念

### VAE Encoder 的作用
- 将高维图像压缩成低维潜在向量
- 输入：原始图像数据
- 输出：潜在向量，用于与状态向量拼接

### LeRobot ACT 架构中的位置
```
输入 (Images + States)
    ↓
[VAE Encoder] ← 处理图像
    ↓
潜在向量 + 状态 → [Concatenation] → Transformer → 输出动作
```

---

## 2️⃣ VAE Encoder 的预期输入形状

### 完整形状要求

| 维度 | 含义 | 形状 | 说明 |
|------|------|------|------|
| **Batch** | 批次大小 | `B` | 例如 32 |
| **Time Steps** | 观测时间步 | `n_obs_steps` | **关键：ACT 仅支持 1** |
| **Channels** | 图像通道 | `C=3` | RGB 图像 |
| **Height** | 图像高度 | `H` | 例如 480 |
| **Width** | 图像宽度 | `W` | 例如 640 |

### 标准输入形状

```python
# 标准输入形状（n_obs_steps=1 时）
images.shape = (B, n_obs_steps, C, H, W)
                = (32, 1, 3, 480, 640)

# VAE Encoder 期望的形状
# Option 1: 如果 encoder 处理单个时间步
images_squeezed.shape = (B, C, H, W)
                       = (32, 3, 480, 640)

# Option 2: 如果需要保留时间维度（ACT的做法）
# encoder 应该重新形状为 (B*T, C, H, W)
images_flattened.shape = (B*T, C, H, W)
                        = (32, 3, 480, 640)  # 当 n_obs_steps=1
```

---

## 3️⃣ 正确的维度处理方式

### ✅ 推荐方案：在 VAE Encoder 前展平

这是 LeRobot ACT 采用的标准方式：

```python
import torch

# 从 DataLoader 收到的数据
batch = {
    "observation.images.front": torch.randn(B, n_obs_steps, C, H, W),  # (32, 1, 3, 480, 640)
    "observation.state": torch.randn(B, n_obs_steps, state_dim),        # (32, 1, 15)
}

# 关键步骤：展平 batch 和 time 维度
B, T, C, H, W = batch["observation.images.front"].shape
images_for_encoder = batch["observation.images.front"].reshape(B * T, C, H, W)
# 结果：(32, 3, 480, 640) ✅ 符合 VAE Encoder 期望

# 传递给 VAE Encoder
vae_encoder_input = images_for_encoder  # (B*T, C, H, W)
image_features = vae_encoder(vae_encoder_input)
# 输出：(B*T, latent_dim) 例如 (32, 128)

# 重新 reshape 回时间维度
image_features = image_features.reshape(B, T, -1)  # (32, 1, 128)
```

### ❌ 错误做法 1: Squeeze 导致维度不匹配

```python
# ❌ 错误：squeeze 后失去了 time 维度信息
images_squeezed = batch["observation.images.front"].squeeze(1)  # (32, 3, 480, 640) ✅ 形状对了
states_squeezed = batch["observation.states"].squeeze(1)         # (32, 15) ✅ 形状对了

# 但后续的 torch.cat 会出错
# 如果 image_encoder 输出 (B, latent_dim) = (32, 128)
# 而 states 是 (B, state_dim) = (32, 15)
# 那么 torch.cat([image_features, states], dim=-1) 是可以的
# 但这违反了 ACT 对时间维度的期望

# 问题：如果有多个时间步（n_obs_steps > 1），squeeze 会丢失信息
```

### ❌ 错误做法 2: 直接 Concatenate 高维张量

```python
# ❌ 错误：直接拼接 (B, T, 3, H, W) 和 (B, T, state_dim) 会失败
images = torch.randn(B, T, 3, H, W)  # (32, 1, 3, 480, 640) - 4D for image
states = torch.randn(B, T, state_dim)  # (32, 1, 15) - 3D for state

# torch.cat([images, states], dim=-1) → RuntimeError: 
# "Tensors must have same number of dimensions: got 3 and 4"
```

---

## 4️⃣ 完整的数据流示例

### 从 DataLoader 到 VAE Encoder 的完整流程

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === 1. DataLoader 提供的批次 ===
batch = {
    "observation.images.front": torch.randn(32, 1, 3, 480, 640),  # (B, T, C, H, W)
    "observation.state": torch.randn(32, 1, 15),                  # (B, T, state_dim)
    "action": torch.randn(32, 8, 6),                              # (B, action_steps, action_dim)
}

# === 2. 提取并展平图像 ===
B, T, C, H, W = batch["observation.images.front"].shape
# B=32, T=1, C=3, H=480, W=640

images = batch["observation.images.front"]  # (32, 1, 3, 480, 640)
images_flat = images.reshape(B * T, C, H, W)  # (32, 3, 480, 640) ✅

# === 3. 通过 VAE Encoder ===
vae_encoder = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 128),  # 输出 latent_dim=128
)

image_features = vae_encoder(images_flat)  # (32, 128)
image_features = image_features.reshape(B, T, -1)  # (32, 1, 128) ✅

# === 4. 提取并处理状态 ===
states = batch["observation.state"]  # (32, 1, 15)
states_flat = states.reshape(B * T, -1)  # (32, 15)

# === 5. Concatenate (正确的方式) ===
# 方法 A: 在展平空间拼接
combined = torch.cat([image_features, states_flat], dim=-1)  # (32, 128+15=143) ✅

# 方法 B: 如果需要保持时间维度
combined = torch.cat([image_features, states], dim=-1)  # (32, 1, 128+15=143) ✅

# === 6. 后续处理 ===
# 如果是展平的，可以直接送入 MLP
output = some_mlp(combined)  # (32, output_dim)

# 如果保持了时间维度，需要展平或进一步处理
# output = some_mlp(combined.reshape(B*T, -1))  # (32, output_dim)
```

---

## 5️⃣ n_obs_steps 的特殊处理

### 当 n_obs_steps = 1 时（ACT 的标准配置）

```python
# ✅ 推荐：使用 reshape
n_obs_steps = 1
batch_size = 32
images = torch.randn(batch_size, n_obs_steps, 3, 480, 640)  # (32, 1, 3, 480, 640)

# 方法 1: reshape 展平
images_for_vae = images.reshape(batch_size * n_obs_steps, 3, 480, 640)  # (32, 3, 480, 640)
image_features = vae_encoder(images_for_vae)  # (32, 128)

# 方法 2: squeeze（仅当 n_obs_steps=1 时安全）
images_squeezed = images.squeeze(1)  # (32, 3, 480, 640)
image_features = vae_encoder(images_squeezed)  # (32, 128)

# ✅ reshape 是更通用的，支持 n_obs_steps > 1
```

### 当 n_obs_steps > 1 时（理论支持，但 ACT 通常不用）

```python
# 假设 n_obs_steps = 2
n_obs_steps = 2
batch_size = 32
images = torch.randn(batch_size, n_obs_steps, 3, 480, 640)  # (32, 2, 3, 480, 640)

# ✅ 必须使用 reshape
images_for_vae = images.reshape(batch_size * n_obs_steps, 3, 480, 640)  # (64, 3, 480, 640)
image_features = vae_encoder(images_for_vae)  # (64, 128)

# ✅ 恢复时间维度
image_features = image_features.reshape(batch_size, n_obs_steps, -1)  # (32, 2, 128)

# ❌ squeeze 会直接删除 time 维度，导致形状丢失：
images_squeezed = images.squeeze(1)  # ❌ 这会删除第 2 个时间步！
```

---

## 6️⃣ 完整的 PyTorch 数据加载和处理

### DataLoader 配置

```python
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class RealDataACTDataset(Dataset):
    """ACT 真机数据集"""
    
    def __init__(self, task_dir, horizon=16, n_obs_steps=1):
        self.task_dir = task_dir
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        # ... 加载数据 ...
    
    def __getitem__(self, idx):
        # 返回观测和动作
        # ✅ 关键：保持 n_obs_steps 维度
        return {
            "observation": {
                "images": np.zeros((self.n_obs_steps, 3, 480, 640)),  # (T, C, H, W)
                "states": np.zeros((self.n_obs_steps, 15)),            # (T, state_dim)
            },
            "action": np.zeros((self.horizon, 6)),  # (action_steps, action_dim)
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    observations = {"images": [], "states": []}
    actions = []
    
    for item in batch:
        obs = item["observation"]
        observations["images"].append(obs["images"])
        observations["states"].append(obs["states"])
        actions.append(item["action"])
    
    # ✅ 关键：在 collate 时堆叠，得到 (B, T, C, H, W)
    return {
        "observation": {
            "images": torch.from_numpy(np.stack(observations["images"], axis=0)).float(),
            # 结果：(B, n_obs_steps, 3, H, W) = (32, 1, 3, 480, 640)
            "states": torch.from_numpy(np.stack(observations["states"], axis=0)).float(),
            # 结果：(B, n_obs_steps, state_dim) = (32, 1, 15)
        },
        "action": torch.from_numpy(np.stack(actions, axis=0)).float(),
        # 结果：(B, horizon, action_dim) = (32, 8, 6)
    }


# 创建 DataLoader
dataset = RealDataACTDataset(task_dir="real_data/lift")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

# 从 DataLoader 获取批次
for batch in dataloader:
    images = batch["observation"]["images"]  # (32, 1, 3, 480, 640)
    states = batch["observation"]["states"]  # (32, 1, 15)
    actions = batch["action"]                # (32, 8, 6)
    
    # ✅ 现在可以安全地传递给模型
    # model(batch)
    break
```

### 模型前向传播

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig

class ACTPolicyWrapper:
    """ACT 政策包装器，正确处理维度"""
    
    def __init__(self, config: ACTConfig):
        self.model = ACTPolicy(config)
        self.model.eval()
    
    def forward(self, batch):
        """
        Args:
            batch: {
                "observation.images.front": (B, n_obs_steps, C, H, W),
                "observation.state": (B, n_obs_steps, state_dim),
                "action": (B, action_steps, action_dim),
            }
        """
        # ✅ 展平 images 用于 VAE encoder
        images = batch["observation.images.front"]  # (B, T, C, H, W)
        B, T, C, H, W = images.shape
        images_for_vae = images.reshape(B * T, C, H, W)  # (B*T, C, H, W)
        
        # VAE encoder 处理展平的图像
        # 内部 LeRobot ACT 会处理这个细节
        
        # ✅ 正确的输入格式
        input_dict = {
            "observation.images.front": images,  # (B, T, C, H, W) ✅
            "observation.state": batch["observation.state"],  # (B, T, state_dim) ✅
            "action": batch["action"],  # (B, action_steps, action_dim)
        }
        
        # 调用模型
        output = self.model(input_dict)
        return output
```

---

## 7️⃣ 常见错误及解决方案

### 错误 1: Tensor 维度不匹配

**症状**：`RuntimeError: Tensors must have same number of dimensions: got 3 and 4`

**原因**：
```python
# ❌ 错误：混合不同维度的张量
images = torch.randn(B, T, C, H, W)  # 5D
states = torch.randn(B, T, state_dim)  # 3D

# 在 VAE encoder 输出后直接拼接
image_features = vae_encoder(images)  # 可能返回 (B, T, latent_dim) 3D
# torch.cat([image_features, states], dim=-1)  # ❌ 3D + 3D 可以，但如果返回 4D 就会错
```

**解决方案**：
```python
# ✅ 确保一致的维度
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)
image_features = vae_encoder(images_flat)  # (B*T, latent_dim)
image_features = image_features.reshape(B, T, -1)  # (B, T, latent_dim)

# 现在都是 3D，可以拼接
states = batch["observation.state"]  # (B, T, state_dim)
combined = torch.cat([image_features, states], dim=-1)  # ✅ (B, T, latent_dim + state_dim)
```

### 错误 2: Squeeze 导致维度丢失

**症状**：后续处理时维度不符合预期

**原因**：
```python
# ❌ 错误：squeeze 删除了时间维度
images = torch.randn(32, 1, 3, 480, 640)
images_sq = images.squeeze(1)  # (32, 3, 480, 640) - 丢失了 time 维度 1

# 如果后续有 n_obs_steps > 1 的数据，就会出问题
```

**解决方案**：
```python
# ✅ 明确指定维度
images = torch.randn(32, 1, 3, 480, 640)
B, T, C, H, W = images.shape
images_reshaped = images.reshape(B * T, C, H, W)  # (32, 3, 480, 640)
# 处理后恢复时间维度
images_restored = images_reshaped.reshape(B, T, C, H, W)  # (32, 1, 3, 480, 640)
```

### 错误 3: 批处理中的形状不一致

**症状**：某些批次通过，某些批次失败

**原因**：DataLoader 的 collate_fn 处理不当

**解决方案**：
```python
def collate_fn(batch):
    """确保所有张量有一致的形状"""
    images_list = []
    states_list = []
    actions_list = []
    
    for item in batch:
        # ✅ 每个 item 都应该已经有 (T, C, H, W) 的形状
        images_list.append(item["observation"]["images"])  # (T, C, H, W)
        states_list.append(item["observation"]["states"])  # (T, state_dim)
        actions_list.append(item["action"])  # (horizon, action_dim)
    
    # 堆叠成 (B, T, C, H, W)
    images = torch.stack(images_list, dim=0)  # (B, T, C, H, W) ✅
    states = torch.stack(states_list, dim=0)  # (B, T, state_dim) ✅
    actions = torch.stack(actions_list, dim=0)  # (B, horizon, action_dim) ✅
    
    return {
        "observation": {
            "images": images,
            "states": states,
        },
        "action": actions,
    }
```

---

## 8️⃣ ACT 官方实现参考

### LeRobot ACT 的实际处理方式

基于项目中的 `inference_engine.py` 和 `train_act_real_data.py`：

```python
# 来自 inference_engine.py 的真实代码片段
B, T, C, H, W = batch["observation.images.front"].shape
# 关键修复：展平 image 的 batch 和 time 维度供 rgb_encoder 使用
# rgb_encoder 期望 (B, C, H, W)，但我们有 (B, T, C, H, W)
# 所以展平为 (B*T, C, H, W)，rgb_encoder 会处理它
batch["observation.images.front"] = batch["observation.images.front"].reshape(B * T, C, H, W)

# 注意：这就是正确的做法！
```

### LeRobot 官方 ACTPolicy forward 方法期望

```python
# 官方期望的输入格式
batch = {
    "observation.images.front": Tensor,  # (B, C, H, W) 或 (B*T, C, H, W)
    "observation.state": Tensor,          # (B, state_dim) 或 (B*T, state_dim)
    "action": Tensor,                     # (B, action_steps, action_dim) 或 (B*T, action_dim)
}

# ACT 会在内部处理 VAE encoder，
# 展平的图像会通过 VAE encoder，
# 然后与状态拼接形成完整的 observation embedding
```

---

## 9️⃣ 总结表格

| 场景 | 输入形状 | 处理方式 | 输出形状 | 说明 |
|------|---------|---------|---------|------|
| **从 Dataset** | (T, C, H, W) | - | (T, C, H, W) | 单个样本 |
| **从 Collate** | - | Stack | (B, T, C, H, W) | 批次形成 |
| **传给 VAE** | (B, T, C, H, W) | Reshape | (B*T, C, H, W) | 展平处理 |
| **VAE 输出** | - | - | (B*T, latent) | 潜在向量 |
| **恢复时间维** | (B*T, latent) | Reshape | (B, T, latent) | 恢复结构 |
| **与 State 拼接** | (B, T, latent) + (B, T, state) | Concat | (B, T, latent+state) | 完整观测 |

---

## 🔟 快速参考

### ✅ 正确的代码模板

```python
# 1. 创建批次
batch = next(iter(dataloader))
images = batch["observation"]["images"]  # (B, T, C, H, W)
states = batch["observation"]["states"]  # (B, T, state_dim)

# 2. 展平图像
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)

# 3. 通过 VAE encoder
vae_encoder = ...
image_features = vae_encoder(images_flat)  # (B*T, latent_dim)

# 4. 恢复时间维度
image_features = image_features.reshape(B, T, -1)  # (B, T, latent_dim)

# 5. 展平状态（如果需要）
states_flat = states.reshape(B * T, -1)  # (B*T, state_dim)

# 6. 拼接
combined = torch.cat([image_features.reshape(B*T, -1), states_flat], dim=-1)
# 结果：(B*T, latent_dim + state_dim) ✅
```

### ❌ 常见错误

```python
# ❌ 1. 直接拼接高维张量
torch.cat([images, states], dim=-1)  # 维度不匹配

# ❌ 2. Squeeze 丢失信息
images_sq = images.squeeze(1)  # 丢失时间维度

# ❌ 3. 不一致的形状处理
image_features = vae_encoder(images)  # (B, T, C, H, W) 输入
# 返回形状可能不清楚
```

---

## 参考资源

1. [LeRobot 官方文档](https://github.com/huggingface/lerobot)
2. 项目文件：
   - [train_act_real_data.py](./scripts/train_act_real_data.py) - 完整训练实现
   - [inference_engine.py](./scripts/inference_engine.py) - 推理实现
   - [test_act_minimal.py](./test_act_minimal.py) - 最小化测试
3. 相关文档：
   - [QUICK_START_ACT.md](./scripts/QUICK_START_ACT.md)
   - [README_ACT_TRAINING.md](./scripts/README_ACT_TRAINING.md)

---

**最后更新**：2026-01-17  
**关键点**：使用 `reshape(B*T, C, H, W)` 而不是 `squeeze()` 来处理 VAE encoder 的输入！
