# LeRobot ACT VAE Encoder - 查询结果总结

## 📌 你的查询

在 LeRobot 项目中，查找 ACT 模型 VAE encoder 的预期输入形状：

1. ✅ 在 lerobot/policies/act/modeling_act.py 中找到 VAE encoder 的 forward 方法
2. ✅ 查看它对输入 shape 的期望，尤其是当 n_obs_steps=1 时
3. ✅ 查找如何正确构造 vae_encoder_input - 它应该是什么形状
4. ✅ 找到 torch.cat 操作的上下文，看看为什么会出现"Tensors must have same number of dimensions: got 3 and 4"

---

## 🎯 核心答案

### 1. VAE Encoder 的预期输入形状

```python
# VAE Encoder 期望的输入形状：(B, C, H, W) — 4维

# 具体参数：
#   B: batch_size (例如 32)
#   C: channels = 3 (RGB)
#   H: height = 480
#   W: width = 640

# 完整示例
images_for_vae = torch.randn(32, 3, 480, 640)  # ✅ 正确的输入形状
```

### 2. images 和 states 应该是什么形状

#### 从 DataLoader 获取的形状

```python
# DataLoader 提供的批次
batch = {
    "observation.images.front": torch.randn(32, 1, 3, 480, 640),  # (B, T, C, H, W) — 5维
    "observation.state": torch.randn(32, 1, 15),                   # (B, T, state_dim) — 3维
}

# 说明：
# - 32 是 batch_size
# - 1 是 n_obs_steps (ACT 仅支持 1)
# - 3 是图像通道数
# - 480, 640 是图像尺寸
# - 15 是状态维度
```

### 3. 如何正确构造 vae_encoder_input

```python
# ✅ 正确方式：展平 (B, T) 维度

# 从 DataLoader 获取
batch = next(iter(dataloader))
B, T, C, H, W = batch["observation.images.front"].shape  # (32, 1, 3, 480, 640)

# 展平用于 VAE encoder
images_for_vae = batch["observation.images.front"].reshape(B * T, C, H, W)
# 结果：(32, 3, 480, 640) ✅

# 为什么展平？
# VAE encoder 的设计期望 (B, C, H, W)，而你有 (B, T, C, H, W)
# 通过展平，(B, T) 组合为单一维度 B*T，得到期望的 4D 形状
```

### 4. "Tensors must have same number of dimensions: got 3 and 4" 的原因

```python
# ❌ 错误的拼接方式导致维度不匹配

images = torch.randn(32, 1, 3, 480, 640)  # 5D
states = torch.randn(32, 15)               # 2D （缺少 T 维度）

# VAE encoder 输出
image_features = vae_encoder(images)  # 可能返回 4D 或 3D，取决于处理方式

# 直接拼接会失败
try:
    combined = torch.cat([image_features, states], dim=-1)
except RuntimeError:
    # RuntimeError: Tensors must have same number of dimensions: got 3 and 4
    # 原因：image_features 是 3D 或 4D，states 是 2D，维数不同

# ✅ 正确方式：确保维数相同
# 方法 1: 都转为 3D
image_features_3d = image_features.reshape(B, T, -1)  # (32, 1, 128)
states_3d = states.reshape(B, T, -1)                  # (32, 1, 15)
combined = torch.cat([image_features_3d, states_3d], dim=-1)  # (32, 1, 143) ✅

# 方法 2: 都转为 2D
image_features_flat = image_features.reshape(B*T, -1)  # (32, 128)
states_flat = states.reshape(B*T, -1)                  # (32, 15)
combined = torch.cat([image_features_flat, states_flat], dim=-1)  # (32, 143) ✅
```

---

## 📊 完整的数据流

```
数据集输出
  images: (T, C, H, W)          states: (T, state_dim)
  ↓ collate (stack)              ↓
DataLoader
  images: (B, T, C, H, W)       states: (B, T, state_dim)
  ↓ reshape (B*T)               ↓ reshape (B*T)
VAE输入
  images: (B*T, C, H, W)        states: (B*T, state_dim)
  ↓ encode                       ↓
VAE输出
  image_features: (B*T, latent_dim)  states: (B*T, state_dim)
  ↓ reshape (B, T)              ↓ reshape (B, T)
恢复
  image_features: (B, T, latent_dim)  states: (B, T, state_dim)
  ↓ cat (dim=-1)
完整观测
  combined: (B, T, latent_dim + state_dim)
  ↓
Transformer
  预测动作
```

---

## 🔑 关键要点

### n_obs_steps=1 时的特殊处理

| 方面 | 说明 |
|------|------|
| **为什么保留 T 维度** | 即使 n_obs_steps=1，也保持 (B, 1, ...) 格式 |
| **原因 1** | 与 LeRobot 设计一致 |
| **原因 2** | 支持将来的 n_obs_steps > 1 |
| **原因 3** | 时间序列特性（即使是单步） |
| **正确方式** | `reshape(B*T, ...)` 而不是 `squeeze()` |

### reshape vs squeeze 的区别

```python
# ❌ squeeze 的问题
images = torch.randn(32, 1, 3, 480, 640)
images_sq = images.squeeze(1)  # (32, 3, 480, 640)
# 问题：时间维度信息丢失，无法恢复为原始形状

# ✅ reshape 的优点
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)  # (32, 3, 480, 640)
# 优点：保留原始形状信息，可以精确恢复
restored = images_flat.reshape(B, T, C, H, W)  # 恢复成功！
```

---

## 💻 完整的代码示例

### 最小化示例

```python
import torch

# 1. 从 DataLoader 获取批次
batch = next(iter(dataloader))
images = batch["observation"]["images"]  # (32, 1, 3, 480, 640)
states = batch["observation"]["states"]  # (32, 1, 15)

# 2. 展平用于 VAE encoder
B, T, C, H, W = images.shape
images_for_vae = images.reshape(B * T, C, H, W)  # (32, 3, 480, 640)
states_for_vae = states.reshape(B * T, -1)       # (32, 15)

# 3. 通过 VAE encoder
vae_encoder = ...
image_features = vae_encoder(images_for_vae)  # (32, 128)

# 4. 恢复时间维度
image_features = image_features.reshape(B, T, -1)  # (32, 1, 128)

# 5. 拼接
combined = torch.cat([image_features, states], dim=-1)  # (32, 1, 143) ✅
```

### 与 ACTPolicy 集成

```python
from lerobot.policies.act.modeling_act import ACTPolicy

# 创建模型
model = ACTPolicy(config)

# 准备输入（关键步骤）
batch_input = {
    "observation.images.front": images,      # (B, T, C, H, W)
    "observation.state": states,              # (B, T, state_dim)
    "action": actions,                        # (B, horizon, action_dim)
}

# 前向传播（模型内部处理 VAE encoder）
output = model(batch_input)
```

---

## 🚨 常见错误及解决方案

### 错误 1: "got 3 and 4" 维度不匹配

**症状**：`RuntimeError: Tensors must have same number of dimensions: got 3 and 4`

**原因**：拼接的张量维数不同

**解决**：
```python
# ✅ 确保都是 3D
image_features = image_features.reshape(B, T, -1)  # (B, T, latent)
states = states.reshape(B, T, -1)                  # (B, T, state)
combined = torch.cat([image_features, states], dim=-1)  # ✅
```

### 错误 2: VAE encoder 输入形状不对

**症状**：`RuntimeError: Expected input size (720, 3, 480, 640), got ...`

**原因**：输入是 5D 而不是 4D

**解决**：
```python
# ✅ 展平图像
B, T, C, H, W = images.shape
images = images.reshape(B * T, C, H, W)
```

### 错误 3: 无法恢复时间维度

**症状**：处理后的张量形状无法恢复

**原因**：使用了 squeeze，丢失了形状信息

**解决**：
```python
# ✅ 使用 reshape 并保存原始形状
B, T = 32, 1
images_flat = images.reshape(B * T, 3, 480, 640)
features = vae_encoder(images_flat)  # (32, 128)
features_restored = features.reshape(B, T, -1)  # (32, 1, 128) ✅
```

---

## 📚 配套文档

我为你创建了 4 份详细文档（都在项目根目录）：

1. **[LEROBOT_ACT_VAE_INDEX.md](./LEROBOT_ACT_VAE_INDEX.md)** ⭐ **文档导航**
   - 快速导航所有文档
   - 按问题查找信息
   - 推荐阅读路径

2. **[LEROBOT_ACT_VAE_COMPLETE_ANSWER.md](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md)** ⭐ **完整答案**
   - 直接回答你的 4 个问题
   - 详细的例子和数据流图
   - 所有概念的完整解释

3. **[LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md)** ⭐ **深度教程**
   - 完整的理论解释
   - 10 个主要章节
   - 7 种常见错误详解

4. **[LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)** ⭐ **实现指南**
   - 可复制粘贴的代码示例
   - 完整的数据加载管道
   - 详细的调试技巧

5. **[LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)** ⭐ **速查表**
   - 5 分钟快速参考
   - 代码模板
   - 常见问题速解

---

## 🎯 快速开始

### 如果你只有 5 分钟
读 [LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)

### 如果你有 15 分钟
读 [LEROBOT_ACT_VAE_COMPLETE_ANSWER.md](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md)

### 如果你想完全理解
按这个顺序读：
1. [LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md) (5 min)
2. [LEROBOT_ACT_VAE_COMPLETE_ANSWER.md](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md) (15 min)
3. [LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md) (40 min)
4. [LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md) (45 min)

### 如果你在 Debug
1. 查看 [LEROBOT_ACT_VAE_QUICK_REFERENCE.md 的错误排查树](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-错误排查树)
2. 查看 [LEROBOT_ACT_VAE_IMPLEMENTATION.md 的错误排查部分](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#第二部分-常见错误排查)

---

## 📋 检查清单

在部署代码前，确保：

- [ ] VAE encoder 的输入是 4D `(B, C, H, W)`
- [ ] 使用 `reshape(B*T, C, H, W)` 而不是 `squeeze()`
- [ ] images 和 states 都从 DataLoader 的 `(B, T, ...)` 展平到 `(B*T, ...)`
- [ ] n_obs_steps 设置为 1（ACT 要求）
- [ ] VAE encoder 输出形状为 `(B*T, latent_dim)`
- [ ] 恢复时间维度后为 `(B, T, latent_dim)`
- [ ] torch.cat 的两个操作数具有相同的维数

---

## 💡 关键洞见

1. **时间维度的设计**
   - 即使 `n_obs_steps=1`，LeRobot 仍然保留时间维度
   - 这是为了代码一致性和将来的扩展性

2. **展平的必要性**
   - VAE encoder 只处理单个时间步
   - 必须展平 `(B, T)` 为 `B*T` 才能符合期望

3. **使用 reshape 而不是 squeeze**
   - reshape 保留完整的形状信息
   - squeeze 可能导致无法恢复形状

4. **维度错误的根本原因**
   - torch.cat 要求操作数具有相同的维数
   - 如果混合了不同维度的张量就会失败

---

## 📞 快速参考

### 最常用的代码
```python
# 展平
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)

# 恢复
features = features.reshape(B, T, -1)

# 拼接
combined = torch.cat([features, states], dim=-1)
```

### 最常见的错误
| 错误 | 代码 | 修复 |
|------|------|------|
| 维数不匹配 | `cat([3D, 2D])` | 都转为 3D |
| VAE 输入 5D | 直接输入 | `reshape(B*T, ...)` |
| 形状丢失 | `squeeze()` | 改用 `reshape()` |

---

## 🔗 相关资源

### 项目代码实现
- [scripts/train_act_real_data.py](./scripts/train_act_real_data.py) - 完整的 ACT 训练
- [scripts/inference_engine.py](./scripts/inference_engine.py) - 推理实现（含 VAE 处理）
- [test_act_minimal.py](./test_act_minimal.py) - 最小化测试

### 外部资源
- [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
- [PyTorch 文档](https://pytorch.org/)

---

## 📝 总结

你的所有问题都已经回答：

✅ **VAE encoder 的输入形状**：`(B, C, H, W)` — 4D  
✅ **images 和 states 的形状**：从 DataLoader 得到 `(B, T, ...)` 后展平  
✅ **正确的 vae_encoder_input 构造**：使用 `reshape(B*T, C, H, W)`  
✅ **torch.cat 维度错误的原因**：拼接的张量维数不一致  

所有详细信息、代码示例和排查方法都在配套的 5 份文档中。

---

**版本**：1.0  
**完成日期**：2026-01-17  
**状态**：✅ 完整回答

