# LeRobot ACT VAE Encoder - 快速参考卡

## 🎯 一句话总结

**使用 `reshape(B*T, C, H, W)` 而不是 `squeeze()` 来处理 VAE encoder 的输入！**

---

## 📋 核心知识点

### VAE Encoder 的输入要求

| 内容 | 要求 |
|------|------|
| **预期输入形状** | `(B, C, H, W)` — 4维 |
| **来自 DataLoader** | `(B, T, C, H, W)` — 5维 |
| **n_obs_steps** | **仅支持 1**（ACT 限制）|
| **处理方式** | `reshape(B*T, C, H, W)` ✅ |
| **错误方式** | `squeeze(1)` ❌ |

---

## ⚡ 快速代码示例

### ✅ 正确做法

```python
# 1. 从 DataLoader 得到批次
batch = {
    "images": torch.randn(32, 1, 3, 480, 640),  # (B, T, C, H, W)
    "states": torch.randn(32, 1, 15),            # (B, T, state_dim)
}

# 2. 展平图像用于 VAE encoder
B, T, C, H, W = batch["images"].shape
images_for_vae = batch["images"].reshape(B * T, C, H, W)  # (32, 3, 480, 640) ✅

# 3. 通过 VAE encoder
image_features = vae_encoder(images_for_vae)  # (32, 128)

# 4. 恢复时间维度
image_features = image_features.reshape(B, T, -1)  # (32, 1, 128) ✅

# 5. 与状态拼接
combined = torch.cat([image_features, batch["states"]], dim=-1)  # (32, 1, 143) ✅
```

### ❌ 常见错误

```python
# ❌ 错误 1: squeeze 丢失信息
images_sq = batch["images"].squeeze(1)  # (32, 3, 480, 640) — 丢失时间维度

# ❌ 错误 2: 直接拼接高维张量
torch.cat([batch["images"], batch["states"]], dim=-1)  # 维度不匹配！

# ❌ 错误 3: 维度不一致
image_features = vae_encoder(batch["images"])  # 期望 4D 输入但得到 5D
```

---

## 📊 形状变换表

| 阶段 | 张量 | 形状 | 说明 |
|------|------|------|------|
| **Dataset** | images | `(T, C, H, W)` | 单个样本 |
| **DataLoader** | images | `(B, T, C, H, W)` | B=32, T=1 |
| **展平** | images_flat | `(B*T, C, H, W)` | ← 输入 VAE |
| **VAE 输出** | features | `(B*T, latent)` | B*T=32 |
| **恢复时间** | features | `(B, T, latent)` | 恢复结构 |
| **拼接** | combined | `(B, T, latent+state)` | ← 输入 Transformer |

---

## 🔧 常见问题速解

### Q1: 为什么不能用 squeeze?

```python
# squeeze(1) 的问题
images = torch.randn(32, 1, 3, 480, 640)
images_sq = images.squeeze(1)  # (32, 3, 480, 640)

# 时间维度信息丢失，无法恢复为原始的 (B, T, C, H, W)
# 而 reshape 保留了所有信息
images_flat = images.reshape(32 * 1, 3, 480, 640)  # (32, 3, 480, 640)
images_restored = images_flat.reshape(32, 1, 3, 480, 640)  # ✅ 恢复成功
```

### Q2: 为什么要展平?

```
VAE Encoder 的设计：
  - 期望：(B, C, H, W) — 4D
  - 你有：(B, T, C, H, W) — 5D
  
解决方案：
  - 将 (B, T) 组合为单一维度 B*T
  - 得到 (B*T, C, H, W) — 4D
  - 处理后恢复原始结构
```

### Q3: n_obs_steps 只支持 1 为什么还要维护 T 维度?

```python
# 原因 1: 代码兼容性
# 如果未来支持 n_obs_steps > 1，代码无需改动
images = torch.randn(32, 1, 3, 480, 640)  # 现在支持
images = torch.randn(32, 2, 3, 480, 640)  # 将来可能支持

# 原因 2: 与 ACT 设计一致
# ACT 的数据流保留时间维度，即使是 1
```

### Q4: 为什么会出现 "got 3 and 4" 错误?

```python
# 这个错误：torch.cat([A, B], dim=-1) 中 A 和 B 维数不同

# ❌ 错误的拼接
image_features = torch.randn(32, 1, 128)  # 3D
states = torch.randn(32, 15)               # 2D ← 维数不同！
torch.cat([image_features, states], dim=-1)  # RuntimeError

# ✅ 正确的拼接
image_features = torch.randn(32, 1, 128)  # 3D
states = torch.randn(32, 1, 15)            # 3D ← 维数相同！
torch.cat([image_features, states], dim=-1)  # (32, 1, 143) ✅
```

---

## 🎬 完整工作流

```python
# 1️⃣ DataLoader 输出
batch = next(iter(dataloader))
images = batch["observation"]["images"]  # (B, T, C, H, W)
states = batch["observation"]["states"]  # (B, T, state_dim)

# 2️⃣ 展平
B, T, C, H, W = images.shape
images_flat = images.reshape(B * T, C, H, W)  # (B*T, C, H, W)
states_flat = states.reshape(B * T, -1)       # (B*T, state_dim)

# 3️⃣ VAE Encoder
image_features = vae_encoder(images_flat)  # (B*T, latent_dim)

# 4️⃣ 恢复时间维度
image_features = image_features.reshape(B, T, -1)  # (B, T, latent_dim)

# 5️⃣ 与状态拼接
combined = torch.cat([image_features, states], dim=-1)  # (B, T, latent+state) ✅

# 6️⃣ 传递给 Transformer
output = transformer(combined)  # 处理完整的观测
```

---

## 💾 数据结构总结

### Dataset `__getitem__` 返回

```python
{
    "observation": {
        "images": np.ndarray,  # shape (n_obs_steps, 3, 480, 640)
        "states": np.ndarray,  # shape (n_obs_steps, 15)
    },
    "action": np.ndarray,      # shape (8, 6)
}
```

### DataLoader 输出

```python
{
    "observation": {
        "images": torch.Tensor,  # shape (B, n_obs_steps, 3, 480, 640)
        "states": torch.Tensor,  # shape (B, n_obs_steps, 15)
    },
    "action": torch.Tensor,      # shape (B, 8, 6)
}
```

### VAE Encoder 输入

```python
# 展平版本
{
    "images": torch.Tensor,  # shape (B*T, 3, 480, 640)  ← 4D
    "states": torch.Tensor,  # shape (B*T, 15)           ← 2D
}
```

---

## 🚨 错误排查树

```
错误：RuntimeError: Tensors must have same number of dimensions: got 3 and 4

├─ 检查 image_features 的维度
│  └─ 如果是 4D，需要 reshape 或 squeeze
│     ├─ 如果包含时间信息：reshape(B, T, -1)
│     └─ 如果不包含：squeeze()
│
├─ 检查 states 的维度
│  └─ 确保与 image_features 维数相同
│
└─ 如果维数相同，检查拼接轴
   └─ torch.cat([A, B], dim=-1) 的 A 和 B 应该有相同的维数
```

---

## 📌 记住这个

| 操作 | 输入 | 输出 | 何时用 |
|------|------|------|--------|
| **reshape** | `(B, T, C, H, W)` | `(B*T, C, H, W)` | ✅ 处理 VAE 输入 |
| **squeeze** | `(B, 1, C, H, W)` | `(B, C, H, W)` | ❌ 避免使用 |
| **stack** | list of (T, ...) | `(B, T, ...)` | ✅ collate 时使用 |
| **cat** | `(B, T, A)`, `(B, T, B)` | `(B, T, A+B)` | ✅ 拼接特征 |

---

## 🎓 学习资源

1. **本项目文档**：
   - [LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md) — 详细指南
   - [LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md) — 代码实现

2. **项目代码**：
   - [scripts/train_act_real_data.py](./scripts/train_act_real_data.py) — 完整训练脚本
   - [scripts/inference_engine.py](./scripts/inference_engine.py) — 推理实现
   - [test_act_minimal.py](./test_act_minimal.py) — 最小化测试

3. **外部资源**：
   - [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
   - PyTorch 文档

---

## ✅ 部署检查清单

在生产环境部署前，确保：

- [ ] 所有输入张量的形状已验证
- [ ] 使用 `reshape()` 而不是 `squeeze()`
- [ ] 时间维度始终保留为维度 1
- [ ] n_obs_steps 设置为 1（ACT 要求）
- [ ] torch.cat 的操作数具有相同的维数
- [ ] VAE encoder 输入是 4D `(B*T, C, H, W)`
- [ ] 恢复后的特征是 3D `(B, T, latent)`
- [ ] 拼接后的结果是 3D `(B, T, combined_dim)`

---

**版本**：1.0  
**最后更新**：2026-01-17  
**维护者**：So101 项目

