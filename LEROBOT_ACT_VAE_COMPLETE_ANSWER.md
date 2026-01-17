# LeRobot ACT VAE Encoder 输入形状 - 完整答案

## 📋 查询需求回顾

你要求查找以下信息：
1. VAE encoder 的 forward 方法及其对输入形状的期望
2. 当 n_obs_steps=1 时的特殊处理
3. 如何正确构造 vae_encoder_input（应该是什么形状）
4. torch.cat 操作中的维度错误原因

---

## ✅ 完整答案

### 1. VAE Encoder 的输入形状要求

#### 标准期望（来自 LeRobot 官方实现）

```python
# VAE Encoder 期望的输入形状
images_input = torch.Tensor  # 形状：(B, C, H, W) — 4维

# 具体参数：
#   B: batch_size (例如 32)
#   C: channels = 3 (RGB)
#   H: height = 480
#   W: width = 640

# 完整示例
batch_size = 32
images_for_vae = torch.randn(batch_size, 3, 480, 640)  # ✅ 正确
```

#### 来自 DataLoader 的实际输入

```python
# DataLoader 提供的形状（包含时间维度）
batch = {
    "observation.images.front": torch.randn(32, 1, 3, 480, 640),  # (B, T, C, H, W) — 5维
    "observation.state": torch.randn(32, 1, 15),                   # (B, T, state_dim) — 3维
}
```

#### 为什么会有 T 维度？

即使 ACT 仅支持 `n_obs_steps=1`，LeRobot 仍然在数据中保留时间维度：

```python
# 原因 1: 设计一致性
#   LeRobot 的所有政策都使用 (B, T, ...) 格式
#   即使 T=1，也保持维度以保持代码一致

# 原因 2: 代码灵活性
#   如果未来支持 n_obs_steps > 1，代码无需改动

# 原因 3: 数据处理管道
#   数据集返回 (T, C, H, W)
#   DataLoader collate 时堆叠成 (B, T, C, H, W)
#   这是标准的处理流程
```

---

### 2. 正确的 VAE Encoder 输入构造方式

#### 核心原则

```python
# ✅ 正确方式：展平 (B, T) 维度
B, T, C, H, W = batch["observation.images.front"].shape
images_for_vae = batch["observation.images.front"].reshape(B * T, C, H, W)
# 结果形状：(32*1, 3, 480, 640) = (32, 3, 480, 640)

# ❌ 错误方式：squeeze
images_squeezed = batch["observation.images.front"].squeeze(1)
# 虽然形状看起来对了 (32, 3, 480, 640)，但丢失了时间维度信息
```

#### 完整的数据流

```python
# Step 1: 从 DataLoader 获取批次
batch = next(iter(dataloader))
images = batch["observation.images.front"]  # (B, T, C, H, W) = (32, 1, 3, 480, 640)
states = batch["observation.state"]          # (B, T, state_dim) = (32, 1, 15)

# Step 2: 展平用于 VAE encoder
B, T, C, H, W = images.shape
images_for_vae = images.reshape(B * T, C, H, W)      # (32, 3, 480, 640) ✅
states_for_vae = states.reshape(B * T, -1)           # (32, 15) ✅

# Step 3: 通过 VAE encoder
vae_encoder = ...  # 你的 VAE encoder 模型
image_features = vae_encoder(images_for_vae)         # (32, latent_dim) = (32, 128)

# Step 4: 恢复时间维度
image_features = image_features.reshape(B, T, -1)    # (32, 1, 128) ✅

# Step 5: 拼接图像特征和状态
combined = torch.cat([image_features, states], dim=-1)  # (32, 1, 128+15) = (32, 1, 143) ✅

# Step 6: 传递给 Transformer
# combined 现在的形状是 (B, T, latent_dim + state_dim)
# Transformer 会处理这个完整的观测
```

---

### 3. images 和 states 应该是什么形状

#### 数据来源和形状演变

| 来源 | images 形状 | states 形状 | 说明 |
|------|------------|-----------|------|
| **Dataset** | `(n_obs_steps, 3, 480, 640)` | `(n_obs_steps, 15)` | 单个样本 |
| **DataLoader** | `(B, n_obs_steps, 3, 480, 640)` | `(B, n_obs_steps, 15)` | 批次 |
| **展平后** | `(B*n_obs_steps, 3, 480, 640)` | `(B*n_obs_steps, 15)` | VAE 输入 |
| **VAE 输出** | — | — | `(B*n_obs_steps, latent_dim)` |
| **恢复后** | — | — | `(B, n_obs_steps, latent_dim)` |

#### 当 n_obs_steps=1 时的具体值

```python
# 所有张量的具体形状
batch_size = 32
n_obs_steps = 1
image_height = 480
image_width = 640
state_dim = 15
latent_dim = 128

# 从 DataLoader
images = torch.randn(32, 1, 3, 480, 640)      # (B, T, C, H, W)
states = torch.randn(32, 1, 15)                # (B, T, state_dim)

# 展平
images_flat = images.reshape(32, 3, 480, 640)  # (B*T, C, H, W) = (32, 3, 480, 640)
states_flat = states.reshape(32, 15)            # (B*T, state_dim) = (32, 15)

# VAE 输出
image_features = torch.randn(32, 128)           # (B*T, latent_dim)

# 恢复时间维度
image_features_restored = image_features.reshape(32, 1, 128)  # (B, T, latent_dim)

# 拼接
combined = torch.cat([image_features_restored, states], dim=-1)  # (32, 1, 143)
```

---

### 4. n_obs_steps 维度的正确处理方式

#### 为什么 squeeze 是错误的

```python
# 情况 1: n_obs_steps = 1
images = torch.randn(32, 1, 3, 480, 640)
images_squeezed = images.squeeze(1)  # (32, 3, 480, 640)

# 问题：如果后面要恢复时间维度，无法确定原始的 T 值
# reshape(32, 1, 3, 480, 640) 需要知道 B=32, T=1
# 但从 (32, 3, 480, 640) 看不出来

# 情况 2: n_obs_steps = 2（虽然 ACT 不支持，但理论上）
images = torch.randn(32, 2, 3, 480, 640)
images_squeezed = images.squeeze(1)  # ❌ 这不会 squeeze，因为维度 1 的大小是 2
# 或者如果错误地用 squeeze()（不指定维度）
# squeeze() 会删除所有大小为 1 的维度，导致无法预测结果

# 结论：squeeze 容易导致问题，reshape 更安全
```

#### reshape 的正确用法

```python
# ✅ 推荐方式：保存原始形状信息
B, T, C, H, W = images.shape  # (32, 1, 3, 480, 640)

# 展平
images_flat = images.reshape(B * T, C, H, W)  # (32, 3, 480, 640)

# 处理后恢复
processed_features = process(images_flat)      # (32, latent_dim)
restored = processed_features.reshape(B, T, -1)  # (32, 1, latent_dim) ✅

# 即使 T 改变（理论上），代码仍然有效
B_new, T_new = 32, 2
images_new = torch.randn(B_new, T_new, 3, 480, 640)
images_flat_new = images_new.reshape(B_new * T_new, 3, 480, 640)  # (64, 3, 480, 640)
```

#### n_obs_steps=1 时的特殊处理

```python
# 虽然 n_obs_steps=1，但不应该删除这个维度
# 原因：
# 1. 保持与 ACT 设计的一致性
# 2. 代码兼容性（如果支持 n_obs_steps > 1）
# 3. 时间序列特性（即使是单步，也表示观测时间点）

# ✅ 正确的 n_obs_steps=1 处理
n_obs_steps = 1
images = torch.randn(B, n_obs_steps, 3, 480, 640)  # 保持维度
states = torch.randn(B, n_obs_steps, state_dim)     # 保持维度

# 展平时保留 T 信息
images_flat = images.reshape(B * n_obs_steps, 3, 480, 640)
states_flat = states.reshape(B * n_obs_steps, -1)

# 处理后恢复
# ... 处理 ...
features_restored = features.reshape(B, n_obs_steps, -1)

# ✅ 即使 n_obs_steps=1，这种方式也工作良好
```

---

### 5. torch.cat 维度不匹配错误的原因

#### 完整的错误情景

```python
# ❌ 错误示例
images = torch.randn(32, 1, 3, 480, 640)      # 5D
states = torch.randn(32, 15)                   # 2D （缺少 T 维度）

# VAE encoder 返回可能是 4D 或 3D
image_features = vae_encoder(images)           # 返回形状不确定

# 直接拼接会失败
try:
    combined = torch.cat([image_features, states], dim=-1)
except RuntimeError as e:
    # RuntimeError: Tensors must have same number of dimensions: got 3 and 4
```

#### 错误的根本原因

```python
# 问题分析
# 1. states 只有 2D: (B, state_dim) = (32, 15)
# 2. image_features 可能是 3D: (B, latent) 或 4D: (B, T, latent)
#    ↑ 这取决于 VAE encoder 的输入和输出处理方式

# 如果：
#   image_features = (32, 1, 128)  — 3D
#   states = (32, 15)               — 2D
# 无法拼接：维数不同

# 解决方案：确保维数相同
# 选项 A: 都转为 2D
image_features = image_features.reshape(B * T, -1)  # (32, 128)
states = states.reshape(B * T, -1)                  # (32, 15)
combined = torch.cat([image_features, states], dim=-1)  # (32, 143) ✅

# 选项 B: 都转为 3D
image_features = image_features.reshape(B, T, -1)   # (32, 1, 128)
states = states.reshape(B, T, -1)                   # (32, 1, 15)
combined = torch.cat([image_features, states], dim=-1)  # (32, 1, 143) ✅
```

#### 标准的 ACT 实现方式

```python
# 来自 inference_engine.py 的真实实现
B, T, C, H, W = batch["observation.images.front"].shape

# 关键步骤：展平 images 用于 VAE encoder
batch["observation.images.front"] = batch["observation.images.front"].reshape(B * T, C, H, W)

# 现在：
#   images: (B*T, C, H, W) = (32, 3, 480, 640)  ← VAE encoder 期望的输入
#   states: (B, n_obs_steps, state_dim) = (32, 1, 15)

# ✅ ACT 内部处理这些展平和恢复的细节
# 用户只需要确保：
#   1. images 在进入前展平
#   2. states 保持 (B, T, state_dim) 格式
#   3. 模型内部会正确处理拼接
```

---

## 📊 完整的数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                       数据源（Dataset）                          │
│  images: (n_obs_steps, 3, 480, 640)  states: (n_obs_steps, 15) │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │  DataLoader collate_fn │
            │    (stack along B)     │
            └────────────┬───────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    批次（Batch）                                 │
│  images: (B, T, 3, 480, 640)      states: (B, T, 15)          │
│           ↓                                 ↓                   │
│        展平                              保持 T                 │
│           ↓                                 ↓                   │
│  (B*T, 3, 480, 640)                   (B, T, 15)             │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │    VAE Encoder         │
            │  输入: (B*T, C, H, W)  │
            │  输出: (B*T, latent)   │
            └────────────┬───────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │  恢复时间维度           │
            │  reshape(B, T, latent) │
            └────────────┬───────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              处理后的观测（已准备好拼接）                         │
│  image_features: (B, T, latent)    states: (B, T, state_dim)  │
│                      ↓                       ↓                  │
│                   128 dims              15 dims                 │
│                      ↓                       ↓                  │
│              ┌─────────────────────────┐                        │
│              │  torch.cat(..., dim=-1) │                        │
│              └─────────────┬───────────┘                        │
│                            ↓                                    │
│            combined: (B, T, 143)  ✅                           │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │   Transformer          │
            │  完整的观测处理         │
            └────────────┬───────────┘
                         │
                         ↓
            ┌────────────────────────┐
            │   预测动作              │
            │  output: (B, horizon) │
            └────────────────────────┘
```

---

## 🎓 核心要点总结

### VAE Encoder 输入要求

1. **预期形状**：`(B, C, H, W)` — 4D
2. **为什么不是 5D**：VAE 只处理单个时间步的图像
3. **如何转换**：从 `(B, T, C, H, W)` → `reshape(B*T, C, H, W)`

### 维度处理原则

1. **保存时间信息**：始终保留 `T` 维度信息（即使 T=1）
2. **使用 reshape**：而不是 squeeze
3. **恢复形状**：处理后立即恢复原始维度结构

### 拼接的关键

1. **维数要相同**：两个拼接的张量维数必须相同
2. **形状一致**：除了拼接维度外，其他维度必须相同
3. **顺序无关**：`cat([A, B], dim=-1)` 和 `cat([B, A], dim=-1)` 都可以

### n_obs_steps=1 的特殊性

1. **仍然保留维度**：即使 T=1，也要维持 `(B, 1, ...)`
2. **代码兼容性**：支持将来的 `n_obs_steps > 1`
3. **标准化处理**：与 LeRobot 其他政策保持一致

---

## 📚 参考资源

### 本项目的详细文档

1. **[LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md)**
   - 完整的理论解释
   - 错误原因分析
   - 验证清单

2. **[LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)**
   - 完整的 Python 代码示例
   - 数据加载管道
   - 常见错误排查

3. **[LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)**
   - 快速参考卡
   - 常见问题速解
   - 代码模板

### 项目实现

- [scripts/train_act_real_data.py](./scripts/train_act_real_data.py)
- [scripts/inference_engine.py](./scripts/inference_engine.py)
- [test_act_minimal.py](./test_act_minimal.py)

---

## 🎯 最后的建议

### 实施时的检查清单

- [ ] 数据集返回 `(T, C, H, W)` 格式的图像
- [ ] DataLoader collate_fn 堆叠为 `(B, T, C, H, W)`
- [ ] 展平时使用 `reshape(B*T, C, H, W)` 而不是 `squeeze()`
- [ ] 保存原始的 `B` 和 `T` 值用于恢复
- [ ] VAE encoder 接收 4D 输入 `(B*T, C, H, W)`
- [ ] VAE encoder 输出形状为 `(B*T, latent_dim)`
- [ ] 恢复时间维度：`reshape(B, T, -1)`
- [ ] 拼接前确保 image_features 和 states 的维数相同
- [ ] `n_obs_steps` 设置为 1（ACT 要求）

### 调试技巧

```python
# 随时检查形状
print(f"images: {images.shape}")
print(f"images.ndim: {images.ndim}D")

# 逐步跟踪变换
print(f"Before reshape: {images.shape}")
images_flat = images.reshape(B*T, C, H, W)
print(f"After reshape: {images_flat.shape}")

# 验证拼接
print(f"image_features.ndim: {image_features.ndim}")
print(f"states.ndim: {states.ndim}")
if image_features.ndim == states.ndim:
    combined = torch.cat([image_features, states], dim=-1)
else:
    print(f"ERROR: Dimension mismatch!")
```

---

**版本**：1.0  
**最后更新**：2026-01-17  
**完成状态**：✅ 完整

