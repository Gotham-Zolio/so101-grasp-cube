# LeRobot ACT VAE Encoder 文档索引

## 📚 文档导航

本索引帮助你快速找到所需的信息。

---

## 📋 文档列表

### 1. 🎯 [LEROBOT_ACT_VAE_COMPLETE_ANSWER.md](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md) ⭐ **从这里开始**

**适合**：需要快速了解全貌的人  
**内容**：
- ✅ 直接回答你的 4 个问题
- ✅ VAE encoder 输入的确切要求
- ✅ images 和 states 应该是什么形状
- ✅ n_obs_steps=1 的特殊处理
- ✅ torch.cat 维度错误的原因

**长度**：约 15 分钟阅读  
**格式**：结构化答案 + 代码示例

---

### 2. 📖 [LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md) ⭐ **完整教程**

**适合**：想深入理解的人  
**内容**：
- 📌 VAE Encoder 的基本概念
- 📊 标准输入形状表
- 🔧 5 种维度处理方式（对比分析）
- 📍 完整的数据流示例
- 🚨 7 种常见错误及解决方案
- ✅ 验证清单

**长度**：约 40 分钟阅读  
**格式**：教科书式 + 大量代码示例

---

### 3. 💻 [LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md) ⭐ **实现指南**

**适合**：要实现代码的人  
**内容**：
- 💡 最小化测试代码
- 📊 完整的数据加载管道
- 🔄 与 ACTPolicy 集成的例子
- 🐛 常见错误排查（包含调试技巧）
- ✅ 验证清单（可复制粘贴）

**长度**：约 45 分钟阅读  
**格式**：代码优先 + 详细注释

---

### 4. ⚡ [LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md) ⭐ **速查表**

**适合**：需要快速查找信息的人  
**内容**：
- 🎯 一句话总结
- ⚡ 5 行代码示例（正确）
- ❌ 常见错误代码
- 📊 形状变换表
- 🔧 常见问题速解
- 📌 关键点记住

**长度**：约 5 分钟阅读  
**格式**：速查表 + 表格 + 简洁代码

---

## 🎯 如何选择合适的文档

### 我刚开始，想快速了解
→ **[LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)**
- 5 分钟快速入门
- 学习最核心的概念
- 看到正确和错误的做法

### 我想回答具体的问题
→ **[LEROBOT_ACT_VAE_COMPLETE_ANSWER.md](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md)**
- 直接回答 4 个问题
- 包含完整的例子
- 有数据流图

### 我想理解为什么
→ **[LEROBOT_ACT_VAE_ENCODER_GUIDE.md](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md)**
- 详细的概念解释
- 为什么每一步都重要
- 所有常见错误的原因

### 我要写代码
→ **[LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)**
- 可复制粘贴的代码
- 数据加载管道示例
- 详细的调试步骤

### 我在debug，很着急
→ **[LEROBOT_ACT_VAE_QUICK_REFERENCE.md](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)**  
→ **[LEROBOT_ACT_VAE_IMPLEMENTATION.md](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)** (搜索 "常见错误排查")

---

## 🔑 核心知识速览

### 一句话总结
**使用 `reshape(B*T, C, H, W)` 而不是 `squeeze()` 来处理 VAE encoder 的输入！**

### 关键形状变换

```
DataLoader      VAE输入       VAE输出        恢复              拼接
   ↓              ↓            ↓              ↓                ↓
(B,T,C,H,W)  →  (B*T,C,H,W)  →  (B*T,L)  →  (B,T,L)  →  (B,T,L+S)
(B,T,state)       展平           编码        恢复            拼接
```

### 最常见的错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `RuntimeError: got 3 and 4` | 维数不匹配 | 确保都是 3D 或都是 2D |
| `Expected (B,C,H,W), got (B,T,C,H,W)` | 输入维度多了 | 展平：`reshape(B*T, C, H, W)` |
| `无法恢复时间维度` | 用了 squeeze | 改用 reshape 并保存原始形状 |

---

## 📁 文档内容快速查找

### VAE Encoder 基础
- **预期输入形状**：[LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#1-vae-encoder的输入形状要求](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#1-vae-encoder的输入形状要求)
- **为什么是这个形状**：[LEROBOT_ACT_VAE_ENCODER_GUIDE.md#1️⃣-vae-encoder-的基本概念](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#1️⃣-vae-encoder-的基本概念)

### 正确的实现方式
- **完整代码示例**：[LEROBOT_ACT_VAE_IMPLEMENTATION.md#第一部分-完整实现示例](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#第一部分-完整实现示例)
- **数据加载管道**：[LEROBOT_ACT_VAE_IMPLEMENTATION.md#2-完整的数据加载管道](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#2-完整的数据加载管道)
- **快速模板**：[LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-完整工作流](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-完整工作流)

### n_obs_steps 处理
- **为什么保留 T 维度**：[LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#为什么会有-t-维度](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#为什么会有-t-维度)
- **reshape vs squeeze**：[LEROBOT_ACT_VAE_ENCODER_GUIDE.md#4️⃣-n_obs_steps-的特殊处理](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#4️⃣-n_obs_steps-的特殊处理)
- **使用建议**：[LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q1-为什么不能用-squeeze](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q1-为什么不能用-squeeze)

### torch.cat 错误
- **错误的完整分析**：[LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#5-torchcat-维度不匹配错误的原因](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#5-torchcat-维度不匹配错误的原因)
- **调试步骤**：[LEROBOT_ACT_VAE_IMPLEMENTATION.md#错误-1-tensor-维度不匹配](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#错误-1-tensor-维度不匹配)
- **快速排查**：[LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-错误排查树](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-错误排查树)

### 常见问题
- **完整 Q&A**：[LEROBOT_ACT_VAE_ENCODER_GUIDE.md#7️⃣-常见错误及解决方案](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#7️⃣-常见错误及解决方案)
- **快速 Q&A**：[LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-常见问题速解](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-常见问题速解)
- **排查指南**：[LEROBOT_ACT_VAE_IMPLEMENTATION.md#第二部分-常见错误排查](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#第二部分-常见错误排查)

---

## 🚀 推荐阅读路径

### 路径 1：快速学习（15 分钟）
```
1. 快速参考卡 (5 min)
   ↓
2. 完整答案 (10 min)
```
**得到**：核心知识 + 基本实现思路

### 路径 2：深入学习（60 分钟）
```
1. 快速参考卡 (5 min)
   ↓
2. 完整答案 (15 min)
   ↓
3. 编码指南 (30 min)
   ↓
4. 实现代码 (10 min)
```
**得到**：完整理解 + 可用代码

### 路径 3：快速参考（5 分钟）
```
1. 快速参考卡
   ↓
2. 需要时查阅其他文档
```
**得到**：快速查找表 + 链接到详细文档

### 路径 4：Debug（10-30 分钟）
```
1. 确定错误类型 (5 min)
   ↓
2. 查快速参考卡或实现指南的错误排查 (5-25 min)
   ↓
3. 根据链接查看完整解析 (如需要)
```
**得到**：错误原因 + 解决方案

---

## 💡 学习建议

### 初学者
1. 先读[快速参考卡](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)了解整体框架
2. 再读[完整答案](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md)理解细节
3. 查看[代码示例](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)学习实现
4. 遇到错误时参考[错误排查](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#7️⃣-常见错误及解决方案)

### 有经验的开发者
1. 快速扫一下[快速参考卡](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md)
2. 需要时查阅[完整答案](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md)或[实现指南](./LEROBOT_ACT_VAE_IMPLEMENTATION.md)
3. 复制相关代码示例适配到你的项目

### 调试中的开发者
1. 看[快速参考卡的错误排查树](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-错误排查树)
2. 查[实现指南的 debug 部分](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#第二部分-常见错误排查)
3. 根据需要深入[完整指南](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md)

---

## 🔍 按问题查找

### "为什么会出现 'got 3 and 4' 错误?"
- [LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#5-torchcat-维度不匹配错误的原因](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#5-torchcat-维度不匹配错误的原因)
- [LEROBOT_ACT_VAE_ENCODER_GUIDE.md#错误-1-tensor-维度不匹配](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#错误-1-tensor-维度不匹配)
- [LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q4-为什么会出现-got-3-and-4-错误](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q4-为什么会出现-got-3-and-4-错误)

### "应该用 squeeze 还是 reshape?"
- [LEROBOT_ACT_VAE_ENCODER_GUIDE.md#❌-错误做法-1-squeeze-导致维度不匹配](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#❌-错误做法-1-squeeze-导致维度不匹配)
- [LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q1-为什么不能用-squeeze](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#q1-为什么不能用-squeeze)

### "VAE encoder 具体期望什么输入?"
- [LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#1-vae-encoder的输入形状要求](./LEROBOT_ACT_VAE_COMPLETE_ANSWER.md#1-vae-encoder的输入形状要求)
- [LEROBOT_ACT_VAE_ENCODER_GUIDE.md#2️⃣-vae-encoder-的预期输入形状](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#2️⃣-vae-encoder-的预期输入形状)

### "我想看完整的代码示例"
- [LEROBOT_ACT_VAE_IMPLEMENTATION.md#1-最小化测试验证基础功能](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#1-最小化测试验证基础功能)
- [LEROBOT_ACT_VAE_IMPLEMENTATION.md#2-完整的数据加载管道](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#2-完整的数据加载管道)
- [LEROBOT_ACT_VAE_IMPLEMENTATION.md#3-与-actpolicy-集成的完整例子](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#3-与-actpolicy-集成的完整例子)

### "我要验证我的代码是否正确"
- [LEROBOT_ACT_VAE_ENCODER_GUIDE.md#9️⃣-总结表格](./LEROBOT_ACT_VAE_ENCODER_GUIDE.md#9️⃣-总结表格)
- [LEROBOT_ACT_VAE_IMPLEMENTATION.md#第三部分-验证清单](./LEROBOT_ACT_VAE_IMPLEMENTATION.md#第三部分-验证清单)
- [LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-部署检查清单](./LEROBOT_ACT_VAE_QUICK_REFERENCE.md#-部署检查清单)

---

## 📞 快速参考

### 最常用的代码片段

**展平图像用于 VAE encoder**
```python
B, T, C, H, W = batch["images"].shape
images_for_vae = batch["images"].reshape(B * T, C, H, W)
```

**恢复时间维度**
```python
image_features = image_features.reshape(B, T, -1)
```

**拼接图像特征和状态**
```python
combined = torch.cat([image_features, states], dim=-1)
```

### 最常见的错误模式

| 错误 | 代码 | 修复 |
|------|------|------|
| 维数不匹配 | `cat([3D, 2D])` | 确保都是 `(B, T, ...)` |
| 输入维度多 | VAE 输入 5D | `reshape(B*T, ...)` |
| 无法恢复形状 | `squeeze()` | 改用 `reshape()` |

---

## 🔗 相关资源

### 项目文件
- [scripts/train_act_real_data.py](./scripts/train_act_real_data.py) - 完整训练实现
- [scripts/inference_engine.py](./scripts/inference_engine.py) - 推理实现
- [test_act_minimal.py](./test_act_minimal.py) - 最小化测试

### 外部资源
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [PyTorch 文档](https://pytorch.org/docs/)

---

**版本**：1.0  
**最后更新**：2026-01-17  
**维护者**：So101 项目

