# ACT 训练脚本 - 文档索引

**🎯 快速导航**

---

## 🚀 我想立即开始训练

👉 **推荐**：[QUICK_START_ACT.md](./QUICK_START_ACT.md)

最快 5 分钟开始训练：
```bash
python scripts/train_act_real_data.py --task lift
```

---

## 📚 我想了解完整细节

👉 **参考**：[README_ACT_TRAINING.md](./README_ACT_TRAINING.md)

包含：
- ✅ 关键参数说明
- ✅ 训练流程详解
- ✅ 常见问题解答
- ✅ 性能调优建议

---

## 🔄 我想对比 DiffusionPolicy 和 ACT

👉 **对比**：[ACT_vs_DiffusionPolicy_COMPARISON.md](./ACT_vs_DiffusionPolicy_COMPARISON.md)

包含：
- 📊 架构对比
- ⚡ 性能数据
- 🔧 配置差异
- 🎯 使用建议

---

## 📝 我想了解本次修改的内容

👉 **总结**：[MODIFICATIONS_SUMMARY.md](./MODIFICATIONS_SUMMARY.md)

包含：
- 📌 修改清单
- 🎯 核心功能
- 🚀 后续步骤
- ✅ 验证清单

---

## 💻 我想查看代码

### 三个训练脚本

| 脚本 | 用途 | 推荐度 |
|------|------|--------|
| [**train_act_real_data.py**](./train_act_real_data.py) | 直接 Parquet 加载 | ⭐⭐⭐⭐⭐ |
| [**train_act_real_data_lerobot_dataset.py**](./train_act_real_data_lerobot_dataset.py) | LeRobotDataset 官方格式 | ⭐⭐⭐⭐ |
| [**train_all_act_models.py**](./train_all_act_models.py) | 一键训练所有任务 | ⭐⭐⭐⭐ |

---

## 🎓 按使用场景导航

### 场景 1：我是新用户，想快速上手
```
1. 阅读：QUICK_START_ACT.md（5分钟）
2. 运行：python scripts/train_act_real_data.py --task lift
3. 完成！
```

### 场景 2：我了解 DiffusionPolicy，想迁移到 ACT
```
1. 阅读：ACT_vs_DiffusionPolicy_COMPARISON.md
2. 检查：导入和配置变化
3. 运行：新训练脚本
4. 对比：性能差异
```

### 场景 3：我想优化训练性能
```
1. 阅读：README_ACT_TRAINING.md 中的 "常见问题"
2. 调整：学习率、batch size 等参数
3. 监控：损失曲线
4. 评估：最终模型
```

### 场景 4：我遇到问题
```
1. 查看：README_ACT_TRAINING.md 中的 "常见问题"
2. 或查看：QUICK_START_ACT.md 中的 "故障排除"
3. 检查：脚本中的注释
```

### 场景 5：我要部署到真机
```
1. 了解：ACT_vs_DiffusionPolicy_COMPARISON.md
2. 训练：python scripts/train_act_real_data.py
3. 部署：python serve_act_policy.py --checkpoint ...
4. 推理：现有客户端代码兼容
```

---

## 📖 文档结构

```
scripts/
├── 📚 QUICK_START_ACT.md                    # 入门指南
├── 📚 README_ACT_TRAINING.md                # 详细参考
├── 📚 ACT_vs_DiffusionPolicy_COMPARISON.md  # 技术对比
├── 📚 MODIFICATIONS_SUMMARY.md              # 修改总结
├── 📚 INDEX.md                              # 本文件
│
├── 💻 train_act_real_data.py                # 主训练脚本
├── 💻 train_act_real_data_lerobot_dataset.py # 官方格式训练
└── 💻 train_all_act_models.py               # 批量训练脚本
```

---

## 🔍 快速查询

### 我想知道...

| 问题 | 答案位置 |
|------|---------|
| 如何开始训练？ | [QUICK_START_ACT.md](./QUICK_START_ACT.md) § 一行命令 |
| ACT 有什么优势？ | [ACT_vs_DiffusionPolicy_COMPARISON.md](./ACT_vs_DiffusionPolicy_COMPARISON.md) § 性能对比 |
| 参数怎么调？ | [README_ACT_TRAINING.md](./README_ACT_TRAINING.md) § 关键参数说明 |
| GPU 内存不足？ | [QUICK_START_ACT.md](./QUICK_START_ACT.md) § 故障排除 |
| 如何评估模型？ | [README_ACT_TRAINING.md](./README_ACT_TRAINING.md) § 使用训练好的模型 |
| 如何迁移代码？ | [ACT_vs_DiffusionPolicy_COMPARISON.md](./ACT_vs_DiffusionPolicy_COMPARISON.md) § 迁移指南 |
| 修改了什么？ | [MODIFICATIONS_SUMMARY.md](./MODIFICATIONS_SUMMARY.md) § 修改清单 |

---

## ⏱️ 阅读时间指南

| 文档 | 长度 | 阅读时间 | 优先级 |
|------|------|---------|--------|
| QUICK_START_ACT.md | 短 | 5 分钟 | 🔴 必读 |
| README_ACT_TRAINING.md | 中 | 15 分钟 | 🟠 推荐 |
| ACT_vs_DiffusionPolicy_COMPARISON.md | 中 | 15 分钟 | 🟡 可选 |
| MODIFICATIONS_SUMMARY.md | 长 | 20 分钟 | 🟢 参考 |

---

## 🎯 按角色推荐阅读顺序

### 👨‍💻 开发者（快速上手）
1. QUICK_START_ACT.md（5分钟）
2. train_act_real_data.py 脚本（浏览）
3. 直接运行！

### 🔬 研究员（深入了解）
1. README_ACT_TRAINING.md（完整）
2. ACT_vs_DiffusionPolicy_COMPARISON.md（技术细节）
3. 脚本代码注释（30分钟）
4. 论文参考

### 🚀 运维人员（部署关注）
1. QUICK_START_ACT.md
2. MODIFICATIONS_SUMMARY.md § 后续步骤
3. train_all_act_models.py 脚本
4. 部署流程

### 📚 学生（学习目标）
1. ACT_vs_DiffusionPolicy_COMPARISON.md § 架构对比
2. README_ACT_TRAINING.md § 训练流程
3. train_act_real_data.py 代码详读
4. ACT 论文

---

## 🔗 相关链接

### LeRobot 官方资源
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot 文档](https://huggingface.co/docs/lerobot/)
- [ACT 论文](https://arxiv.org/abs/2304.13705)

### 本项目关键文件
- `grasp_cube/real/act_policy.py` — ACTPolicy 实现
- `grasp_cube/real/serve_act_policy.py` — 推理服务器
- `real_data/` — 真机数据目录

---

## 📋 检查清单

使用前确保完成：

- [ ] 阅读了 QUICK_START_ACT.md
- [ ] 检查了数据完整性（`ls real_data/lift/data/`）
- [ ] 安装了 LeRobot（`pip install lerobot`）
- [ ] 有可用的 CUDA GPU 或 CPU
- [ ] 了解了训练参数含义
- [ ] 记录了想要的超参数

---

## 🎁 获益速查

| 想获得 | 查看文件 |
|--------|---------|
| 快速训练模型 | QUICK_START_ACT.md |
| 参数优化建议 | README_ACT_TRAINING.md |
| 技术理论知识 | ACT_vs_DiffusionPolicy_COMPARISON.md |
| 完整 API 参考 | train_act_real_data.py 源码注释 |
| 迁移指导 | ACT_vs_DiffusionPolicy_COMPARISON.md § 迁移指南 |
| 故障排除 | QUICK_START_ACT.md § 故障排除 |
| 最佳实践 | README_ACT_TRAINING.md § 最佳实践 |

---

## 💬 常见问题速查

### Q: 我应该从哪里开始？
A: 先读 [QUICK_START_ACT.md](./QUICK_START_ACT.md)，然后直接运行脚本。

### Q: 哪个脚本推荐使用？
A: `train_act_real_data.py`（直接 Parquet 加载，最灵活）

### Q: 需要多久学会？
A: 
- 快速上手：5 分钟
- 完整理解：1 小时
- 熟练使用：1 天

### Q: 可以边学边练吗？
A: 是的！边读 QUICK_START_ACT.md 边运行脚本是最快的学习方式。

### Q: 代码有没有例子？
A: 有！每个文档都包含可直接运行的代码示例。

---

## 📞 获取帮助

### 第一步：检查文档
1. 查看对应文档中的 "常见问题" 部分
2. 搜索关键词

### 第二步：查看脚本注释
1. 打开 `.py` 文件
2. 查看代码注释（中英文都有）

### 第三步：查看错误日志
1. 完整阅读错误信息
2. 搜索错误关键词

### 第四步：调整参数
1. 参考对应文档推荐值
2. 逐步调整并测试

---

## 🎉 开始使用

最快开始方式（复制粘贴）：

```bash
# 选项 1：训练一个任务
python scripts/train_act_real_data.py --task lift

# 选项 2：训练所有任务
python scripts/train_all_act_models.py

# 选项 3：自定义参数
python scripts/train_act_real_data.py \
    --task lift \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 5e-5
```

👉 **然后阅读对应文档以了解细节。**

---

## 📊 文档关系图

```
QUICK_START_ACT.md (5分钟入门)
    ↓
    ├─→ README_ACT_TRAINING.md (详细参考)
    │      ↓
    │      └─→ train_act_real_data.py (代码)
    │
    └─→ ACT_vs_DiffusionPolicy_COMPARISON.md (技术对比)
           ↓
           └─→ 决定迁移策略
    
MODIFICATIONS_SUMMARY.md (修改总结)
    ↓
    └─→ 了解项目变化
```

---

## ✨ 总结

| 需求 | 解决方案 |
|------|---------|
| 🚀 快速开始 | 阅读 QUICK_START_ACT.md，然后运行脚本 |
| 📖 详细学习 | 阅读 README_ACT_TRAINING.md 的完整内容 |
| 🔄 从 DiffusionPolicy 迁移 | 参考 ACT_vs_DiffusionPolicy_COMPARISON.md |
| 💻 查看代码 | 打开 train_act_real_data.py |
| 🎯 了解改动 | 查看 MODIFICATIONS_SUMMARY.md |
| 🔍 找特定答案 | 使用本索引文件的快速查询表 |

---

**最后更新**：2024年
**推荐开始**：QUICK_START_ACT.md
**推荐脚本**：train_act_real_data.py
**推荐命令**：`python scripts/train_act_real_data.py --task lift`

祝你训练顺利！🎉
