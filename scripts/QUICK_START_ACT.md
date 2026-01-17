# ACT 模型训练 - 快速开始指南

## 🚀 5 分钟快速开始

### 前置条件
- Python 3.8+
- PyTorch with CUDA support
- LeRobot 0.3.3+

### 安装依赖
```bash
# 如果还没安装 LeRobot
pip install lerobot

# 或从源代码安装（最新版本）
pip install git+https://github.com/huggingface/lerobot.git
```

### 一行命令训练所有模型

```bash
# 训练三个任务（lift, sort, stack）
python scripts/train_all_act_models.py

# 或指定自定义参数
python scripts/train_all_act_models.py --epochs 200 --batch-size 16 --device cuda
```

### 训练单个任务

```bash
# 训练 lift 任务
python scripts/train_act_real_data.py --task lift

# 训练 sort 任务
python scripts/train_act_real_data.py --task sort

# 训练 stack 任务
python scripts/train_act_real_data.py --task stack
```

---

## 📋 详细步骤

### 第一步：准备数据

确保数据结构：
```
real_data/
├── lift/
│   ├── meta/
│   │   ├── info.json
│   │   └── stats.json
│   └── data/
│       ├── chunk-0/
│       │   └── *.parquet
│       └── chunk-1/
│           └── *.parquet
├── sort/
│   ├── meta/
│   └── data/
└── stack/
    ├── meta/
    └── data/
```

✅ **数据检查**：
```bash
# 检查数据文件
ls -la real_data/lift/data/chunk-0/
# 输出应该包含 .parquet 文件

# 检查统计信息
cat real_data/lift/meta/stats.json
```

### 第二步：选择训练脚本

#### 方案 1：直接 Parquet（推荐 ✓）
```bash
python scripts/train_act_real_data.py --task lift
```
**优点**：
- 不需要网络连接
- 自动检测数据维度
- 完整的数据管道

#### 方案 2：LeRobotDataset
```bash
python scripts/train_act_real_data_lerobot_dataset.py --task lift
```
**优点**：
- 官方数据加载格式
- 与其他 LeRobot 项目兼容

### 第三步：开始训练

```bash
# 基础训练（推荐参数）
python scripts/train_act_real_data.py \
    --task lift \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4

# 关键输出：
# ✓ Loading dataset... Found 150 parquet files
# ✓ Detected state dimension: 15
# ✓ Detected action dimension: 6
# ✓ Creating ACT model...
# ✓ Model parameters: 8,234,567 (trainable: 8,234,567)
# ✓ Starting training...
```

### 第四步：监控训练进度

训练输出会显示：

```
Epoch 1/100 - Loss: 0.1234
Epoch 2/100 - Loss: 0.0987
...
✓ Saved best model to checkpoints/lift_act/checkpoint-best (loss: 0.0234)
```

**预期时间**：
- lift: ~60 分钟（100 epoch）
- sort: ~90 分钟（100 epoch，更复杂）
- stack: ~60 分钟（100 epoch）

### 第五步：验证训练结果

```bash
# 检查生成的检查点
ls -la checkpoints/lift_act/checkpoint-best/

# 输出应该包含：
# - config.json
# - pytorch_model.bin
# - optimizer.pt (optional)
```

---

## 🎯 使用训练好的模型

### 在推理脚本中加载

```python
from grasp_cube.real.act_policy import LeRobotACTPolicy

# 加载训练好的模型
policy = LeRobotACTPolicy.from_pretrained(
    "checkpoints/lift_act/checkpoint-best"
)

# 推理
observation = {
    "observation.images.front": image,  # (3, 480, 640)
    "observation.state": state,  # (state_dim,)
}
action = policy.select_action(observation)
# action.shape = (action_dim,)
```

### 在服务器中使用

```bash
# 使用已有的 ACT 服务器（无需修改）
python serve_act_policy.py \
    --checkpoint checkpoints/lift_act/checkpoint-best \
    --port 5000
```

---

## ⚙️ 常见配置

### 参数调整表

| 场景 | 推荐设置 |
|------|--------|
| **快速测试** | `--epochs 20 --batch-size 64` |
| **标准训练** | `--epochs 100 --batch-size 32` |
| **高精度** | `--epochs 200 --batch-size 16` |
| **内存不足** | `--batch-size 8 --learning-rate 5e-5` |
| **快速收敛** | `--learning-rate 2e-4 --epochs 50` |

### 任务特定参数

**lift 任务**：
```bash
python scripts/train_act_real_data.py \
    --task lift \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

**sort 任务**（更复杂）：
```bash
python scripts/train_act_real_data.py \
    --task sort \
    --epochs 150 \
    --batch-size 16 \
    --learning-rate 5e-5
```

**stack 任务**：
```bash
python scripts/train_act_real_data.py \
    --task stack \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

---

## 🐛 故障排除

### 问题 1: 找不到数据文件
```
FileNotFoundError: No parquet files found in real_data/lift/data/
```
**解决**：
```bash
# 检查文件是否存在
find real_data/ -name "*.parquet"

# 检查目录结构
ls -la real_data/lift/
ls -la real_data/lift/data/
```

### 问题 2: GPU 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**：
```bash
# 减小 batch size
python scripts/train_act_real_data.py --task lift --batch-size 8

# 或使用 CPU（速度慢）
python scripts/train_act_real_data.py --task lift --device cpu
```

### 问题 3: 导入错误
```
ImportError: No module named 'lerobot'
```
**解决**：
```bash
pip install lerobot --upgrade
```

### 问题 4: 损失不下降
**可能原因**：
1. 学习率太高
2. 数据不正确
3. 模型配置不合适

**解决**：
```bash
# 降低学习率
python scripts/train_act_real_data.py --task lift --learning-rate 5e-5

# 检查数据
python -c "
from scripts.train_act_real_data import RealDataACTDataset
import pathlib
dataset = RealDataACTDataset(pathlib.Path('real_data/lift'))
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Action shape: {sample[\"action\"].shape}')
"
```

---

## 📊 性能期望

### 训练损失曲线
```
初始损失：0.2-0.4（取决于数据规模）
50 epoch：0.05-0.1
100 epoch：0.01-0.05
```

### 推理性能
```
延迟：15-25ms（GPU）
吞吐量：~50 fps（批处理）
内存：~2GB
```

---

## 🔄 完整工作流示例

```bash
# 1. 训练所有模型
python scripts/train_all_act_models.py --epochs 100

# 2. 或依次训练
python scripts/train_act_real_data.py --task lift --epochs 100
python scripts/train_act_real_data.py --task sort --epochs 150
python scripts/train_act_real_data.py --task stack --epochs 100

# 3. 检查生成的模型
ls -la checkpoints/*/checkpoint-best/

# 4. 在真机上部署
python serve_act_policy.py --checkpoint checkpoints/lift_act/checkpoint-best

# 5. 运行推理客户端
python hello_pick_cube_web.py --checkpoint checkpoints/lift_act/checkpoint-best
```

---

## 📚 详细文档

- 📖 [完整训练指南](./README_ACT_TRAINING.md)
- 📊 [ACT vs DiffusionPolicy 对比](./ACT_vs_DiffusionPolicy_COMPARISON.md)
- 🔧 [API 参考](./train_act_real_data.py)

---

## 💡 最佳实践

### 1. 数据验证
```python
# 训练前检查数据
from scripts.train_act_real_data import RealDataACTDataset
dataset = RealDataACTDataset(pathlib.Path('real_data/lift'))
assert len(dataset) > 0, "No samples!"
print(f"✓ Dataset ready: {len(dataset)} samples")
```

### 2. 参数记录
```bash
# 用描述性名称保存检查点
mkdir -p checkpoints/lift_act_v2_lr1e-4_bs32
python scripts/train_act_real_data.py \
    --task lift \
    --output-dir checkpoints/lift_act_v2_lr1e-4_bs32 \
    --learning-rate 1e-4 \
    --batch-size 32
```

### 3. 增量训练
```bash
# 从现有检查点继续训练（需要修改脚本）
# model = ACTPolicy.from_pretrained("checkpoints/lift_act/checkpoint-best")
```

### 4. 定期备份
```bash
# 训练完成后备份
cp -r checkpoints/lift_act /backup/
```

---

## 🎓 下一步

1. **评估模型**：使用现有评估脚本测试模型性能
2. **微调**：在更多数据上继续训练
3. **部署**：部署到真机（现有服务器兼容）
4. **对比**：与 DiffusionPolicy 对比性能

---

## 📞 常见问题

**Q: 训练多个任务需要多长时间？**
A: 约 3-4 小时（顺序训练三个任务）

**Q: 能否同时训练多个任务？**
A: 可以，用不同的 GPU 或多进程

**Q: 能否从 DiffusionPolicy 的权重初始化？**
A: 不能，架构不同

**Q: 最小数据量是多少？**
A: 建议至少 100 条轨迹，>1000 条更佳

---

## 📝 脚本选择指南

```
你想训练 ACT 模型吗？
├─ 使用我们的真机数据？
│  └─ 是 → train_act_real_data.py ✓ 推荐
├─ 数据在 LeRobot Hub？
│  └─ 是 → train_act_real_data_lerobot_dataset.py
├─ 训练所有任务？
│  └─ 是 → train_all_act_models.py
└─ 只要一个脚本就行？
   └─ train_act_real_data.py (最灵活)
```

---

**最后更新**：2024年
**推荐开始方式**：`python scripts/train_all_act_models.py`
