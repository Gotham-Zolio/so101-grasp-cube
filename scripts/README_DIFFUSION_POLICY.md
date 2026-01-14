# Diffusion Policy 训练和部署指南

本指南说明如何使用 motion planning 数据训练 Diffusion Policy，并在仿真和真机上部署。

## 工作流程概览

```
1. 数据收集 (Motion Planning) 
   → 2. 格式转换 (ManiSkill → LeRobot)
   → 3. 训练 (LeRobot train.py)
   → 4. 评估 (仿真/真机)
```

## 步骤 1: 数据收集

使用 motion planning 解决方案生成成功的轨迹数据。

```bash
# 收集 lift 任务数据
uv run python scripts/collect_motion_planning_data.py \
    -e LiftCubeSO101-v1 \
    -n 500 \
    --output-dir ./datasets \
    --seed 42

# 收集 stack 任务数据
uv run python scripts/collect_motion_planning_data.py \
    -e StackCubeSO101-v1 \
    -n 500 \
    --output-dir ./datasets \
    --seed 42

# 收集 sort 任务数据
uv run python scripts/collect_motion_planning_data.py \
    -e SortCubeSO101-v1 \
    -n 500 \
    --output-dir ./datasets \
    --seed 42
```

**输出**: ManiSkill 格式的轨迹文件（h5格式），保存在 `./datasets/{env_id}/motionplanning/`

## 步骤 2: 格式转换

将 ManiSkill 轨迹转换为 LeRobot Dataset 格式。

```bash
# 转换 lift 任务数据
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./datasets/LiftCubeSO101-v1/motionplanning \
    --output-dir ./datasets/lift \
    --task-name lift \
    --apply-distortion

# 转换 stack 任务数据
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./datasets/StackCubeSO101-v1/motionplanning \
    --output-dir ./datasets/stack \
    --task-name stack \
    --apply-distortion

# 转换 sort 任务数据
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./datasets/SortCubeSO101-v1/motionplanning \
    --output-dir ./datasets/sort \
    --task-name sort \
    --apply-distortion
```

**注意**: 
- `--apply-distortion` 会对前向相机图像应用畸变，匹配真机相机特性
- 转换脚本会提取 joint positions 作为 actions（不是 end-effector poses）

## 步骤 3: 训练

### 3.1 安装 LeRobot

首先需要安装 LeRobot 库（只需要模型类，不需要训练命令）：

```bash
# 使用 uv 安装
uv add lerobot

# 或者使用 pip
pip install lerobot
```

**注意**: LeRobot 主要用于加载 Diffusion Policy 模型类，我们使用自定义训练循环，不需要 LeRobot 的训练命令。

### 3.2 训练 Diffusion Policy

使用自定义训练脚本直接加载 `.npz` 文件进行训练：

```bash
# 训练 lift 任务策略
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./datasets/lift \
    --output-dir ./checkpoints/lift \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10

# 训练 stack 任务策略
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./datasets/stack \
    --output-dir ./checkpoints/stack \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10

# 训练 sort 任务策略
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./datasets/sort \
    --output-dir ./checkpoints/sort \
    --num-epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10
```

**训练参数说明**:
- `--dataset-dir`: 包含 `.npz` 文件的目录（步骤2的输出）
- `--output-dir`: 保存模型 checkpoints 的目录
- `--num-epochs`: 训练轮数（默认100）
- `--batch-size`: 批次大小（默认32，根据GPU内存调整）
- `--learning-rate`: 学习率（默认1e-4）
- `--device`: 设备（cuda 或 cpu）
- `--save-freq`: 每N个epoch保存一次checkpoint（默认10）
- `--num-workers`: 数据加载线程数（默认4）
- `--no-wrist-cameras`: 不使用手腕相机（仅使用前向相机）

**训练输出**:
- Checkpoints 保存在 `--output-dir` 目录下
- 每个 checkpoint 包含模型权重和训练信息
- `checkpoint-best` 目录保存最佳模型（loss最低）

## 步骤 4: 评估

### 仿真评估

```bash
# 评估 lift 任务策略
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/lift/checkpoint-best \
    -e LiftCubeSO101-v1 \
    -n 50 \
    --device cuda

# 评估 stack 任务策略
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/stack/checkpoint-best \
    -e StackCubeSO101-v1 \
    -n 50 \
    --device cuda

# 评估 sort 任务策略
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/sort/checkpoint-best \
    -e SortCubeSO101-v1 \
    -n 50 \
    --device cuda \
    --robot-type bi_so101
```

**目标**: 成功率 > 50% (c1 + c2 + c3 = 3 分)

### 真机评估

```bash
# 评估策略（需要配置机器人接口）
uv run python scripts/eval_real_policy.py \
    --policy-path ./checkpoints/lift/checkpoint-best \
    --camera-config-path configs/so101.json \
    --task "Pick up the red cube and lift it." \
    -n 10 \
    --device cuda
```

**目标**: 成功率 > 30% (c1 + c2 + c3 = 3 分)

## 关键配置说明

### 相机配置

- **前向相机**: 480×640分辨率，50°垂直FoV，已配置在环境类中
- **手腕相机**: 需要附加到机器人的 `camera_link`，当前为占位符实现
- **畸变处理**: 在转换阶段应用，使用PDF提供的畸变参数

### Action 格式

- **格式**: Joint positions (qpos)，不是 end-effector poses
- **维度**: 
  - 单臂任务 (lift, stack): 6维（6个关节）
  - 双臂任务 (sort): 12维（6+6个关节）

### 任务 Prompt

- `LiftCubeSO101-v1`: "Pick up the red cube and lift it."
- `StackCubeSO101-v1`: "Stack the red cube on top of the green cube."
- `SortCubeSO101-v1`: "Move the red cube to the left region and the green cube to the right region."

## 故障排除

### 数据收集失败

- 检查 motion planning 解决方案是否正常工作
- 确保环境配置正确（相机、机器人位置等）
- 检查随机种子是否导致某些配置不可达

### 训练失败

- 检查 `.npz` 文件是否存在且格式正确
- 确保 GPU 内存足够（可以减小 `--batch-size`）
- 检查 LeRobot 是否正确安装：`python -c "import lerobot; print(lerobot.__version__)"`
- 如果遇到 CUDA 内存不足，减小 batch_size 或使用 CPU 训练（`--device cpu`）

### 评估成功率低

- 增加训练数据量（500-1000 episodes）
- 调整模型容量（增加 hidden_dim, num_layers）
- 调整超参数（learning rate, action chunk size）
- 检查数据质量（动作是否平滑，图像是否清晰）

## 优势说明

### 为什么使用自定义训练脚本？

1. **简单直接**: 直接加载 `.npz` 文件，无需复杂的数据格式转换
2. **易于调试**: 训练循环完全可控，方便添加日志和调试
3. **真机部署不受影响**: 真机部署只需要模型权重和推理代码，与训练方式无关
4. **成功率相同**: 使用相同的 LeRobot Diffusion Policy 模型，成功率主要取决于数据和超参数

### 与 LeRobot 训练命令的对比

| 特性 | 自定义训练脚本 | LeRobot 训练命令 |
|------|--------------|-----------------|
| 数据格式 | `.npz` 文件 | Parquet + 元数据 |
| 训练循环 | 自定义，简单 | 内置，功能完整 |
| 调试难度 | 容易 | 较难 |
| 真机部署 | 相同 | 相同 |
| 成功率 | 相同 | 相同 |

## 下一步优化

1. **实现手腕相机渲染**: 当前为占位符，需要实现实际的相机渲染
2. **Domain Randomization**: 在数据收集中添加光照、纹理等随机化
3. **Action Chunking 优化**: 调整 action chunk size 以平衡延迟和性能
4. **超参数调优**: 根据验证集调整学习率、batch size 等超参数
