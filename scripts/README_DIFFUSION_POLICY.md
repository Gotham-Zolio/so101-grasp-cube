# Diffusion Policy 训练和部署指南

本指南说明如何使用 motion planning 数据训练 Diffusion Policy，并在仿真和真机上部署。

## 工作流程概览

```
1. 数据收集 (Motion Planning) 
   → 1.1 验证数据质量（包含图像）
   → 2. 格式转换 (ManiSkill → LeRobot)
   → 3. 训练 (自定义训练脚本)
   → 4. 评估 (仿真/真机)
```

## 快速开始

以 lift 任务为例，完整流程：

```bash
# 步骤1: 收集数据（50个episodes，约5-10分钟）
uv run python scripts/collect_motion_planning_data.py \
    --env-id LiftCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# 步骤1.1: 验证数据（检查图像是否有效）
# 注意：文件名是时间戳格式，使用最新的文件或指定实际文件名
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_data_visualization.png \
    --no-display

# 步骤2: 转换数据格式（约1-2分钟）
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./demos/LiftCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets/lift_cube \
    --task-name lift \
    --apply-distortion

# 步骤3: 训练模型（约1-3小时，取决于GPU）
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./lerobot_datasets/lift_cube \
    --output-dir ./checkpoints/lift \
    --num-epochs 200 \
    --batch-size 16 \
    --device cuda

# 步骤4: 评估模型（约5-10分钟）
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/lift/checkpoint-best \
    -e LiftCubeSO101-v1 \
    -n 50 \
    --device cuda \
    --save-camera \
    --verbose
```

**预期结果**：
- 数据收集：生成包含有效图像的HDF5文件
- 数据验证：图像平均亮度 > 50（非全黑）
- 训练：loss逐渐下降，模型收敛
- 评估：成功率 > 50%（目标）

## 步骤 1: 数据收集

使用 motion planning 解决方案生成成功的轨迹数据。**重要**：确保数据包含有效的相机图像（sensor_data）。

```bash
# 收集 lift 任务数据（推荐50-500个episodes）
uv run python scripts/collect_motion_planning_data.py \
    --env-id LiftCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# 收集 stack 任务数据
uv run python scripts/collect_motion_planning_data.py \
    --env-id StackCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos

# 收集 sort 任务数据
uv run python scripts/collect_motion_planning_data.py \
    --env-id SortCubeSO101-v1 \
    --num-episodes 50 \
    --output-dir ./demos
```

**输出**: ManiSkill 格式的轨迹文件（h5格式），保存在 `./demos/{env_id}/motionplanning/`

**文件命名**：
- 文件使用时间戳格式命名（如 `20260115_063449.h5`），不是固定的 `lift_demo.h5`
- 可以使用 `ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1` 获取最新文件
- 或者直接查看目录：`ls demos/LiftCubeSO101-v1/motionplanning/`

**重要提示**：
- 数据收集脚本会自动验证 `sensor_data` 是否可用
- 如果看到 "✓ sensor_data is available in get_obs()"，说明数据收集正常
- 如果看到 "⚠️ WARNING: sensor_data NOT in get_obs()!"，需要检查环境配置

### 步骤 1.1: 验证数据质量

在训练前，务必验证收集的数据包含有效的图像：

```bash
# 可视化训练数据图像（检查图像是否有效）
# 注意：数据收集会生成时间戳格式的文件名（如 20260115_063449.h5）
# 使用最新的文件或指定实际文件名
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_data_visualization.png \
    --no-display

# 或者直接指定文件名（替换为实际的文件名）
# uv run python scripts/visualize_training_data_images.py \
#     --hdf5-path ./demos/LiftCubeSO101-v1/motionplanning/20260115_063449.h5 \
#     --max-images 20 \
#     --mode static \
#     --save-static ./training_data_visualization.png \
#     --no-display
```

**验证标准**：
- ✓ obs组包含sensor_data
- ✓ sensor_data包含front/rgb
- ✓ 图像平均亮度 > 50（非全黑）
- ✓ 图像形状正确 (480, 640, 3)

如果图像平均亮度 < 5.0，说明图像是全黑的，需要重新收集数据。

## 步骤 2: 格式转换

将 ManiSkill 轨迹转换为 LeRobot Dataset 格式（.npz文件）。

```bash
# 转换 lift 任务数据
# 注意：脚本期望输入目录（包含所有h5文件），不是单个文件
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./demos/LiftCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets/lift_cube \
    --task-name lift \
    --apply-distortion

# 转换 stack 任务数据
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./demos/StackCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets/stack_cube \
    --task-name stack \
    --apply-distortion

# 转换 sort 任务数据
uv run python scripts/convert_trajectory_to_lerobot.py \
    --input-dir ./demos/SortCubeSO101-v1/motionplanning \
    --output-dir ./lerobot_datasets/sort_cube \
    --task-name sort \
    --apply-distortion
```

**注意**: 
- `--input-dir` 参数需要传入包含 h5 文件的目录（如 `./demos/LiftCubeSO101-v1/motionplanning`），不是单个文件
- 脚本会自动处理目录中的所有 `.h5` 文件
- `--output-dir` 参数指定输出目录（不是 `--output`）
- `--apply-distortion` 会对前向相机图像应用畸变，匹配真机相机特性
- 转换脚本会提取 joint positions 作为 actions（不是 end-effector poses）
- 输出目录会包含多个 `episode_*.npz` 文件，每个文件对应一个轨迹

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
# 训练 lift 任务策略（使用默认GPU 0）
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./lerobot_datasets/lift_cube \
    --output-dir ./checkpoints/lift \
    --num-epochs 200 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10

# 指定使用特定的GPU（例如GPU 5）
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./lerobot_datasets/lift_cube \
    --output-dir ./checkpoints/lift \
    --num-epochs 200 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda:5 \
    --save-freq 10

# 训练 stack 任务策略
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./lerobot_datasets/stack_cube \
    --output-dir ./checkpoints/stack \
    --num-epochs 200 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10

# 训练 sort 任务策略
uv run python scripts/train_diffusion_policy_custom.py \
    --dataset-dir ./lerobot_datasets/sort_cube \
    --output-dir ./checkpoints/sort \
    --num-epochs 200 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --save-freq 10
```

**训练参数说明**:
- `--dataset-dir`: 包含 `.npz` 文件的目录（步骤2的输出）
- `--output-dir`: 保存模型 checkpoints 的目录
- `--num-epochs`: 训练轮数（推荐200，默认100）
- `--batch-size`: 批次大小（推荐16，根据GPU内存调整，默认32）
- `--learning-rate`: 学习率（默认1e-4）
- `--device`: 设备（`cuda` 使用默认GPU 0，`cuda:5` 使用GPU 5，`cpu` 使用CPU）
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
# 评估 lift 任务策略（保存相机视频用于分析）
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/lift/checkpoint-best \
    -e LiftCubeSO101-v1 \
    -n 50 \
    --device cuda \
    --save-camera \
    --verbose

# 评估 stack 任务策略
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/stack/checkpoint-best \
    -e StackCubeSO101-v1 \
    -n 50 \
    --device cuda \
    --save-camera

# 评估 sort 任务策略
uv run python scripts/eval_sim_policy.py \
    --policy-path ./checkpoints/sort/checkpoint-best \
    -e SortCubeSO101-v1 \
    -n 50 \
    --device cuda \
    --robot-type bi_so101 \
    --save-camera
```

**评估参数说明**:
- `--save-camera`: 保存相机视角视频到 `./eval_results/camera_videos/`
- `--save-video`: 保存渲染视角视频到 `./eval_results/videos/`
- `--save-failed-only`: 只保存失败的episode视频
- `--verbose`: 显示详细的调试信息

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

## 数据验证工具

### 可视化训练数据图像

在训练前，强烈建议可视化数据以确认图像质量：

```bash
# 静态可视化（网格显示多个图像）
# 使用最新生成的文件
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 20 \
    --mode static \
    --save-static ./training_images.png \
    --no-display

# 动画可视化（视频形式，更直观）
LATEST_H5=$(ls -t demos/LiftCubeSO101-v1/motionplanning/*.h5 | head -1)
uv run python scripts/visualize_training_data_images.py \
    --hdf5-path "$LATEST_H5" \
    --max-images 50 \
    --mode animation \
    --save-animation ./training_images.mp4 \
    --fps 10 \
    --no-display
```

**输出说明**：
- 静态可视化：保存为PNG图片，显示多个时间步的图像网格
- 动画可视化：保存为MP4视频，可以查看图像序列
- 图像统计：显示平均亮度、最大值、最小值等信息

**判断标准**：
- ✓ 图像平均亮度 > 50：图像质量正常
- ⚠️ 图像平均亮度 5-50：图像较暗，可能影响训练
- ✗ 图像平均亮度 < 5：图像全黑，**必须重新收集数据**

## 关键配置说明

### 相机配置

- **前向相机**: 480×640分辨率，50°垂直FoV，已配置在环境类中
- **手腕相机**: 需要附加到机器人的 `camera_link`，当前为占位符实现
- **畸变处理**: 在转换阶段应用，使用PDF提供的畸变参数
- **数据收集**: 使用 `obs_mode="rgb"`，确保sensor_data被正确保存

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

### 数据收集问题

**问题：obs组为空，没有sensor_data**
- **原因**：环境配置问题，sensor_data没有被保存
- **解决**：
  1. 确保使用 `obs_mode="rgb"`（脚本已自动设置）
  2. 检查数据收集时的调试输出，确认 "✓ sensor_data is available"
  3. 如果仍然没有sensor_data，检查ManiSkill版本和环境配置

**问题：图像全黑（平均亮度 < 5.0）**
- **原因**：场景渲染问题或相机配置错误
- **解决**：
  1. 检查相机配置是否正确
  2. 确保场景中有物体和光照
  3. 重新收集数据

### 训练失败

- 检查 `.npz` 文件是否存在且格式正确
- 确保 GPU 内存足够（可以减小 `--batch-size`）
- 检查 LeRobot 是否正确安装：`python -c "import lerobot; print(lerobot.__version__)"`
- 如果遇到 CUDA 内存不足，减小 batch_size 或使用 CPU 训练（`--device cpu`）

### 评估成功率低

- **检查数据质量**：使用可视化工具确认图像有效
- **增加训练数据量**：50-500个episodes（推荐200+）
- **调整模型容量**：增加 hidden_dim, num_layers
- **调整超参数**：learning rate, action chunk size
- **检查动作质量**：动作是否平滑，图像是否清晰

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
