# 真机部署完整集成指南（按照项目要求）

## 📋 核心需求回顾

项目要求使用 **server-client 架构**：

```
Client（真机环境或模拟环境）
  ↓ WebSocket ws://0.0.0.0:8000
Server（推理服务）
  ↓
DiffusionPolicyInferenceEngine
  ↓
Robot Actions
```

**优势**：完全解耦模型和环境，防止紧耦合风险

---

## 🎯 现在需要做的具体步骤（详细版）

### 步骤1：理解现有的 Server-Client 架构

#### 1.1 查看现有的 ACT Policy 实现（参考）

```bash
# 查看 ACT 政策的服务器实现
cat grasp_cube/real/act_policy.py        # LeRobotACTPolicy 类
cat grasp_cube/real/serve_act_policy.py  # 服务器启动脚本
```

**关键文件内容分析**：

- **act_policy.py**: 
  - `LeRobotACTPolicyConfig`: 配置类（路径、设备等）
  - `LeRobotACTPolicy`: 政策类，实现 `get_actions(observation)` 方法
  - 接收真机观测，返回动作序列

- **serve_act_policy.py**: 
  - `ActPolicyServerConfig`: 服务器配置
  - `create_act_policy_server()`: 创建 WebSocket 服务器
  - `main()`: 启动服务

#### 1.2 理解观测和动作格式

```python
# 客户端（真机环境）发送的观测格式：
observation = {
    "images": {
        "front": np.array (480, 640, 3) uint8,      # 前视摄像头
        "left_wrist": np.array (480, 640, 3) uint8,  # 左腕摄像头（可选）
        "right_wrist": np.array (480, 640, 3) uint8  # 右腕摄像头（可选）
    },
    "states": {
        # 单臂（SO101）
        "arm": np.array (6,) float32,  # 6个关节角度
        
        # 或双臂（BI-SO101）
        "left_arm": np.array (6,) float32,
        "right_arm": np.array (6,) float32
    }
}

# 服务器（政策）返回的动作格式：
actions = np.array (horizon, action_dim) float32
# horizon: 通常为 16（从模型的 config.json 中定义）
# action_dim: 6（单臂）或 12（双臂）
# 范围: [-1, 1]
```

#### 1.3 理解 env_client 库的接口

```python
# env_client 库提供的 WebsocketPolicyServer 类
from env_client import websocket_policy_server

# 创建服务器的方式：
server = websocket_policy_server.WebsocketPolicyServer(
    policy=policy_instance,        # 必须有 get_actions(obs) 方法
    host="0.0.0.0",               # 监听地址
    port=8000,                     # 端口
    metadata={...}                 # 元数据（可选）
)

# 启动服务
server.serve_forever()  # 阻塞运行直到 Ctrl+C
```

---

### 步骤2：已完成 - DiffusionPolicy Server 实现

✅ **已为您创建**: `grasp_cube/real/serve_diffusion_policy.py` (220行)

**文件内容**：
- `LeRobotDiffusionPolicyConfig`: DiffusionPolicy 配置类
- `LeRobotDiffusionPolicy`: 推理政策类
  - 初始化 `RealRobotDiffusionInferenceWrapper`
  - 实现 `get_actions(observation)` 方法
  - 返回完整的动作序列 (16, 6或12)
- `DiffusionPolicyServerConfig`: 服务器配置
- `create_diffusion_policy_server()`: 创建服务器
- `main()`: 命令行启动

---

### 步骤3：测试 Server-Client 集成

#### 3.1 第一次测试：启动 Server

在**Server 端**（模型所在的机器）:

```bash
cd /path/to/so101-grasp-cube

# 安装 env_client（如果还没装）
uv pip install -e packages/env-client

# 启动 DiffusionPolicy 服务器
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda

# 输出应该是：
# ========================================
# Creating DiffusionPolicy WebSocket Server
# ========================================
# ✓ Initializing LeRobotDiffusionPolicy
#   Task: lift
#   Model path: checkpoints/lift_real/checkpoint-best
#   Device: cuda
# ✓ Inference wrapper initialized
# ✓ Server created successfully
#   Host: 0.0.0.0
#   Port: 8000
#   ...
# Starting DiffusionPolicy Policy Server...
# Waiting for client connections at ws://0.0.0.0:8000
```

**关键点**：
- 服务器启动后会 **持续监听** 端口 8000
- 不会自动退出，等待客户端连接
- 按 Ctrl+C 可以优雅关闭

#### 3.2 第二次测试：启动 Fake Client（模拟环境）

在**Client 端**（另一个终端，可以是同一机器的不同环境）:

```bash
cd /path/to/so101-grasp-cube

# 安装依赖（如果还没装）
uv pip install -e packages/env-client
# 或者在 LeRobot 环境中安装
# pip install -e packages/env-client

# 启动模拟客户端
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift

# 输出应该是：
# [MonitorWrapper] Panel: http://0.0.0.0:9000
# [EvalRecordWrapper] Output dir: outputs/eval_records/20251226_124302
# Waiting for server at ws://0.0.0.0:8000...
# Connected!  [如果 server 启动了]
# Episode 1: Running...
```

**关键点**：
- Client 会尝试连接到 `ws://0.0.0.0:8000`（默认）
- 如果 server 没运行，会报 "Connection refused"
- 连接成功后会开始播放数据集中的轨迹，并调用 policy.get_actions()

#### 3.3 验证 Server-Client 通信

打开浏览器访问：**http://0.0.0.0:9000**

**应该看到**：
- 机械臂和环境的可视化
- "Start/Stop" 按钮来控制 episode
- 视频回放（前视摄像头）
- 动作对比（如果已配置）

**如果看到错误**：

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| Connection refused | Server 没启动 | 检查 server 端，确保服务器在运行 |
| Timeout | 网络问题 | 确保两端的 IP/端口配置正确 |
| Invalid observation format | 观测格式不匹配 | 检查 observation 中的 keys |
| Policy inference error | 推理引擎出错 | 查看 server 的错误日志 |

---

### 步骤4：真机部署准备（不执行，只检查）

#### 4.1 了解真机环境设置

```bash
# 查看真机环境的配置
cat grasp_cube/real/lerobot_env.py  # LeRobotEnv 类定义

# 关键参数：
# - robot: SO101FollowerConfig 或 BiSO101FollowerConfig
# - camera_config_path: 摄像头配置 JSON
# - task: 任务名称
# - episode_time_s: 最长运行时间
# - fps: 控制频率 (通常 30 Hz)
# - image_resolution: (480, 640)
```

#### 4.2 配置摄像头（准备工作）

真机需要的摄像头配置文件（camera_config.json）:

```json
{
  "front": {
    "type": "realsense",
    "camera_name": "camera_front",
    "color_resolution": [640, 480],
    "depth_resolution": [640, 480],
    "rgb_topic": "/camera_front/color/image_raw",
    "depth_topic": "/camera_front/depth/image_rect_raw"
  },
  "left_wrist": {
    "type": "realsense",
    "camera_name": "camera_left_wrist",
    "color_resolution": [640, 480],
    "depth_resolution": [640, 480]
  },
  "right_wrist": {
    "type": "realsense",
    "camera_name": "camera_right_wrist",
    "color_resolution": [640, 480],
    "depth_resolution": [640, 480]
  }
}
```

#### 4.3 准备真机的 Robot Config

```python
# 单臂（SO101）配置
from lerobot.robots import so101_follower

robot_config = so101_follower.SO101FollowerConfig(
    hostname="192.168.1.100",  # 真机 IP
    port=12345,                 # 控制端口
    # ... 其他参数
)

# 或双臂（BI-SO101）
from lerobot.robots import bi_so101_follower

robot_config = bi_so101_follower.BiSO101FollowerConfig(
    hostname="192.168.1.100",
    port=12345,
    # ... 其他参数
)
```

---

### 步骤5：部署到真机（实际执行）

#### 5.1 在真机上启动 Server

```bash
# 在真机或具有GPU的服务器上
cd /path/to/so101-grasp-cube

# 根据任务启动对应的服务器
# Lift 任务
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --device cuda

# 或 Sort 任务（双臂）
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/sort_real/checkpoint-best \
    --policy.task sort \
    --device cuda

# 或 Stack 任务
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/stack_real/checkpoint-best \
    --policy.task stack \
    --device cuda
```

#### 5.2 在真机环境上启动 Client

```bash
# 在真机的环境上
cd /path/to/so101-grasp-cube

# 启动真机客户端（NOT fake 版本）
uv run python grasp_cube/real/run_env_client.py \
    --env.robot.hostname 192.168.1.100 \
    --env.camera-config-path configs/camera_config.json \
    --env.robot so101-follower-config \
    --task lift \
    --num-episodes 10

# 或者使用配置文件方式（更推荐）
uv run python grasp_cube/real/run_env_client.py \
    --config configs/lift_deploy.yaml
```

#### 5.3 监控运行

浏览器访问：**http://robot_ip:9000**

实时观看：
- 机械臂执行
- 摄像头反馈
- 任务进度

---

### 步骤6：Docker 打包（最后交付）

#### 6.1 创建 Dockerfile（参考 docker_tutorial.md）

```dockerfile
# Dockerfile.diffusion

FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制代码
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir uv
RUN uv pip install -e .
RUN uv pip install -e packages/env-client

# 安装 LeRobot（如果需要）
RUN cd external/lerobot && uv pip install -e .

# 暴露端口
EXPOSE 8000

# 启动脚本
COPY docker_entrypoint.sh /app/docker_entrypoint.sh
RUN chmod +x /app/docker_entrypoint.sh

ENTRYPOINT ["/app/docker_entrypoint.sh"]
```

#### 6.2 创建启动脚本

```bash
# docker_entrypoint.sh

#!/bin/bash

# 从环境变量读取配置
TASK=${TASK:-lift}
MODEL_PATH=${MODEL_PATH:-checkpoints/lift_real/checkpoint-best}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-cuda}

echo "Starting DiffusionPolicy Server"
echo "  Task: $TASK"
echo "  Model: $MODEL_PATH"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Device: $DEVICE"

# 启动服务器
python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path "$MODEL_PATH" \
    --policy.task "$TASK" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"
```

#### 6.3 构建和运行 Docker 镜像

```bash
# 构建镜像
docker build -f Dockerfile.diffusion -t so101-diffusion-policy:latest .

# 运行容器
docker run -it --gpus all \
    -p 8000:8000 \
    -e TASK=lift \
    -e DEVICE=cuda \
    so101-diffusion-policy:latest

# 或使用 docker-compose
docker-compose -f docker-compose.diffusion.yml up
```

---

## 📊 完整的文件清单

### ✅ 已创建的文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `grasp_cube/real/serve_diffusion_policy.py` | Server 主程序 | 220 |
| `grasp_cube/real/diffusion_inference_wrapper.py` | 推理包装器 | 415 |
| `scripts/inference_engine.py` | 推理引擎 | 401 |

### ⏳ 需要创建的文件

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `docker_entrypoint.sh` | Docker 启动脚本 | 高 |
| `Dockerfile.diffusion` | Docker 镜像定义 | 高 |
| `docker-compose.diffusion.yml` | Docker Compose 配置 | 中 |
| `configs/lift_deploy.yaml` | Lift 任务配置文件 | 中 |
| `configs/sort_deploy.yaml` | Sort 任务配置文件 | 中 |
| `configs/stack_deploy.yaml` | Stack 任务配置文件 | 中 |
| `tests/test_server_client.py` | Server-Client 集成测试 | 中 |

---

## 🧪 测试流程（完整版）

### Phase 1: 本地测试（模拟环境）

```bash
# Terminal 1: 启动 Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Terminal 2: 启动 Fake Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift

# 预期结果：
# - Server 显示 "Waiting for client connections"
# - Client 显示 "Connected!" 然后开始推理
# - 浏览器显示可视化
```

### Phase 2: 真机测试（实际硬件）

```bash
# 在真机上执行相同的命令，但使用真实的摄像头和机械臂数据
# 需要确保：
# 1. 摄像头正确连接和配置
# 2. 机械臂通信正常
# 3. 安全工作空间已清空
```

### Phase 3: Docker 测试

```bash
# 构建镜像
docker build -f Dockerfile.diffusion -t so101-diffusion:latest .

# 运行 Server
docker run -it --gpus all \
    -p 8000:8000 \
    -e TASK=lift \
    so101-diffusion:latest

# 从主机连接 Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift
```

---

## 📝 常见问题解决

### Q1: Server 启动后，Client 连接失败

**症状**: `Connection refused at ws://0.0.0.0:8000`

**解决**:
1. 检查 Server 是否真的在运行：`ps aux | grep serve_diffusion`
2. 检查端口是否被占用：`netstat -tuln | grep 8000`
3. 检查防火墙设置：`sudo ufw status`

### Q2: 推理延迟太高

**症状**: Server 推理每次需要 1-2 秒

**解决**:
1. 检查 GPU 是否可用：`nvidia-smi`
2. 检查 GPU 内存：模型应该 <2GB
3. 考虑使用 FP16 混合精度（修改 inference_engine.py）

### Q3: 观测格式错误

**症状**: `ValueError: Observation missing 'states' key`

**解决**:
1. 检查 run_fake_env_client.py 或 run_env_client.py 发送的观测格式
2. 确保有 "images" 和 "states" 两个 key
3. 查看 lerobot_env.py 的 prepare_observation() 方法

### Q4: 任务维度不匹配

**症状**: `State dim mismatch: stats has 6, actual is 12`

**解决**:
1. 确保选择了正确的任务：`--policy.task sort`（而不是 lift）
2. 确保客户端发送了正确维度的状态向量
3. Sort 任务需要 left_arm + right_arm（12维）

---

## ✅ 部署完成检查清单

在提交前，确保：

- [ ] Server 能正常启动并监听端口 8000
- [ ] Fake Client 能连接到 Server
- [ ] 推理输出格式正确（(16, 6) 或 (16, 12)）
- [ ] 推理速度可接受（<2s/次）
- [ ] 在真机上成功运行至少 1 个 episode
- [ ] Docker 镜像能成功构建和运行
- [ ] 三个任务（lift/sort/stack）都能正常工作
- [ ] 文档完整，包含使用说明
- [ ] 所有测试通过

---

## 🚀 立即开始的命令

```bash
# 1. 查看已创建的 Server 文件
cat grasp_cube/real/serve_diffusion_policy.py

# 2. 测试 Server 启动（需要模型文件）
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# 3. 在另一个终端测试 Fake Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift

# 4. 打开浏览器查看
# http://localhost:9000

# 5. 检查 Server 日志
tail -f server.log  # 如果有的话
```

---

## 📚 参考文档

- `grasp_cube/real/act_policy.py` - ACT 政策实现（参考）
- `grasp_cube/real/serve_act_policy.py` - ACT 服务器（参考）
- `grasp_cube/real/lerobot_env.py` - 真机环境定义
- `grasp_cube/real/run_env_client.py` - 客户端启动脚本
- `docker_tutorial.md` - Docker 打包指南
- `README.md` - 项目总体说明

