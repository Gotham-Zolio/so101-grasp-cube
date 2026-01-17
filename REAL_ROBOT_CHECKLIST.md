# 真机部署快速检查清单

**状态**: 基于项目要求文档的具体实施步骤  
**最后更新**: 2026-01-17

---

## 📋 当前完成情况

### ✅ 已完成（您现在可用的）

- [x] DiffusionPolicyInferenceEngine - 离线推理引擎
- [x] RealRobotDiffusionInferenceWrapper - 推理包装器  
- [x] serve_diffusion_policy.py - **WebSocket 服务器（新建）**
- [x] 详细的部署指南文档

### ⏳ 立即需要做的

- [ ] **测试** - Server-Client 本地测试（模拟环境）
- [ ] **配置** - 真机参数配置（摄像头、机械臂 IP）
- [ ] **Docker** - 镜像打包和测试
- [ ] **验收** - 真机上的实际任务执行

---

## 🚀 零开始快速步骤

### Step 1: 验证 Server 能启动（5分钟）

```bash
cd /path/to/so101-grasp-cube

# 查看 Server 代码
cat grasp_cube/real/serve_diffusion_policy.py

# 尝试启动 Server（需要模型文件）
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --device cuda

# 期望输出：
# Creating DiffusionPolicy WebSocket Server
# ✓ Initializing LeRobotDiffusionPolicy...
# ✓ Server created successfully
# Waiting for client connections at ws://0.0.0.0:8000
```

**问题排查**：
- 如果报 "File not found"：checkpoints/ 目录可能不存在或路径错误
- 如果报 "module not found"：确保 env_client 已安装 (`uv pip install -e packages/env-client`)
- 如果报 GPU 错误：检查 CUDA 可用性，或用 `--device cpu`

### Step 2: 本地集成测试（15分钟）

在两个不同的终端中：

**终端 A（Server）**：
```bash
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift
```

**终端 B（Fake Client）**：
```bash
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift
```

**期望**：
- 终端 A 显示 "Waiting for client connections"
- 终端 B 显示 "Connected!" 
- 浏览器 http://localhost:9000 显示可视化

**常见问题**：
- Client 显示 "Connection refused"：Server 没有成功启动
- Client 显示 "Invalid observation format"：数据格式不匹配
- 查看 REAL_ROBOT_DEPLOYMENT.md 的"常见问题解决"部分

### Step 3: 理解关键接口（10分钟）

```python
# Server 端（serve_diffusion_policy.py）实现了这个接口：
class LeRobotDiffusionPolicy:
    def get_actions(self, observation: dict) -> np.ndarray:
        """
        输入：observation = {
            "images": {"front": (480,640,3) uint8, ...},
            "states": {"arm": (6,) float32}  或  {"left_arm": (6,), "right_arm": (6,)}
        }
        
        输出：actions = (16, 6) 或 (16, 12) 动作序列
        """
        # 内部调用 RealRobotDiffusionInferenceWrapper.predict_from_obs()
        return self.inference_wrapper.predict_from_obs(observation)

# Client 端（run_env_client.py）使用这个接口：
from env_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", 8000)
obs, info = env.reset()
action_chunk = client.infer(obs)["action"]  # 调用上面的 get_actions()
# 然后逐步执行 action_chunk 中的动作
```

### Step 4: 部署到真机（取决于硬件准备）

一旦真机硬件（摄像头、机械臂、网络）就绪：

**真机端启动 Server**：
```bash
# Server 运行在有 GPU 的机器上（可以是同一台真机或远程）
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --host 0.0.0.0  \
    --port 8000     \
    --device cuda
```

**真机环境启动 Client**：
```bash
# Client 运行在真机上，连接到真实的机械臂和摄像头
uv run python grasp_cube/real/run_env_client.py \
    --env.robot.hostname 192.168.1.100  \
    --env.camera-config-path configs/camera_config.json \
    --env.robot so101-follower-config \
    --task lift \
    --num-episodes 10
```

---

## 📦 Docker 打包（提交前必须）

### 创建必要的文件

#### 1. docker_entrypoint.sh
```bash
#!/bin/bash
TASK=${TASK:-lift}
MODEL_PATH=${MODEL_PATH:-checkpoints/lift_real/checkpoint-best}
DEVICE=${DEVICE:-cuda}

python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path "$MODEL_PATH" \
    --policy.task "$TASK" \
    --host 0.0.0.0 \
    --port 8000 \
    --device "$DEVICE"
```

#### 2. Dockerfile.diffusion
```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04
WORKDIR /app
COPY . /app
RUN uv pip install -e . && \
    uv pip install -e packages/env-client && \
    cd external/lerobot && uv pip install -e .
COPY docker_entrypoint.sh /app/
RUN chmod +x /app/docker_entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/docker_entrypoint.sh"]
```

### 构建和测试

```bash
# 构建
docker build -f Dockerfile.diffusion -t so101-diffusion:v1 .

# 测试运行
docker run -it --gpus all \
    -p 8000:8000 \
    -e TASK=lift \
    so101-diffusion:v1

# 在另一个终端测试 Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift
```

---

## 🎯 三个任务的快速启动命令

### Lift Task（单臂，简单）

```bash
# Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Client (Fake)
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift

# Client (Real)
uv run python grasp_cube/real/run_env_client.py \
    --task lift
```

### Sort Task（双臂，中等）

```bash
# Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/sort_real/checkpoint-best \
    --policy.task sort

# Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/sort
```

### Stack Task（双臂，复杂）

```bash
# Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/stack_real/checkpoint-best \
    --policy.task stack

# Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/stack
```

---

## 🔍 故障排查

### 问题：Server 无法找到模型文件

```
FileNotFoundError: checkpoints/lift_real/checkpoint-best/pytorch_model.bin
```

**解决**：
1. 检查模型路径是否正确：`ls checkpoints/*/checkpoint-best/`
2. 确保从项目根目录运行
3. 使用绝对路径：`--policy.path /full/path/to/checkpoints/lift_real/checkpoint-best`

### 问题：Client 无法连接到 Server

```
Connection refused at ws://0.0.0.0:8000
```

**解决**：
1. 确保 Server 在运行：查看 Terminal A 的输出
2. 检查端口是否被占用：`netstat -tuln | grep 8000`
3. 如果跨机器：检查 IP 和防火墙

### 问题：推理输出维度错误

```
ValueError: Action dim mismatch
```

**解决**：
1. 检查任务是否正确：lift/sort 的 action_dim 不同
2. 查看 inference_engine.py 的输出形状
3. 检查 stats.json 是否完整

### 问题：GPU 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**：
1. 尝试 `--device cpu` 运行（会慢但能工作）
2. 减少 batch size（如果有的话）
3. 升级 GPU 或使用模型量化

---

## ✅ 最终提交检查清单

在提交项目前，确保以下全部通过：

### 功能性测试
- [ ] Server 能正常启动
- [ ] Fake Client 能连接到 Server
- [ ] 三个任务都能推理（lift/sort/stack）
- [ ] 推理输出格式正确（(horizon, action_dim)）
- [ ] 动作值在 [-1, 1] 范围内

### 性能测试
- [ ] 推理延迟 < 2 秒/次
- [ ] GPU 内存占用 < 3GB
- [ ] 没有内存泄漏（长时间运行）

### 真机测试（如硬件就绪）
- [ ] 摄像头能正常读取
- [ ] 机械臂能正常执行动作
- [ ] 至少 1 个任务在真机上成功执行
- [ ] 没有碰撞或异常

### Docker 测试
- [ ] Docker 镜像能成功构建
- [ ] 容器能正常启动
- [ ] 容器内的 Server 能被连接
- [ ] 镜像大小合理 (<5GB)

### 文档完整性
- [ ] README 包含使用说明
- [ ] 有 Docker 启动命令示例
- [ ] 有 Server/Client 命令示例
- [ ] 有故障排查部分

---

## 📞 快速查询

| 我想... | 看这里 |
|---------|--------|
| 启动 Server | 本文 "快速步骤 Step 1" |
| 测试 Server-Client | 本文 "快速步骤 Step 2" |
| 理解数据格式 | REAL_ROBOT_DEPLOYMENT.md "观测和动作格式" |
| 解决常见问题 | REAL_ROBOT_DEPLOYMENT.md "常见问题解决" |
| 部署到真机 | REAL_ROBOT_DEPLOYMENT.md "步骤 5" |
| 打包 Docker 镜像 | 本文 "Docker 打包" 部分 |
| 查看原始 ACT 实现 | grasp_cube/real/act_policy.py 和 serve_act_policy.py |

---

## 📚 核心文件导览

```
项目根目录/
├── grasp_cube/real/
│   ├── serve_diffusion_policy.py      ← ✨ 新的 Server 入口
│   ├── diffusion_inference_wrapper.py  ← 推理包装器
│   ├── act_policy.py                  ← 参考：ACT 政策
│   ├── serve_act_policy.py            ← 参考：ACT 服务器
│   ├── lerobot_env.py                 ← 真机环境定义
│   ├── run_env_client.py              ← Client 启动脚本
│   └── run_fake_env_client.py         ← Fake Client 启动脚本
│
├── scripts/
│   ├── inference_engine.py             ← 推理引擎核心
│   ├── test_offline_inference.py       ← 离线验证（已通过）
│   └── test_real_sensor_input.py       ← 真机推理验证
│
├── Dockerfile.diffusion                ← Docker 镜像定义
├── docker_entrypoint.sh                ← Docker 启动脚本
│
└── REAL_ROBOT_DEPLOYMENT.md            ← 完整部署指南（本文档）
```

---

## 🎬 立即行动

现在就可以执行：

```bash
# 1. 启动 Server（Terminal 1）
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# 2. 启动 Client（Terminal 2）
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift

# 3. 打开浏览器
# http://localhost:9000

# 预期：看到机械臂的模拟环境和推理的实时动作
```

如果上面能成功运行，说明整个 server-client 架构已经工作正常！✅

