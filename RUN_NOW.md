# Server-Client 集成测试完全指南

## 📌 现在可以运行的完整流程

### Terminal 1: 启动 DiffusionPolicy Server

```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift
```

**输出示例**：
```
======================================================================
Creating DiffusionPolicy WebSocket Server
======================================================================
✓ Initializing LeRobotDiffusionPolicy
  Task: lift
  Model path: checkpoints/lift_real/checkpoint-best
  Device: cuda
✓ Inference wrapper initialized

✓ Server created successfully
  Host: 0.0.0.0
  Port: 8000

Waiting for client connections at ws://0.0.0.0:8000
INFO:websockets.server:server listening on 0.0.0.0:8000
```

✅ **Server 已启动，等待 Client 连接**

---

### Terminal 2: 启动 Client（模拟环境）

```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --num-episodes 2
```

**会做的事**:
1. 连接到 Server (ws://0.0.0.0:8000)
2. 启动监控 Web 界面 (http://localhost:9000)
3. 运行 2 个 episode
4. 每步获取推理结果并执行

---

### Browser: 查看监控界面

打开浏览器，访问：
```
http://localhost:9000
```

**可以看到**：
- 🎬 机械臂执行的实时视频
- 📊 每步的推理输出
- 📈 执行统计数据
- 🎯 当前任务和进度

---

## 🎯 任务特定命令

### Lift Task (6-dim 单臂)

```bash
# Terminal 1: Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Terminal 2: Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --num-episodes 2
```

### Sort Task (12-dim 双臂)

```bash
# Terminal 1: Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/sort_real/checkpoint-best \
    --policy.task sort

# Terminal 2: Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task sort \
    --num-episodes 2
```

### Stack Task (6-dim 单臂，复杂)

```bash
# Terminal 1: Server
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/stack_real/checkpoint-best \
    --policy.task stack

# Terminal 2: Client
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task stack \
    --num-episodes 2
```

---

## 🔧 高级选项

### Server 选项

```bash
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --host 0.0.0.0 \           # 绑定地址
    --port 8000 \              # 端口
    --device cuda              # cuda 或 cpu
```

### Client 选项

```bash
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --host 0.0.0.0 \           # Server 地址
    --port 8000 \              # Server 端口
    --num-episodes 2 \         # episode 数量
    --monitor-host 0.0.0.0 \   # 监控界面绑定地址
    --monitor-port 9000 \      # 监控界面端口
    --eval.output-dir outputs/ # 评估数据保存目录
```

---

## 📊 架构说明

```
┌──────────────────────────────────────────────────────────────────┐
│                    Your Workspace                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Terminal 1              WebSocket              Terminal 2        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     SERVER                               │   │
│  │  serve_diffusion_policy.py                              │   │
│  │  ├─ LeRobotDiffusionPolicy                              │   │
│  │  │  └─ RealRobotDiffusionInferenceWrapper               │   │
│  │  │     └─ DiffusionPolicyInferenceEngine                │   │
│  │  └─ WebsocketPolicyServer (ws://0.0.0.0:8000)           │   │
│  └──────────────┬───────────────────────────────────────────┘   │
│                 │                                                 │
│                 │ ← observation dict (images + states)            │
│                 │ → action sequence (16, 6) or (16, 12)          │
│                 │                                                 │
│  ┌──────────────▼───────────────────────────────────────────┐   │
│  │                     CLIENT                               │   │
│  │  run_fake_env_client.py (或 run_env_client.py 真机)     │   │
│  │  ├─ WebsocketClientPolicy (ws://0.0.0.0:8000)          │   │
│  │  ├─ SimpleFakeEnv (或 LeRobotEnv 真机)                  │   │
│  │  ├─ MonitorWrapper (http://localhost:9000)             │   │
│  │  └─ EvalRecordWrapper (记录数据)                        │   │
│  └──────────────┬───────────────────────────────────────────┘   │
│                 │                                                 │
│                 └─→ HTTP                                          │
│                     Browser: http://localhost:9000              │
│                     ├─ 实时视频                                 │
│                     ├─ 推理输出                                 │
│                     └─ 性能指标                                 │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✅ 验证检查清单

运行之前，确认：

- [ ] 已安装 `env_client`: `uv pip list | grep env-client`
- [ ] 已安装 `lerobot`: `uv pip list | grep lerobot`
- [ ] 模型文件存在: `ls checkpoints/lift_real/checkpoint-best/`
- [ ] 有两个可用的终端
- [ ] 端口 8000 和 9000 未被占用

---

## 🐛 常见问题解决

### Q: "Connection refused"
**A**: 确保 Server 在 Terminal 1 运行，检查输出有 "server listening on 0.0.0.0:8000"

### Q: "WebSocket connection failed"
**A**: 检查 Server 和 Client 的 `--host` 和 `--port` 是否一致

### Q: "Module not found"
**A**: 运行 `uv sync` 确保所有依赖安装

### Q: 监控界面显示 "no matching results"
**A**: 这是正常的，Client 需要准备数据。确保 Terminal 2 的 Client 在运行

### Q: 推理很慢
**A**: 检查 GPU 是否在使用 (`nvidia-smi`)，尝试减少图像尺寸或使用 FP16

### Q: 环境导入错误
**A**: 通常是 LeRobot 版本不兼容，我们已经修复了主要问题。如遇新问题，检查版本：
```bash
uv pip show lerobot
```

---

## 📈 性能期望

典型的推理延迟：
- **首次推理**: ~2-3 秒（模型加载 + 初始化）
- **后续推理**: ~800-1300ms（取决于 GPU）
- **吞吐量**: 每秒 1 次推理，每次返回 16 步动作

由于使用了 action chunking，实际执行很流畅：
- Server 每秒推理 1 次
- Client 每秒执行 16 步动作

---

## 🚀 真机部署

当硬件就绪时，只需替换 Client：

```bash
# Terminal 1: Server 保持不变
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Terminal 2: 使用真机 Client
uv run python grasp_cube/real/run_env_client.py \
    --env.robot.hostname 192.168.1.100 \
    --task lift
```

一切其他都相同！

---

## 🐳 Docker 部署

当需要打包分发时：

```bash
# 构建镜像
docker build -f Dockerfile.diffusion -t so101-diffusion:v1 .

# 运行（Lift 任务）
docker run -it --gpus all -p 8000:8000 \
    -e TASK=lift so101-diffusion:v1

# 从主机连接
# 修改 Client 的 --host 为 docker 主机 IP
```

详见 `REAL_ROBOT_DEPLOYMENT.md` 的 Docker 部分。

---

## 📚 详细文档

- 🌟 **START_SERVER_CLIENT.md** - 本文，快速启动指南
- 📋 **INTEGRATION_COMPLETE.md** - 集成完成总结
- 📖 **REAL_ROBOT_DEPLOYMENT.md** - 完整部署和故障排查
- ✅ **REAL_ROBOT_CHECKLIST.md** - 部署检查清单
- 📊 **FINAL_REPORT.md** - 项目完成报告

---

## 🎉 准备好了吗？

现在就运行这两个命令开始测试：

```bash
# Terminal 1
cd /home/gotham/shared/so101-grasp-cube && \
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best --policy.task lift

# Terminal 2 (新终端)
cd /home/gotham/shared/so101-grasp-cube && \
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift --num-episodes 2

# Browser
http://localhost:9000
```

**祝你成功！** 🚀
