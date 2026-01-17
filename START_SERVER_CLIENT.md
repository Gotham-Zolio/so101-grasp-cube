# 启动 Server 和 Client 的完整指南

## 步骤 1: 启动 Server（推理服务）

在 **Terminal 1** 运行：

```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift
```

**预期输出**：
```
======================================================================
Creating DiffusionPolicy WebSocket Server
======================================================================
✓ Initializing LeRobotDiffusionPolicy
  ...
✓ Inference wrapper initialized

✓ Server created successfully
  Host: 0.0.0.0
  Port: 8000
  
Waiting for client connections at ws://0.0.0.0:8000
INFO:websockets.server:server listening on 0.0.0.0:8000
```

**Server 现在在等待 Client 连接。请不要关闭此终端。**

---

## 步骤 2: 启动 Client（模拟环境）

在 **Terminal 2** 运行：

```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --num-episodes 2
```

**参数说明**：
- `--env.task lift`: 指定任务（lift/sort/stack）
- `--num-episodes 2`: 运行 2 个 episode
- `--host 0.0.0.0`: Server 地址（默认）
- `--port 8000`: Server 端口（默认）
- `--monitor-port 9000`: 监控界面端口（默认）

**预期输出**：
```
Connecting to policy server at ws://0.0.0.0:8000
Connected to policy server!
Starting environment loop...
Episode 1/2...
...
```

---

## 步骤 3: 查看监控界面

在浏览器打开：
```
http://localhost:9000
```

你会看到：
- 🎬 机械臂执行的实时视频
- 📊 推理的实时信息
- 📈 性能指标

---

## 完整工作流总结

| 步骤 | 位置 | 命令 |
|------|------|------|
| 1 | Terminal 1 | `serve_diffusion_policy.py --policy.path checkpoints/lift_real/checkpoint-best --policy.task lift` |
| 2 | Terminal 2 | `run_fake_env_client.py --env.task lift --num-episodes 2` |
| 3 | 浏览器 | `http://localhost:9000` |

---

## 多任务测试

### Lift Task
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --num-episodes 2
```

### Sort Task
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/sort_real/checkpoint-best \
    --policy.task sort

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task sort \
    --num-episodes 2
```

### Stack Task
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/stack_real/checkpoint-best \
    --policy.task stack

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task stack \
    --num-episodes 2
```

---

## 常见问题

### 问题：Connection refused
**原因**：Server 未运行  
**解决**：确认 Terminal 1 的 Server 已成功启动

### 问题：WebSocket connection failed
**原因**：Client 连接的 Server 地址错误  
**解决**：检查 `--host` 和 `--port` 是否与 Server 一致

### 问题：推理输出维度错误
**原因**：任务选择错误  
**解决**：确保两边的 `--task` 参数一致

### 问题：监控界面无法访问
**原因**：Client 未运行，监控服务未启动  
**解决**：确保 Terminal 2 的 Client 正在运行

---

## 环境变量配置

### 使用远程 Server
```bash
# Client 连接远程 Server
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --host 192.168.1.100 \
    --port 8000
```

### 自定义监控界面端口
```bash
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --monitor-port 9001
```

### 修改 Server 绑定地址
```bash
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --host 192.168.1.50 \
    --port 8000
```

---

## 真机部署

当使用真机时，只需替换 Client：

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

---

## 系统架构

```
┌─────────────────────────────────────┐
│     Terminal 1: Server              │
│  serve_diffusion_policy.py          │
│  WebSocket: ws://0.0.0.0:8000       │
└──────────────┬──────────────────────┘
               │
               │ WebSocket connection
               │
┌──────────────▼──────────────────────┐
│     Terminal 2: Client              │
│  run_fake_env_client.py             │
│  Monitoring: http://localhost:9000  │
└──────────────┬──────────────────────┘
               │
               │ 推理请求 + 推理结果
               │
┌──────────────▼──────────────────────┐
│     Browser: Monitoring UI          │
│  http://localhost:9000              │
│  实时可视化和性能监控                 │
└─────────────────────────────────────┘
```

---

## 下一步

✅ 按照上述步骤运行 Server 和 Client  
✅ 在浏览器中查看监控界面  
✅ 验证推理和执行正常  
✅ 如果都正常，准备进行真机部署

---

## 获取帮助

- 查看 `REAL_ROBOT_DEPLOYMENT.md` 获取详细部署指南
- 查看 `REAL_ROBOT_CHECKLIST.md` 获取快速参考
- 查看 `serve_diffusion_policy.py` 的代码注释了解实现细节
