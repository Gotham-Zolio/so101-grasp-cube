# 快速测试 DiffusionPolicy Server

## 架构说明

```
Terminal 1: Server (WebSocket)          Terminal 2: Client (HTTP + WebSocket)
                                        
serve_diffusion_policy.py          →    run_fake_env_client.py
   ws://0.0.0.0:8000                       (连接到 ws://0.0.0.0:8000)
   (纯 WebSocket，无 HTTP)                 (启动 http://localhost:9000)
```

**关键点**：
- ❌ **不能**直接访问 `http://0.0.0.0:8000`（Server 只支持 WebSocket）
- ✅ **必须**运行 Client 来启动监控界面

---

## 3 个终端操作步骤

### Terminal 1: 启动 Server（推理服务）
```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift
```

**预期输出**:
```
✓ Server created successfully
  Host: 0.0.0.0
  Port: 8000
  ...

Waiting for client connections at ws://0.0.0.0:8000
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### Terminal 2: 启动 Client（模拟环境 + 监控界面）
```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift
```

**预期输出**:
```
Connected to policy server at ws://0.0.0.0:8000
Starting MonitorWrapper...
Server started on port 9000
```

### Terminal 3 或浏览器: 查看监控界面
```
http://localhost:9000
```

---

## 完整工作流

| 步骤 | 位置 | 命令 | 作用 |
|------|------|------|------|
| 1 | Terminal 1 | `serve_diffusion_policy.py` | 启动 WebSocket 推理服务 |
| 2 | Terminal 2 | `run_fake_env_client.py` | 连接到 Server，启动模拟环境和监控界面 |
| 3 | 浏览器 | `http://localhost:9000` | 查看实时执行结果和可视化 |

---

## 监控界面会显示

- 🎬 **实时视频**：机械臂执行的画面
- 📊 **推理信息**：每一步的推理输出
- 📈 **性能指标**：推理延迟、吞吐量等
- 🎮 **控制面板**：任务进度、错误信息等

---

## 故障排查

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| "Connection refused" | Server 未运行 | 确认 Terminal 1 的 Server 已启动 |
| "no matching results" | 直接访问 HTTP Server | 运行 Client (Terminal 2)，它会启动 http://localhost:9000 |
| "WebSocket connection failed" | Client 地址错误 | 检查 Server 的 host/port，默认是 `ws://0.0.0.0:8000` |
| 监控界面不更新 | Client 未连接成功 | 检查 Terminal 2 的输出日志 |

---

## 额外选项

### 修改 Server 地址/端口
```bash
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift \
    --host 0.0.0.0 \
    --port 8000
```

### 修改 Client 连接的 Server
```bash
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift \
    --server-url ws://192.168.1.100:8000  # 远程 Server
```

### 修改监控界面端口
```bash
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.dataset-path datasets/lift \
    --monitor-port 9001  # 改为 http://localhost:9001
```

---

## 多任务测试

### Task: Lift
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best --policy.task lift

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py --env.dataset-path datasets/lift
```

### Task: Sort
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/sort_real/checkpoint-best --policy.task sort

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py --env.dataset-path datasets/sort
```

### Task: Stack
```bash
# Terminal 1
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/stack_real/checkpoint-best --policy.task stack

# Terminal 2
uv run python grasp_cube/real/run_fake_env_client.py --env.dataset-path datasets/stack
```

---

## 实际真机部署

当使用真机时，只需替换 Terminal 2 的命令：

```bash
# Terminal 1: 相同
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift

# Terminal 2: 使用真机 Client 代替
uv run python grasp_cube/real/run_env_client.py \
    --env.robot.hostname 192.168.1.100 \
    --task lift
```

---

## 总结

✅ **现在运行**:
1. Terminal 1: `serve_diffusion_policy.py` → 启动 Server
2. Terminal 2: `run_fake_env_client.py` → 启动 Client + 监控
3. 浏览器: `http://localhost:9000` → 查看结果

❌ **不要尝试**:
- 直接访问 `http://0.0.0.0:8000`（Server 不支持 HTTP）
- 只运行 Server 而不运行 Client（无法开始执行）
