# ✅ Server-Client 集成测试 - 完整指南

## 🎯 现在可以测试了

刚修复了两个问题：

### 问题 1: ✅ 已修复 - 浏览器访问地址
**错误**: `http://0.0.0.0:9000` 无法访问  
**原因**: `0.0.0.0` 只能用于服务器绑定，浏览器访问要用 `127.0.0.1`  
**解决**: 访问 `http://127.0.0.1:9000` ✅

### 问题 2: ✅ 已修复 - Client 推理循环
**症状**: Client 无限打印 `Episode: 0, Step: 0`，没有进行推理  
**原因**: SimpleFakeEnv 的 info dict 缺少 `action` 字段  
**修复**: 添加了 `"action": gt_action` 到 info dict ✅

---

## 🚀 现在开始测试

### Terminal 1: 启动 Server
```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path checkpoints/lift_real/checkpoint-best \
    --policy.task lift
```

### Terminal 2: 启动 Client
```bash
cd /home/gotham/shared/so101-grasp-cube
uv run python grasp_cube/real/run_fake_env_client.py \
    --env.task lift \
    --num-episodes 2
```

### Browser: 访问监控界面
```
http://127.0.0.1:9000
```

---

## 📊 应该看到的输出

### Server 输出:
```
✓ Server created successfully
Waiting for client connections at ws://0.0.0.0:8000
INFO:websockets.server:server listening on 0.0.0.0:8000
INFO:websockets.server:connection open
```

### Client 输出:
```
======================================================================
DiffusionPolicy Client Started
======================================================================
Connecting to Server at ws://0.0.0.0:8000
✓ Connected to Server!
✓ MonitorWrapper on http://127.0.0.1:9000
======================================================================

🎬 Episode 1/2
  ✓ Received 16 actions from Server
  Step 10: action shape=(6,), done=False
  Step 20: action shape=(6,), done=False
  ...
  ✓ Episode 1 completed (100 steps)

🎬 Episode 2/2
  ...
  ✓ Episode 2 completed (100 steps)

======================================================================
✅ All episodes completed!
Results saved to: outputs/eval_records/20260117_xxxxx
MonitorWrapper: http://127.0.0.1:9000
======================================================================
```

### Browser (http://127.0.0.1:9000):
- 🎬 实时视频展示
- 📊 推理信息
- 📈 性能指标

---

## 🔍 工作流说明

```
Server (Terminal 1)
  ↓
  WebSocket: ws://0.0.0.0:8000
  ↓
Client (Terminal 2)
  ├─ 推理请求: obs → Server
  ├─ 推理结果: actions ← Server
  ├─ 执行动作: actions → Environment
  └─ 启动 MonitorWrapper: http://127.0.0.1:9000
     ↓
  Browser
  └─ 实时监控推理过程
```

---

## ✨ 关键改动

### 1. SimpleFakeEnv 更新
```python
# 修复前: info 缺少 action 字段，导致 gt_action 失败
info = {"step": self.current_step, "episode": self.current_episode, "success": terminated}

# 修复后: 添加 action 字段
info = {
    "step": self.current_step,
    "episode": self.current_episode,
    "success": terminated,
    "action": gt_action,  # ✅ 添加此字段
}
```

### 2. Client 输出改进
- 添加启动日志
- 添加推理进度
- 添加完成总结
- 更清晰的调试信息

---

## 🎉 总结

✅ **现在一切就绪！**

只需：
1. Terminal 1: 启动 Server
2. Terminal 2: 启动 Client
3. Browser: 访问 `http://127.0.0.1:9000`

享受实时推理和可视化！ 🚀
