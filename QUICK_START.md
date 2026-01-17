# 从离线推理到真机部署 - 快速参考

## 📋 当前状态检查清单

- [x] **离线推理验证** (✅ 6/6 tests passing)
  - DiffusionPolicyInferenceEngine 实现完成
  - 三任务模型支持（lift/sort/stack）
  - 手动归一化和维度适配
  
- [ ] **真机集成阶段1-5** (开始中)

---

## 🚀 快速开始（第1阶段：传感器验证）

### 1. 运行真机传感器推理测试

```bash
cd /path/to/so101-grasp-cube

# 基本测试（使用模拟传感器数据）
uv run python scripts/test_real_sensor_input.py \
  --robot-type so101 \
  --task lift \
  --duration 10 \
  --device cuda

# 输出应该显示：
# ✓ Test 1 PASSED: Single Real Sensor Inference
# ✓ Test 2 PASSED: Continuous Real Sensor Inference (10s)
# ✓ Test 3 PASSED: Multi-Task Model Switching
# ✓ Test 4 PASSED: Inference Error Handling
# Total: 4/4 tests passed
```

### 2. 在代码中使用推理包装器

```python
from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper

# 初始化
wrapper = RealRobotDiffusionInferenceWrapper(
    task_name="lift",
    device="cuda",
    verbose=True
)

# 从观测预测动作序列
observation = {
    "images": {"front": image_480x640},  # uint8 RGB image
    "states": {"arm": joint_state}       # 6-dim float32
}

# 方法1: 获取完整的动作序列
action_chunk = wrapper.predict_from_obs(observation)  # (16, 6)

# 方法2: 逐步获取动作（用于行为执行）
for step in range(100):
    action, remaining = wrapper.get_next_action(observation)
    robot.execute(action)
    if not wrapper.has_pending_actions():
        break
    observation = robot.get_observation()

# 切换任务
wrapper.switch_task("sort")  # 切换到双臂任务

# 获取调试信息
debug_info = wrapper.get_debug_info()
print(f"Task: {debug_info['task_name']}")
print(f"Remaining actions: {debug_info['remaining_actions']}")
```

### 3. 集成到真机环境（参考）

```python
# 在 run_env_client.py 中的使用方式
from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper

# 初始化环境和包装器
env = LeRobotEnv(config)
wrapper = RealRobotDiffusionInferenceWrapper(
    task_name=config.task,
    device=config.device
)

# 主循环
for episode in range(num_episodes):
    obs, info = env.reset()
    wrapper.reset_chunk()
    
    done = False
    action_queue = deque()
    
    while not done:
        # 获取动作块（如果当前块用完了）
        if not action_queue:
            try:
                action_chunk = wrapper.predict_from_obs(obs)
                action_queue.extend(action_chunk)
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # 执行单个动作
        action = action_queue.popleft()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

---

## 📁 关键文件位置

### 推理引擎
- **`scripts/inference_engine.py`** - 核心推理引擎
  - `DiffusionPolicyInferenceEngine` 类
  - 支持多任务、自动图像缩放、手动归一化

### 真机集成（新建）
- **`scripts/test_real_sensor_input.py`** - 第1阶段测试
  - 传感器数据读取验证
  - 推理延迟测试
  - 错误处理验证

- **`grasp_cube/real/diffusion_inference_wrapper.py`** - 推理包装器
  - `RealRobotDiffusionInferenceWrapper` 类
  - 观测数据预处理
  - Action chunking 管理
  - 任务切换

### 真机环境（已有）
- **`grasp_cube/real/lerobot_env.py`** - 真机环境
  - 观测格式定义
  - 图像和状态处理

- **`grasp_cube/real/run_env_client.py`** - 环境客户端
  - WebSocket 连接
  - Action chunking 执行

---

## 🔧 常见任务和代码片段

### 任务1: 验证推理引擎能否处理真机数据

```python
import numpy as np
from scripts.inference_engine import DiffusionPolicyInferenceEngine

engine = DiffusionPolicyInferenceEngine(
    "checkpoints/lift_real/checkpoint-best"
)

# 真机数据格式
image = robot.get_rgb_image()  # (480, 640, 3) uint8
state = robot.get_joint_state()  # (6,) float32 [-π, π]

# 转换为推理格式
image_f32 = image.astype(np.float32) / 255.0  # [0, 1]
image_chw = np.transpose(image_f32, (2, 0, 1))  # (3, 480, 640)

# 推理
actions = engine.predict(image_chw, state)  # (16, 6)
```

### 任务2: 处理多任务模型

```python
from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper

# 初始化为lift任务
wrapper = RealRobotDiffusionInferenceWrapper("lift")
print(f"State dim: {wrapper.engine.state_dim}")  # 6

# 切换到sort任务（双臂）
wrapper.switch_task("sort")
print(f"State dim: {wrapper.engine.state_dim}")  # 12

# 现在可以处理12维的状态向量
obs_12dim = {
    "images": {...},
    "states": {
        "left_arm": np.zeros(6),
        "right_arm": np.zeros(6)
    }
}
actions = wrapper.predict_from_obs(obs_12dim)
```

### 任务3: Action Chunking 执行

```python
# 获取动作块（16个动作）
action_chunk = wrapper.predict_from_obs(observation)

# 逐个执行
for i, action in enumerate(action_chunk):
    robot.execute_action(action)
    time.sleep(1/30)  # 30 Hz control loop
    
    # 如果需要中断（例如检测到完成）
    if task_completed():
        break
    
    # 更新观测用于下一次推理
    if i % 5 == 0:  # 每5步重新推理一次（可选）
        observation = robot.get_observation()
```

### 任务4: 错误处理

```python
from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper

wrapper = RealRobotDiffusionInferenceWrapper("lift")

try:
    actions = wrapper.predict_from_obs(observation)
except ValueError as e:
    if "observation missing" in str(e):
        print("观测数据格式错误")
    elif "cannot extract state" in str(e):
        print("状态向量维度不匹配")
except Exception as e:
    print(f"推理失败: {e}")
    # 回到home位置
    robot.go_home()
```

### 任务5: 性能监控

```python
import time

wrapper = RealRobotDiffusionInferenceWrapper("lift")
debug_info = wrapper.get_debug_info()

print(f"Model: {debug_info['task_name']}")
print(f"State dim: {debug_info['model_state_dim']}")
print(f"Action dim: {debug_info['model_action_dim']}")
print(f"Horizon: {debug_info['horizon']}")

# 推理延迟测试
start = time.time()
actions = wrapper.predict_from_obs(observation)
elapsed = time.time() - start

print(f"Inference time: {elapsed*1000:.2f} ms")
print(f"Actions pending: {debug_info['remaining_actions']}")
```

---

## ⚠️ 常见问题

### Q1: 推理时间太长（>1秒）

**症状**: 每次推理需要1-3秒

**解决方案**:
1. 检查GPU是否被占用：`nvidia-smi`
2. 切换到CPU试试：`wrapper = RealRobotDiffusionInferenceWrapper(..., device="cpu")`
3. 确保模型完全加载到GPU显存
4. 考虑使用模型蒸馏或量化

### Q2: 状态维度不匹配

**症状**: `ValueError: State dim mismatch`

**原因**:
- Sort任务需要12维状态（双臂），但传入了6维
- Lift/Stack任务需要6维状态

**解决方案**:
```python
# 检查任务需要的维度
print(f"Required state dim: {wrapper.engine.state_dim}")

# 确保提供正确维度的状态
if wrapper.task_name == "sort":
    state = np.concatenate([left_arm_state, right_arm_state])  # 12维
else:
    state = arm_state  # 6维
```

### Q3: 推理输出包含NaN或Inf

**症状**: `actions` 数组中有NaN或Inf值

**原因**:
- 输入的标准化失败（归一化时除以0）
- 模型权重有问题

**解决方案**:
```python
# 检查输出
if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
    print("Invalid output! Check:")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
    print(f"  Stats file exists: {Path('checkpoints/.../stats.json').exists()}")
```

### Q4: 图像大小不对

**症状**: `ValueError: Expected image shape (3, 480, 640) or (3, 84, 84)`

**解决方案**:
```python
# 推理引擎自动处理 480x640 到 84x84 的缩放
# 只需确保输入是正确的格式

# 如果你的摄像头分辨率不同，手动缩放
if image.shape != (480, 640, 3):
    image = cv2.resize(image, (640, 480))

# 然后按照格式转换
image_f32 = image.astype(np.float32) / 255.0
image_chw = np.transpose(image_f32, (2, 0, 1))
```

---

## 📊 性能基准

基于当前实现的预期性能：

| 指标 | 目标 | 当前 | 备注 |
|------|------|------|------|
| 单次推理延迟 | <500ms | 800-1300ms | GPU优化空间 |
| 批处理延迟 | <100ms/sample | 100ms | 达到目标 |
| 内存占用 | <2GB | ~1.2GB | 可接受 |
| 推理准确度 | - | 100% | 形状/维度 |
| 多任务切换 | <100ms | ~50ms | 快速 |

---

## 🔄 下一步（推荐顺序）

1. **现在**: 运行 `test_real_sensor_input.py` 验证推理能力
2. **第2阶段**: 实现动作执行器（`grasp_cube/real/action_executor.py`）
3. **第3阶段**: 实现感知反馈检查（`grasp_cube/real/perception_checker.py`）
4. **第4阶段**: 完整任务执行（`scripts/test_real_task_execution.py`）
5. **第5阶段**: 系统集成和Docker打包

每一步都有详细的实现指南在 `DEPLOYMENT_ROADMAP.md` 中。

---

## 📚 参考文档

- `DEPLOYMENT_ROADMAP.md` - 完整的部署路线图
- `scripts/test_offline_inference.py` - 离线推理测试（已验证）
- `scripts/inference_engine.py` - 核心推理引擎
- `grasp_cube/real/lerobot_env.py` - 真机环境定义
- `README.md` - 项目总体说明

---

## 💡 最佳实践

1. **始终先验证数据格式**
   ```python
   wrapper = RealRobotDiffusionInferenceWrapper("lift")
   print(wrapper.get_debug_info())
   ```

2. **使用verbose模式调试**
   ```python
   wrapper = RealRobotDiffusionInferenceWrapper(
       "lift",
       verbose=True  # 打印详细信息
   )
   ```

3. **处理所有异常**
   ```python
   try:
       actions = wrapper.predict_from_obs(obs)
   except Exception as e:
       logger.error(f"Inference failed: {e}")
       robot.emergency_stop()
   ```

4. **定期检查系统状态**
   ```python
   debug = wrapper.get_debug_info()
   if debug['remaining_actions'] == 0:
       # 需要生成新的动作chunk
       pass
   ```

5. **记录失败case用于调试**
   ```python
   failed_obs = observation  # 保存失败时的观测
   failed_actions = actions  # 保存推理输出
   # 后续分析
   ```

