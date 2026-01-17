# 真机集成和测试检查清单

## 📋 **A. 代码集成检查**

### A1. 推理引擎集成到你的真机控制代码
- [ ] 在 `hello_real_robot.py` 或类似文件中导入 `RobotPolicyController`
  ```python
  from scripts.robot_policy_controller import SO101RobotPolicyController
  ```

- [ ] 在机器人初始化时创建控制器
  ```python
  controller = SO101RobotPolicyController(
      robot_interface=your_robot,
      camera_interface=your_camera,
      action_scale=0.1  # 从 10% 速度开始
  )
  ```

- [ ] 在主控制循环中调用推理
  ```python
  while True:
      success = controller.step("lift")
      if not success:
          print("Step failed, breaking...")
          break
  ```

### A2. 集成点清单
- [ ] 相机驱动返回格式正确：(3, 480, 640) float32 [0, 1]
  - 检查方法：
    ```python
    image = camera.get_frame()
    print(f"Shape: {image.shape}, dtype: {image.dtype}")
    print(f"Range: [{image.min():.3f}, {image.max():.3f}]")
    assert image.shape == (3, 480, 640)
    assert image.dtype == np.float32
    assert 0 <= image.min() and image.max() <= 1.0
    ```

- [ ] 关节状态返回格式正确：(state_dim,) float32
  - 对于 lift/stack：应该是 (6,) 双臂 6 个关节
  - 对于 sort：可能是 (12,) 或其他维度
  - 检查方法：
    ```python
    state = robot.get_state()
    print(f"Shape: {state.shape}, dtype: {state.dtype}")
    assert state.dtype == np.float32
    ```

- [ ] 动作执行接受 (action_dim,) float32
  - 检查方法：
    ```python
    action = np.random.randn(6).astype(np.float32)
    robot.execute_action(action)  # 应该没有错误
    ```

---

## 🔧 **B. 真机硬件检查**

### B1. 相机系统
- [ ] 相机能正常启动
- [ ] 分辨率是 640×480（或确认实际分辨率，修改推理代码）
- [ ] 帧率 ≥ 30 FPS
- [ ] 图像格式是 RGB（不是 BGR）
- [ ] 内参标定完成（如果需要裁剪或缩放）

测试代码：
```python
import cv2
cap = cv2.VideoCapture(0)  # 或你的相机索引
print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
```

### B2. 关节传感器
- [ ] 6 个关节的角度传感器都能读取
- [ ] 单位：弧度（rad）
- [ ] 更新频率 ≥ 100 Hz
- [ ] 零位对齐（robot.get_state()[0] ≈ 0 当在初始位置）

### B3. 动作执行
- [ ] 动作命令能正常发送
- [ ] 执行延迟 < 50ms
- [ ] 速度限制已设置（防止机器人超速）
- [ ] 加速度限制已设置

### B4. 安全系统
- [ ] 急停按钮工作正常
- [ ] 碰撞检测已启用（如有）
- [ ] 关节限位已校准

---

## 🧪 **C. 离线测试（无需真机）**

按这个顺序执行：

### C1. 基础推理测试
```bash
# 验证模型可以加载和推理
uv run python scripts/test_offline_inference.py
```

**预期结果**：所有 6 个测试通过 ✓

### C2. 推理性能测试
```bash
uv run python scripts/inference_engine.py
```

**预期**：
- 推理延迟 < 100ms（理想 < 50ms）
- 如果 > 100ms，考虑：
  - 使用 GPU (CUDA)
  - 减少图像分辨率
  - 量化模型权重

### C3. 控制器集成测试
```bash
uv run python scripts/robot_policy_controller.py
```

**预期结果**：运行 10 个模拟步骤，无错误

---

## 🤖 **D. 增量式真机测试**

### D1. 阶段 1：机器人断电测试
**目的**：验证代码能通过整个循环而不会崩溃

```bash
# 机器人断电！
# 运行推理和"执行动作"（但机器人不动）
python hello_real_robot.py \
    --task lift \
    --mode simulation \  # 或 --dry-run
    --steps 100
```

**成功标准**：
- ✓ 程序运行 100 步无崩溃
- ✓ 推理时间稳定（< 100ms）
- ✓ 动作命令格式正确

---

### D2. 阶段 2：低速测试（10% 速度）
**目的**：验证机器人能接收和执行命令

```python
controller = SO101RobotPolicyController(
    action_scale=0.1  # 只有 10% 速度
)
```

**成功标准**：
- ✓ 机器人平缓移动（无急动作）
- ✓ 能完成基本动作（如举起物体）
- ✓ 无碰撞或异常噪音

---

### D3. 阶段 3：50% 速度测试
**目的**：验证中速操作

```python
action_scale=0.5
```

**成功标准**：
- ✓ 完成速度加快
- ✓ 仍无碰撞
- ✓ 动作精准

---

### D4. 阶段 4：100% 速度测试
**目的**：完整性能评估

```python
action_scale=1.0
```

**成功标准**：
- ✓ 任务成功完成
- ✓ 完成时间合理（< 30 秒/任务）
- ✓ 无碰撞或失败

---

### D5. 完整任务循环测试
**目的**：验证三个任务都能工作

```bash
for task in lift sort stack; do
    python hello_real_robot.py --task $task --num-trials 5
done
```

**预期成功率**：
- Lift: > 80%（相对简单）
- Sort: > 70%（更复杂）
- Stack: > 70%（同样复杂）

---

## 📊 **E. 性能监控**

在运行过程中监控这些指标：

### E1. 推理性能
```python
controller.print_stats()
# 输出：平均延迟、标准差、最小/最大延迟
```

目标：
- 平均延迟 < 50ms（30Hz 控制）
- 标准差 < 20ms（稳定）
- 最大延迟 < 100ms（无超时）

### E2. 任务成功率
```python
result = controller.run_task("lift", max_steps=500)
print(f"Success rate: {result['success_rate']:.1f}%")
```

目标：
- 无碰撞
- 无失败（或 < 5% 失败）

### E3. 运行时间
记录每个任务的完成时间：
```python
start = time.time()
controller.run_task("lift")
elapsed = time.time() - start
print(f"Completed in {elapsed:.1f}s")
```

---

## ⚠️ **F. 故障排查**

### 问题 1：推理很慢（> 100ms）
**症状**：动作很卡顿
**解决方案**：
- [ ] 确保使用 GPU：`engine.device == torch.device('cuda')`
- [ ] 检查 CUDA 可用性：`torch.cuda.is_available()`
- [ ] 如果没有 GPU，考虑量化或减少模型大小

### 问题 2：动作格式不匹配
**症状**：`robot.execute_action(action)` 报错
**解决方案**：
- [ ] 检查 action 维度：`action.shape == (state_dim,)`
- [ ] 检查数据类型：`action.dtype == np.float32`
- [ ] 检查值范围：通常应在 [-1, 1] 或 [-π, π]

### 问题 3：相机图像格式错误
**症状**：推理失败或输出全是 NaN
**解决方案**：
- [ ] 验证图像形状：`image.shape == (3, 480, 640)`
- [ ] 验证图像值范围：`0 <= image.min(), image.max() <= 1.0`
- [ ] 如果图像是 BGR，转换为 RGB：`image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
- [ ] 确保是 float32：`image = image.astype(np.float32)`

### 问题 4：碰撞或异常动作
**症状**：机器人移动异常或碰撞
**解决方案**：
- [ ] 降低 action_scale（例如 0.5）
- [ ] 添加动作平滑化：`action_smoothing=5`
- [ ] 检查关节限位设置是否正确
- [ ] 验证初始位置是否正确

---

## ✅ **G. 完整检查清单**

在进行真机测试前，确认以下所有项目：

```
代码集成：
  [ ] RobotPolicyController 导入成功
  [ ] 推理引擎可加载所有三个模型
  [ ] 相机接口返回正确格式
  [ ] 机器人状态接口返回正确格式
  [ ] 动作执行接口能接收 float32 向量

硬件检查：
  [ ] 相机 640×480, ≥30FPS, RGB 格式
  [ ] 关节传感器 6 个，单位弧度，≥100Hz
  [ ] 动作执行 < 50ms 延迟
  [ ] 安全限制已设置

离线测试：
  [ ] test_offline_inference.py 全部通过
  [ ] 推理延迟 < 100ms
  [ ] 控制器模拟运行正常

真机测试 - 阶段式：
  [ ] 阶段 1：机器人断电，代码运行 100 步无错误
  [ ] 阶段 2：10% 速度，平缓移动，无碰撞
  [ ] 阶段 3：50% 速度，动作精准，无碰撞
  [ ] 阶段 4：100% 速度，完整任务成功
  [ ] 阶段 5：三个任务循环，成功率 > 70%

监控：
  [ ] 推理延迟 < 50ms（目标）
  [ ] 任务成功率 > 70%
  [ ] 无安全事件或碰撞
  [ ] 运行时间合理

```

---

## 🚀 **下一步**

1. **立即执行**：
   ```bash
   bash scripts/run_offline_tests.sh
   ```

2. **准备真机集成**：
   - 检查你的 `hello_real_robot.py` 或等效控制脚本
   - 添加 `RobotPolicyController` 集成
   - 测试相机和传感器接口

3. **逐步进行真机测试**：
   - 从断电模拟开始
   - 逐步增加速度
   - 监控推理延迟和任务成功率

4. **性能优化**（如需要）：
   - 模型量化
   - 图像缩放
   - 动作平滑化调优

---

有任何问题，随时告诉我！
