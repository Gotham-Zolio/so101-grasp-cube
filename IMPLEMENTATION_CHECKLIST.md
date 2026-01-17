# 真机部署实现检查清单

## 📋 总体计划

这份文档列出了从当前状态（离线推理验证完成）到真机完整任务执行的所有必要步骤。

---

## 第1阶段：真机推理能力验证（不执行动作）

### ✅ 已完成

- [x] 离线推理测试：`scripts/test_offline_inference.py` (6/6 passing)
- [x] DiffusionPolicyInferenceEngine：`scripts/inference_engine.py` (完全实现)
- [x] 推理包装器：`grasp_cube/real/diffusion_inference_wrapper.py` (完全实现)
- [x] 传感器验证测试：`scripts/test_real_sensor_input.py` (完全实现)
- [x] 快速参考指南：`QUICK_START.md` (完全实现)
- [x] 部署路线图：`DEPLOYMENT_ROADMAP.md` (完全实现)

### ⏳ 待实现（优先级：高）

- [ ] 1.1 在真机上运行 `test_real_sensor_input.py`
  - **文件**: `scripts/test_real_sensor_input.py`
  - **步骤**: 
    ```bash
    uv run python scripts/test_real_sensor_input.py \
      --robot-type so101 \
      --task lift \
      --device cuda
    ```
  - **验收标准**: 
    - ✓ Test 1 PASSED (单次推理 <1s)
    - ✓ Test 2 PASSED (连续推理 10步)
    - ✓ Test 3 PASSED (多任务切换)
    - ✓ Test 4 PASSED (错误处理)
  - **预计时间**: 1-2小时
  - **关键代码**:
    ```python
    # 将模拟数据替换为真机数据
    obs = get_real_robot_observation()  # 替换 get_mock_observation()
    image = obs["rgb"]  # 从真机摄像头
    state = obs["joint_positions"]  # 从真机关节编码器
    ```

- [ ] 1.2 集成真机摄像头读取
  - **文件**: `scripts/test_real_sensor_input.py` (修改)
  - **需要实现的**:
    - 连接真机摄像头（OpenCV或RealSense）
    - 读取RGB图像数据
    - 验证图像质量和帧率
  - **参考代码**:
    ```python
    import cv2
    cap = cv2.VideoCapture(camera_id)  # 打开摄像头
    ret, frame = cap.read()  # 读取一帧
    # frame 是 (480, 640, 3) uint8
    ```
  - **预计时间**: 1-2小时

- [ ] 1.3 集成真机关节状态读取
  - **文件**: `scripts/test_real_sensor_input.py` (修改)
  - **需要实现的**:
    - 连接真机机械臂控制器
    - 读取关节位置
    - 将位置缩放到 [-π, π] 范围
  - **预计时间**: 1-2小时

**第1阶段总耗时**: 3-6小时（取决于真机驱动和网络连接复杂度）

---

## 第2阶段：真机动作执行能力验证（小幅度动作）

### 待实现（优先级：高）

- [ ] 2.1 创建动作执行器
  - **新文件**: `grasp_cube/real/action_executor.py`
  - **核心类**: `RealRobotActionExecutor`
  - **需要实现的方法**:
    ```python
    class RealRobotActionExecutor:
        def __init__(self, robot_config):
            # 初始化机械臂控制
            # 获取当前关节位置
            pass
        
        def execute_action(self, action: np.ndarray) -> bool:
            # 输入：action (action_dim,) 单个动作 [-1, 1]
            # 逻辑：
            #   1. 转换为实际关节角度增量
            #   2. 计算目标位置
            #   3. 执行到目标位置（使用PID或轨迹插值）
            #   4. 等待完成或超时
            pass
        
        def get_safety_limits(self) -> dict:
            # 返回安全限制参数
            pass
        
        def emergency_stop(self):
            # 紧急停止，回到home位置
            pass
    ```
  - **关键参数需要定义**:
    - SO101单臂的最大关节角速度
    - 最大关节加速度
    - 工作空间限制
    - 碰撞检测阈值
  - **预计时间**: 4-6小时

- [ ] 2.2 创建动作chunk执行器
  - **修改文件**: `grasp_cube/real/diffusion_inference_wrapper.py`
  - **需要添加的方法**:
    ```python
    class ActionChunkExecutor:
        def execute_action_chunk(self, action_chunk: np.ndarray) -> bool:
            # 循环执行chunk中的所有动作
            # 支持中断和错误恢复
            pass
    ```
  - **预计时间**: 2-3小时

- [ ] 2.3 创建安全测试
  - **新文件**: `scripts/test_real_safe_execution.py`
  - **需要实现的测试**:
    - Test 1: 回到home位置安全性
    - Test 2: 小范围动作执行
    - Test 3: Chunk执行（16个动作）
    - Test 4: 中断响应时间
    - Test 5: 边界条件安全检查
  - **参考**: `scripts/test_offline_inference.py` 的测试结构
  - **预计时间**: 2-4小时

- [ ] 2.4 调试和优化
  - **关键调试点**:
    - 动作映射是否正确（[-1,1] → 关节角度）
    - 执行延迟是否满足 <100ms/动作
    - 是否能正确处理异常动作
  - **预计时间**: 2-4小时

**第2阶段总耗时**: 10-17小时

---

## 第3阶段：完整任务执行能力验证

### 待实现（优先级：高）

- [ ] 3.1 创建感知反馈检查器
  - **新文件**: `grasp_cube/real/perception_checker.py`
  - **核心类**: `TaskPerceptionChecker`
  - **需要实现的方法**:
    ```python
    class TaskPerceptionChecker:
        def __init__(self, task_name: str):
            # 不同任务的不同检查逻辑
            pass
        
        def check_grasp_success(self, gripper_state, force_data) -> bool:
            # 验证是否成功抓取
            pass
        
        def check_placement_success(self, object_pose, target_pose) -> bool:
            # 验证是否正确放置
            pass
        
        def check_task_completion(self, observation) -> bool:
            # 检查任务是否完成
            pass
        
        def check_failure_condition(self, observation) -> bool:
            # 检查是否失败（碰撞、超时等）
            pass
    ```
  - **需要集成的传感器**:
    - 视觉识别（前视图像）
    - 力/扭矩传感器（腕部）
    - 夹爪位置反馈
  - **预计时间**: 6-8小时

- [ ] 3.2 创建任务执行管理器
  - **新文件**: `grasp_cube/real/task_executor.py`
  - **核心类**: `RealRobotTaskExecutor`
  - **主循环**:
    ```python
    def execute_task(self, observation, max_steps=100):
        for step in range(max_steps):
            # 1. 推理
            action_chunk = self.inference.predict_from_obs(observation)
            
            # 2. 执行
            result = self.action_executor.execute_action_chunk(action_chunk)
            
            # 3. 观测
            observation = self.robot.get_observation()
            
            # 4. 检查完成
            if self.perception.check_task_completion(observation):
                return TaskResult(success=True)
            
            # 5. 检查失败
            if self.perception.check_failure_condition(observation):
                return TaskResult(success=False, reason="collision")
        
        return TaskResult(success=False, reason="timeout")
    ```
  - **预计时间**: 4-6小时

- [ ] 3.3 创建任务执行测试
  - **新文件**: `scripts/test_real_task_execution.py`
  - **测试用例**:
    - Test 1: Lift cube (单臂，简单任务)
    - Test 2: Sort cube (双臂，中等复杂)
    - Test 3: 重复执行（鲁棒性）
    - Test 4: 任务切换
  - **验收标准**:
    - 成功率 ≥80%
    - 执行时间 <10分钟/任务
    - 没有碰撞或损伤
  - **预计时间**: 4-8小时

**第3阶段总耗时**: 14-22小时

---

## 第4阶段：完整Pick-Place任务

### 待实现（优先级：中）

- [ ] 4.1 扩展任务定义
  - **新文件**: `grasp_cube/real/task_definitions.py`
  - **需要定义**:
    - 任务执行步骤序列
    - 成功/失败条件
    - 安全限制
  - **预计时间**: 2-3小时

- [ ] 4.2 修改评估脚本
  - **修改文件**: `scripts/eval_real_policy.py`
  - **需要修改的**:
    - 集成DiffusionPolicyInferenceEngine替代LeRobotDiffusionPolicy
    - 添加视频记录
    - 添加性能统计
  - **预计时间**: 2-4小时

- [ ] 4.3 性能基准测试
  - **新文件**: `scripts/benchmark_real_policy.py`
  - **需要测试的**:
    - 推理延迟分布
    - 端到端延迟
    - 成功率
    - 鲁棒性（不同光线、位置等）
  - **预计时间**: 3-4小时

**第4阶段总耗时**: 7-11小时

---

## 第5阶段：系统集成

### 待实现（优先级：中）

- [ ] 5.1 修改run_env_client.py
  - **修改文件**: `grasp_cube/real/run_env_client.py`
  - **需要添加**:
    - `--policy-type` 参数（act vs diffusion）
    - DiffusionPolicy推理引擎的加载和使用
    - Action chunking的支持
  - **预计时间**: 2-3小时

- [ ] 5.2 Docker打包
  - **新文件**: `Dockerfile.diffusion`
  - **参考**: `docker_tutorial.md`
  - **包含的内容**:
    - PyTorch基础镜像
    - LeRobot及依赖
    - grasp_cube包
    - 启动脚本
  - **预计时间**: 1-2小时

- [ ] 5.3 模型优化（可选）
  - **选项**:
    - Float16量化
    - ONNX导出
    - TensorRT优化
  - **只在推理延迟>1s时考虑**
  - **预计时间**: 4-8小时（如果必要）

**第5阶段总耗时**: 5-8小时（7-13小时如果包含优化）

---

## 📊 总体时间线

| 阶段 | 任务数 | 预计时间 | 累计时间 |
|------|--------|---------|----------|
| 1 | 4 | 3-6h | 3-6h |
| 2 | 4 | 10-17h | 13-23h |
| 3 | 3 | 14-22h | 27-45h |
| 4 | 3 | 7-11h | 34-56h |
| 5 | 3 | 5-8h | 39-64h |
| **合计** | **17** | **39-64h** | **39-64h** |

**时间估算说明**:
- 上限（64小时）：包含大量调试和优化
- 下限（39小时）：假设一切顺利，不需要额外优化
- 实际时间：根据现有代码质量、真机复杂度等因素

---

## 🎯 关键里程碑和验收标准

### Milestone 1: 推理验证完成
- ✓ `test_real_sensor_input.py` 全部通过
- ✓ 推理延迟在可接受范围内 (<1s)
- 预计：第1周

### Milestone 2: 动作执行验证完成
- ✓ `test_real_safe_execution.py` 全部通过
- ✓ 能安全执行小幅度动作
- ✓ Chunk执行成功率 >95%
- 预计：第2-3周

### Milestone 3: 完整任务执行验证完成
- ✓ `test_real_task_execution.py` 通过
- ✓ Lift/Sort/Stack任务成功率 ≥80%
- 预计：第4-5周

### Milestone 4: 系统集成完成
- ✓ 集成到run_env_client.py
- ✓ Docker镜像可用
- ✓ 文档完整
- 预计：第6周

---

## 🔧 关键代码修改点汇总

### action_executor.py（新建）- 关键映射

```python
# 动作值映射：[-1, 1] → 关节角度增量
def action_to_delta_angle(self, action: np.ndarray) -> np.ndarray:
    # action in [-1, 1]
    # delta_angle in [-MAX_DELTA, MAX_DELTA]
    MAX_DELTA = 0.1  # 弧度，根据SO101规格调整
    return action * MAX_DELTA

# 目标位置计算
def compute_target_position(self, current_pos, action):
    delta = self.action_to_delta_angle(action)
    target = current_pos + delta
    return np.clip(target, self.min_joint_limits, self.max_joint_limits)
```

### perception_checker.py（新建）- 抓取检测

```python
def check_grasp_success(self, gripper_state, force_data):
    # 条件1：夹爪闭合
    is_closed = gripper_state["position"] < 0.1  # 接近完全闭合
    
    # 条件2：夹爪受力（有东西被夹住）
    gripper_force = force_data["gripper_force"]
    has_force = gripper_force > FORCE_THRESHOLD  # 例如 >5N
    
    return is_closed and has_force
```

### run_env_client.py（修改）- 策略集成

```python
# 添加这段到main函数中
if args.policy_type == "diffusion":
    from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper
    policy = RealRobotDiffusionInferenceWrapper(
        task_name=args.env.task,
        device=args.device
    )
    # 使用 policy.predict_from_obs() 替代原有的 client.infer()
else:
    # 原有的ACT策略
    client = _websocket_client_policy.WebsocketClientPolicy(...)
```

---

## ⚠️ 关键风险和缓解方案

### 风险1: 推理延迟过长
**缓解**: GPU预热、模型量化、优化image处理流程

### 风险2: 动作映射错误
**缓解**: 充分的单元测试、可视化验证、保守的安全限制

### 风险3: 感知反馈不稳定
**缓解**: 多模态融合（视觉+力反馈）、滤波、多帧平均

### 风险4: 闭环控制不稳定
**缓解**: PID调参、轨迹平滑、反馈控制

---

## 📝 编码建议

### 1. 遵循现有代码风格
- 参考 `scripts/inference_engine.py` 和 `scripts/test_offline_inference.py`
- 使用类似的日志和错误处理方式
- 添加详细的docstring

### 2. 充分的日志记录

```python
import logging

logger = logging.getLogger(__name__)

# 在关键点添加日志
logger.info(f"Executing action: {action}")
logger.warning(f"Action close to limit: {action_magnitude}")
logger.error(f"Execution failed: {e}")
```

### 3. 单元测试
每个新的module都应该有对应的测试文件
```
新文件: grasp_cube/real/foo.py
测试: scripts/test_real_foo.py
```

### 4. 类型注解
使用完整的类型注解便于IDE支持和代码阅读
```python
def execute_action(
    self,
    action: np.ndarray,
    timeout_s: float = 1.0
) -> Tuple[bool, str]:  # (success, error_msg)
```

---

## 📚 相关文档

- `DEPLOYMENT_ROADMAP.md` - 详细的部署步骤说明
- `QUICK_START.md` - 快速参考和代码示例
- `scripts/test_offline_inference.py` - 测试框架参考
- `grasp_cube/real/lerobot_env.py` - 观测格式定义
- `README.md` - 项目整体说明

---

## 🚀 立即行动项

**现在可以立即开始的**:

1. ✅ 运行离线推理测试验证环境
   ```bash
   uv run python scripts/test_offline_inference.py
   ```

2. ✅ 查看推理包装器文档和示例
   ```bash
   python -c "from grasp_cube.real.diffusion_inference_wrapper import *; help(RealRobotDiffusionInferenceWrapper)"
   ```

3. ✅ 准备真机测试环境
   - 检查摄像头驱动
   - 确保机械臂通信正常
   - 建立安全工作空间

4. ⏳ 开始第1阶段
   ```bash
   uv run python scripts/test_real_sensor_input.py --robot-type so101 --task lift
   ```

---

**最后更新**: 2026-01-17
**状态**: 第1阶段准备就绪，第2-5阶段任务定义完成
