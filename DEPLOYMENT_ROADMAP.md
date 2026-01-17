# 从离线推理到真机完整任务执行的详细步骤

## 当前状态
- ✅ 离线推理测试：6/6 通过
- ✅ DiffusionPolicyInferenceEngine：已实现并验证
- ✅ 三任务模型支持：lift (6dim) / sort (12dim) / stack (6dim)
- ⚠ 真机集成：未完成

---

## 第1阶段：真机推理能力验证（不执行动作）

### 目标
验证推理引擎能否正确处理真机的实时传感器数据（RGB图像+关节状态）

### 1.1 创建真机传感器数据读取模块

**文件位置**：`scripts/test_real_sensor_input.py`

**需要实现的内容**：
```
- 连接真机的摄像头（前视+腕部）
- 读取真机的关节状态（单臂SO101为6维，双臂BI-SO101为12维）
- 数据预处理：
  * RGB图像 (480, 640) → float32 [0, 1] 范围
  * 关节状态 → 标准化到 [-1, 1] 范围
- 单次推理测试：输入真机数据 → 获得动作预测，但不执行
- 性能检测：记录推理延迟（目标 <500ms）
```

**关键文件修改点**：
- 参考 `grasp_cube/real/lerobot_env.py` 的图像读取和处理逻辑（第100-120行）
- 参考 `grasp_cube/real/lerobot_env.py` 的状态获取方式（prepare_observation方法）

**测试验收标准**：
- 能够连接摄像头和机械臂
- 单次推理延迟在200-800ms之间
- 推理输出形状正确：(horizon=16, action_dim=6或12)
- 推理输出值在合理范围内（-1到1之间）

### 1.2 集成推理引擎到真机环境

**文件位置**：`grasp_cube/real/diffusion_inference_wrapper.py`（新建）

**需要实现的内容**：
```
class RealRobotDiffusionInferenceWrapper:
    def __init__(self, task_name: str, model_checkpoint_dir: str):
        # 初始化DiffusionPolicyInferenceEngine
        # 从checkpoints/[task_name]_real/checkpoint-best加载
        # 支持: lift_real, sort_real, stack_real
        
    def predict_from_obs(self, observation: dict) -> np.ndarray:
        # 输入：observation 包含 images 和 states
        # 处理流程：
        #   1. 提取前视图像，转换为float32 [0, 1]
        #   2. 提取关节状态，形状要匹配模型的state_dim
        #   3. 调用self.engine.predict(image, state)
        #   4. 返回动作序列 (horizon, action_dim)
        
    def get_next_action(self, observation: dict) -> np.ndarray:
        # 返回预测序列的第一个动作
        # 用于行为策略部署
```

**关键修改点**：
- 在 `grasp_cube/real/lerobot_env.py` 的 `step()` 方法中集成
- 推理引擎需要处理任务切换（lift/sort/stack有不同的state_dim）
- 需要从环境的observation dict中正确提取state维度信息

### 1.3 创建真机推理测试脚本

**文件位置**：`scripts/test_real_inference.py`

**需要实现的内容**：
```
Test 1: Single Real Sensor Inference
- 连接真机一次
- 采集一帧RGB+一次状态读取
- 执行推理，验证输出形状和范围
- 不执行任何动作
- 打印：推理时间、输出范围、内存使用

Test 2: Continuous Real Inference (10 steps)
- 30Hz采集循环（与真机同步）
- 每一步：读取传感器 → 推理 → 记录延迟
- 统计：平均延迟、最大延迟、最小延迟
- 验证：没有内存泄漏

Test 3: Multi-Task Model Switching
- 依次加载lift/sort/stack模型
- 为每个模型准备对应维度的状态向量
- 验证推理成功和输出维度正确

Test 4: Inference Error Handling
- 输入异常数据：全黑图像、全白图像
- 传入错误维度的状态向量
- 验证错误处理的优雅性
```

**测试执行方式**：
```bash
cd /path/to/so101-grasp-cube
uv run python scripts/test_real_inference.py \
  --robot-type so101 \  # 或 bi_so101
  --task lift \         # 或 sort, stack
  --duration 10 \       # 秒数
  --device cuda
```

**测试验收标准**：
- ✅ Test 1: 单次推理成功，延迟<1s
- ✅ Test 2: 连续推理10步，平均延迟<500ms，无内存泄漏
- ✅ Test 3: 所有任务模型加载和推理正常
- ✅ Test 4: 异常输入处理优雅，不崩溃

---

## 第2阶段：真机动作执行能力验证（小范围动作）

### 目标
验证从推理输出到动作执行的完整管道（仅执行小幅度安全动作）

### 2.1 实现动作执行器

**文件位置**：`grasp_cube/real/action_executor.py`（新建）

**需要实现的内容**：
```
class RealRobotActionExecutor:
    def __init__(self, robot_config):
        # 初始化真机机械臂控制接口
        # 获取当前关节位置作为home_position
        
    def execute_action(self, action: np.ndarray, duration_ms: int = 100) -> bool:
        # 输入：action (action_dim,) 单个动作指令
        # 逻辑：
        #   1. 验证action范围是否在[-1, 1]
        #   2. 转换为实际关节角度增量
        #      delta_joint = action * MAX_JOINT_DELTA
        #   3. 计算目标位置：target = current_position + delta_joint
        #   4. 用PID/轨迹插值执行到目标位置
        #   5. 等待完成或超时
        #   6. 返回执行状态
        
    def get_safety_limits(self) -> dict:
        # 返回安全限制参数
        # MAX_JOINT_DELTA, MAX_VELOCITY, WORKSPACE_LIMITS等
        
    def emergency_stop(self):
        # 紧急停止，回到home位置
```

**关键修改点**：
- 动作映射：推理输出[-1,1] → 关节增量 → 实际关节角度
- 安全检查：目标位置是否在工作空间内
- 速度限制：关节速度不超过安全值（参考 SO101 规格说明书）
- 碰撞检测：通过力/扭矩传感器反馈检测碰撞

### 2.2 实现行为块执行（Action Chunking）

**文件位置**：在 `grasp_cube/real/diffusion_inference_wrapper.py` 中扩展

**需要实现的内容**：
```
class ActionChunkExecutor:
    def __init__(self, executor: RealRobotActionExecutor, chunk_size: int = 16):
        # chunk_size 就是推理的 horizon 参数
        
    def execute_action_chunk(self, action_chunk: np.ndarray, callback=None) -> ExecutionResult:
        # 输入：action_chunk (16, action_dim) 从推理引擎得到
        # 执行逻辑：
        #   for i, action in enumerate(action_chunk):
        #       if should_interrupt():  # 通过callback检查中断信号
        #           break
        #       executor.execute_action(action)
        #       callback(step=i, action=action)  # 报告进度
        
    def interrupt_execution(self):
        # 立即停止当前chunk的执行
```

**关键修改点**：
- 时间同步：每个动作执行周期 = 1/30秒 = 33.3ms
- 实时中断：支持在chunk执行中途中断并回到安全姿势

### 2.3 创建安全验证测试

**文件位置**：`scripts/test_real_safe_execution.py`

**需要实现的内容**：
```
Test 1: Safe Home Position
- 将机械臂移动到home位置
- 验证位置精度（<5°误差）
- 验证没有碰撞警告

Test 2: Small Range Motion
- 生成小范围动作序列（action magnitude < 0.2）
- 执行单个动作块
- 验证：
  * 执行延迟 < 100ms
  * 动作完成度 > 95%
  * 没有位置超限警告

Test 3: Action Chunk Execution
- 推理得到16步动作序列
- 一次执行完整chunk（~500ms）
- 记录每一步的执行时间
- 验证整体执行时间在合理范围内

Test 4: Interrupt Safety
- 执行chunk中途发送中断信号
- 验证机械臂立即回到当前位置
- 验证没有抖动或超调

Test 5: Error Recovery
- 发送超限动作（测试边界）
- 验证拒绝执行并返回错误
- 验证系统状态仍然安全
```

**测试执行方式**：
```bash
cd /path/to/so101-grasp-cube
uv run python scripts/test_real_safe_execution.py \
  --robot-type so101 \
  --safety-level strict \  # 可选: strict, normal, aggressive
  --max-retries 3
```

**测试验收标准**：
- ✅ Test 1: Home位置精度 <5°，无碰撞
- ✅ Test 2: 小范围动作执行成功
- ✅ Test 3: Chunk执行成功，时间 <600ms
- ✅ Test 4: 中断响应时间 <50ms
- ✅ Test 5: 边界安全校验工作正常

---

## 第3阶段：完整任务执行能力验证

### 目标
验证推理+执行+感知反馈的完整闭环，执行一个简单的真机任务

### 3.1 实现感知反馈检查

**文件位置**：`grasp_cube/real/perception_checker.py`（新建）

**需要实现的内容**：
```
class TaskPerceptionChecker:
    def __init__(self, task_name: str):
        # 不同任务的感知验证逻辑
        # - lift_cube: 检查抓取成功（夹爪闭合+位置变化）
        # - sort_cube: 检查放置位置
        # - stack_cube: 检查堆叠稳定性
        
    def check_grasp_success(self, gripper_state: dict, force_data: dict) -> bool:
        # 验证是否成功抓取物体
        # 条件：夹爪闭合 + 夹爪受力 > 阈值
        
    def check_placement_success(self, object_pose: np.ndarray, target_pose: np.ndarray) -> bool:
        # 验证物体放置位置是否正确
        # 条件：位置偏差 < 5cm，姿态偏差 < 30°
        
    def check_task_completion(self, observation: dict) -> bool:
        # 检查当前是否已完成任务
```

**关键修改点**：
- 集成视觉识别：使用前视摄像头检测物体位置
- 力反馈集成：通过腕部力/扭矩传感器判断抓取
- 目标位置定义：在任务开始前定义目标位置

### 3.2 实现任务执行管理器

**文件位置**：`grasp_cube/real/task_executor.py`（新建）

**需要实现的内容**：
```
class RealRobotTaskExecutor:
    def __init__(self, 
                 task_name: str,
                 inference_wrapper: RealRobotDiffusionInferenceWrapper,
                 action_executor: ActionChunkExecutor,
                 perception_checker: TaskPerceptionChecker):
        pass
        
    def execute_task(self, observation: dict, max_steps: int = 100) -> TaskResult:
        # 主执行循环
        # 逻辑：
        #   current_obs = observation
        #   for step in range(max_steps):
        #       # 1. 推理
        #       action_chunk = inference_wrapper.predict_from_obs(current_obs)
        #       
        #       # 2. 执行
        #       exec_result = action_executor.execute_action_chunk(action_chunk)
        #       
        #       # 3. 观测
        #       current_obs = robot.get_observation()
        #       
        #       # 4. 检查任务完成
        #       if perception_checker.check_task_completion(current_obs):
        #           return TaskResult(success=True, steps=step)
        #       
        #       # 5. 检查失败条件
        #       if perception_checker.check_failure_condition(current_obs):
        #           return TaskResult(success=False, reason="collision")
        #
        #   return TaskResult(success=False, reason="timeout")
```

**关键修改点**：
- 实时推理循环：与真机30Hz同步
- 错误恢复：如果失败，自动回到home位置重试
- 日志记录：记录每一步的观测、推理、执行和反馈

### 3.3 创建任务执行测试脚本

**文件位置**：`scripts/test_real_task_execution.py`

**需要实现的内容**：
```
Test 1: Single Task Execution (Lift Cube)
- 目标：将物体从初始位置抬升10cm
- 流程：
  1. 初始化：放置物体，回到home位置
  2. 执行：运行lift模型推理+执行循环
  3. 验证：检查物体最终位置
  4. 恢复：将物体放回原位
- 成功标准：物体高度提升7-13cm

Test 2: Single Task Execution (Sort Cube) - 双臂
- 目标：将双臂物体移动到指定位置
- 验证：物体最终位置偏差<5cm

Test 3: Repeated Task (5次)
- 执行same task 5次
- 统计成功率和平均执行步数
- 验证系统稳定性

Test 4: Task Switching
- 依次执行lift → sort → stack
- 验证模型切换不影响执行质量
```

**测试执行方式**：
```bash
cd /path/to/so101-grasp-cube
uv run python scripts/test_real_task_execution.py \
  --task lift \           # 或 sort, stack
  --num-episodes 3 \
  --device cuda \
  --save-video
```

**测试验收标准**：
- ✅ Test 1: Lift成功率 ≥80%，高度 7-13cm
- ✅ Test 2: Sort成功率 ≥80%，位置偏差 <5cm
- ✅ Test 3: 5次重复成功率 ≥80%
- ✅ Test 4: 任务切换执行正常

---

## 第4阶段：完整Pick-Place任务执行

### 目标
执行完整的抓取-搬运-放置任务序列，验证模型在真机复杂任务中的有效性

### 4.1 扩展任务定义

**文件位置**：`grasp_cube/real/task_definitions.py`（新建）

**需要实现的内容**：
```
# 定义所有任务的执行步骤、成功条件、安全限制

TaskDefinition:
    - name: 任务名称
    - required_model: 所需的推理模型
    - max_steps: 最大执行步数
    - max_attempts: 失败时的最大重试次数
    - success_criteria: 成功条件（位置精度、力反馈阈值等）
    - safety_limits: 安全限制（关节限制、碰撞距离等）
    - target_objects: 要操作的物体列表
    
# 为每个任务定义subtask序列
# 例如 pick_place_cube:
#   1. Move to object: 使用motion planner
#   2. Approach: 使用lift model缓慢逼近
#   3. Grasp: 夹爪闭合
#   4. Lift: 使用lift model提升
#   5. Move to target: Motion planning
#   6. Release: 松开夹爪
```

### 4.2 创建完整任务执行脚本

**文件位置**：`scripts/eval_real_diffusion_policy.py`（修改现有的eval_real_policy.py）

**需要实现的内容**：
```
Main Evaluation Pipeline:
1. 初始化：
   - 连接真机（SO101或BI-SO101）
   - 加载对应任务的推理模型
   - 连接传感器（摄像头、力传感器）
   
2. 循环（num_episodes次）：
   - reset：将机械臂回到home位置，放置新物体
   - execute：
     for step in range(max_steps):
       obs = robot.get_observation()
       action_chunk = inference_engine.predict(obs)
       robot.execute(action_chunk)
       check task completion
   - record：记录video、metrics、失败原因
   
3. 统计和保存：
   - 成功率
   - 平均步数
   - 平均执行时间
   - 失败原因分布
```

**关键修改点**：
- 参考 `scripts/eval_real_policy.py` 的框架，但改为使用 DiffusionPolicyInferenceEngine
- 集成 `grasp_cube/real/lerobot_env.py` 的观测处理
- 集成视频记录（EvalRecordWrapper）

### 4.3 性能基准测试

**文件位置**：`scripts/benchmark_real_policy.py`

**需要实现的内容**：
```
Benchmark测试：
1. 推理延迟基准
   - 测试单次推理的平均延迟
   - CPU vs GPU延迟对比
   - 不同图像分辨率的延迟影响
   
2. 端到端延迟
   - 从传感器读取到动作执行的总延迟
   - 闭环控制频率测试（目标30Hz）
   
3. 成功率基准
   - 每个任务执行10次，统计成功率
   - 记录失败原因（重力、碰撞、超时等）
   
4. 鲁棒性测试
   - 不同初始物体位置（10个位置）
   - 不同光线条件（亮/暗）
   - 不同物体（如有多个相似物体可用）

输出：Benchmark Report (JSON/CSV)
```

---

## 第5阶段：系统集成和优化

### 5.1 集成到run_env_client.py

**文件位置**：修改 `grasp_cube/real/run_env_client.py`

**需要实现的内容**：
```
# 在run_env_client.py中支持DiffusionPolicy推理引擎
# 当前的框架是：env.step() -> policy.get_actions() -> action

# 修改点：
1. 添加--policy-type参数（act vs diffusion）
2. 当policy_type=diffusion时，加载DiffusionPolicyInferenceEngine
3. 适配接口：
   - DiffusionPolicyInferenceEngine.predict() 
   - 返回action_chunk (horizon, action_dim)
   - 需要在client端进行action sequencing
```

### 5.2 Docker打包

**文件位置**：参考 `docker_tutorial.md`，创建 `Dockerfile.diffusion`

**需要实现的内容**：
```
# Dockerfile关键部分：
1. 基础镜像：pytorch:2.0-cuda11.8
2. 依赖：
   - lerobot及其依赖
   - grasp_cube包
   - env_client包
3. 启动命令：python -m grasp_cube.real.serve_diffusion_policy
```

### 5.3 模型性能优化（可选）

**如果推理时间过长，考虑以下优化**：
```
1. 模型量化：转换为int8或float16
2. ONNX导出：PyTorch → ONNX → TensorRT（NVIDIA GPU优化）
3. 批推理：如果允许多个机械臂，可以批处理
4. 缓存优化：预先warming up GPU显存
```

---

## 完整的文件修改清单

### 需要新建的文件
1. `scripts/test_real_sensor_input.py` - 真机传感器读取测试
2. `scripts/test_real_inference.py` - 真机推理测试
3. `scripts/test_real_safe_execution.py` - 安全动作执行测试
4. `scripts/test_real_task_execution.py` - 完整任务执行测试
5. `scripts/eval_real_diffusion_policy.py` - 最终评估脚本
6. `scripts/benchmark_real_policy.py` - 性能基准测试
7. `grasp_cube/real/diffusion_inference_wrapper.py` - 推理包装器
8. `grasp_cube/real/action_executor.py` - 动作执行器
9. `grasp_cube/real/task_executor.py` - 任务执行管理
10. `grasp_cube/real/perception_checker.py` - 感知反馈检查
11. `grasp_cube/real/task_definitions.py` - 任务定义
12. `Dockerfile.diffusion` - Docker打包

### 需要修改的文件
1. `grasp_cube/real/run_env_client.py` - 添加DiffusionPolicy支持
2. `scripts/serve_act_policy.py` - 参考实现新的serve_diffusion_policy.py
3. `grasp_cube/real/__init__.py` - 导出新增的模块

---

## 关键代码参考

### 从scripts/inference_engine.py中的关键方法
```python
# 已有的推理接口
engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
actions = engine.predict(image, state)  # 返回 (16, 6) 或 (16, 12)
```

### 从grasp_cube/real/lerobot_env.py中的关键接口
```python
# 环境观测格式
observation = {
    "images": {
        "front": np.ndarray (480, 640, 3) uint8,
        "wrist": np.ndarray (480, 640, 3) uint8
    },
    "states": {
        "arm": np.ndarray (6,) float32  # 单臂
        # 或
        "left_arm": np.ndarray (6,) float32,  # 双臂
        "right_arm": np.ndarray (6,) float32
    },
    "task": str  # 任务名称
}

# 环境step接口
obs, reward, terminated, truncated, info = env.step(action)
```

### 从scripts/test_offline_inference.py中的测试模式
```python
# 参考test_offline_inference.py的测试结构
# 每个test函数独立验证一个功能
# 使用时间测量统计性能
# 捕获异常并优雅降级
```

---

## 时间线估计

| 阶段 | 任务 | 预计耗时 | 风险 |
|------|------|---------|------|
| 1 | 传感器数据读取 | 2-4小时 | 摄像头/机械臂连接问题 |
| 1 | 推理能力验证 | 1-2小时 | 推理引擎与真机数据格式不匹配 |
| 2 | 动作执行器实现 | 3-6小时 | 关节映射、安全限制定义 |
| 2 | 安全验证测试 | 2-4小时 | 需要多次真机调试 |
| 3 | 感知反馈集成 | 4-8小时 | 视觉识别、力反馈需要调参 |
| 3 | 完整任务执行 | 4-8小时 | 闭环控制稳定性问题 |
| 4 | Pick-Place任务 | 4-6小时 | 任务复杂度高，需迭代 |
| 5 | 系统集成优化 | 2-4小时 | 集成测试 |
| **总计** | | **22-42小时** | |

---

## 关键风险点和缓解方案

### 风险1：推理延迟过长（>1秒）
**缓解**：
- 检查GPU内存，确保模型完全加载
- 使用FP16混合精度训练
- 考虑模型蒸馏到更小模型

### 风险2：推理输出与真机约束不匹配
**缓解**：
- 在inference_engine中添加硬约束（clamp）
- 设计action_executor的映射函数时考虑real-to-sim gap
- 使用domain randomization的训练模型

### 风险3：闭环控制不稳定
**缓解**：
- 使用PID控制层平滑推理输出
- 添加轨迹平滑器（低通滤波）
- 实现简单的反馈控制

### 风险4：任务完成检测失败
**缓解**：
- 添加超时保护（max_steps限制）
- 实现多模态任务完成检测（视觉+力反馈）
- 记录失败case用于后续调试

---

## 调试建议

### 1. 从最简单的开始
先从lift_cube开始（单臂、简单任务），再到sort和stack

### 2. 逐步增加复杂度
- 第1阶段：不执行动作，只看推理
- 第2阶段：执行小幅度动作
- 第3阶段：执行完整任务

### 3. 充分的日志和可视化
- 推理的输入/输出visualization
- 每步执行的action和实际位置对比
- 任务完成度的实时显示

### 4. 安全第一
- 始终有emergency_stop按钮
- 每次真机测试前都在empty workspace
- 关节速度限制要保守

