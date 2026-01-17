#!/usr/bin/env python3
"""
真机集成适配器
将 DiffusionPolicy 推理引擎集成到现有的机器人控制系统
"""
import numpy as np
import time
from typing import Tuple, Optional, Dict
from collections import deque
from scripts.inference_engine import DiffusionPolicyInferenceEngine, load_multi_task_models


class RobotPolicyController:
    """
    机器人策略控制器
    处理推理、平滑化、安全检查等
    """
    
    def __init__(
        self,
        checkpoints_dir: str = "checkpoints",
        device: str = "cuda",
        action_smoothing: int = 1,  # 动作平均窗口大小（1 = 无平滑）
        action_scale: float = 1.0,  # 动作缩放因子（<1 表示减速）
        verbose: bool = True
    ):
        """
        初始化控制器
        
        Args:
            checkpoints_dir: 模型检查点目录
            device: "cuda" 或 "cpu"
            action_smoothing: 平滑窗口大小（较大值=更平滑但延迟更大）
            action_scale: 动作缩放（用于安全测试，逐步增加）
            verbose: 打印信息
        """
        self.device = device
        self.action_smoothing = action_smoothing
        self.action_scale = action_scale
        self.verbose = verbose
        
        # 加载所有模型
        self.models = load_multi_task_models(checkpoints_dir, device=device)
        if not self.models:
            raise RuntimeError("Failed to load models")
        
        self.current_task = None
        self.action_history = deque(maxlen=action_smoothing)
        self.inference_times = deque(maxlen=100)  # 记录最近 100 次推理时间
        
        if self.verbose:
            print(f"✓ Policy controller initialized")
            print(f"  Loaded tasks: {list(self.models.keys())}")
            print(f"  Action smoothing: {action_smoothing}")
            print(f"  Action scale: {action_scale:.1%}")
    
    def set_task(self, task_name: str) -> bool:
        """切换任务"""
        if task_name not in self.models:
            print(f"✗ Task '{task_name}' not found. Available: {list(self.models.keys())}")
            return False
        
        self.current_task = task_name
        self.action_history.clear()
        
        if self.verbose:
            engine = self.models[task_name]
            print(f"✓ Switched to '{task_name}' task")
            print(f"  State dim: {engine.state_dim}")
            print(f"  Action dim: {engine.action_dim}")
            print(f"  Horizon: {engine.horizon}")
        
        return True
    
    def predict_action(
        self,
        image: np.ndarray,
        state: np.ndarray,
        apply_smoothing: bool = True,
        apply_scale: bool = True
    ) -> np.ndarray:
        """
        预测下一个动作
        
        Args:
            image: (3, 480, 640) RGB 图像 [0, 1]
            state: 关节状态向量
            apply_smoothing: 是否应用平滑化
            apply_scale: 是否应用速度缩放
            
        Returns:
            action: (action_dim,) 动作向量，可直接发送给机器人
        """
        if self.current_task is None:
            raise RuntimeError("No task selected. Call set_task() first.")
        
        engine = self.models[self.current_task]
        
        # 推理
        start = time.time()
        action_seq = engine.predict(image, state)  # (horizon, action_dim)
        elapsed = time.time() - start
        self.inference_times.append(elapsed)
        
        # 提取第一个动作
        action = action_seq[0].copy()
        
        # 应用平滑化（移动平均）
        if apply_smoothing and self.action_smoothing > 1:
            self.action_history.append(action)
            action = np.mean(self.action_history, axis=0)
        
        # 应用速度缩放（用于逐步增加）
        if apply_scale and self.action_scale < 1.0:
            action = action * self.action_scale
        
        return action
    
    def predict_action_sequence(
        self,
        image: np.ndarray,
        state: np.ndarray
    ) -> np.ndarray:
        """
        获取完整的动作序列（用于提前规划）
        
        Returns:
            actions: (horizon, action_dim)
        """
        if self.current_task is None:
            raise RuntimeError("No task selected. Call set_task() first.")
        
        engine = self.models[self.current_task]
        return engine.predict(image, state)
    
    def get_inference_stats(self) -> Dict:
        """获取推理性能统计"""
        times = list(self.inference_times)
        if not times:
            return {}
        
        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "samples": len(times),
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_inference_stats()
        if stats:
            print(f"\n推理性能统计（基于 {stats['samples']} 次样本）:")
            print(f"  平均: {stats['mean_ms']:.2f}ms")
            print(f"  标准差: {stats['std_ms']:.2f}ms")
            print(f"  范围: [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}]ms")
            
            if stats['mean_ms'] < 50:
                print(f"  ✓ 性能良好 (可实时控制 30Hz)")
            elif stats['mean_ms'] < 100:
                print(f"  ✓ 性能可接受 (可实时控制 10Hz)")
            else:
                print(f"  ⚠ 性能不理想 (>100ms 延迟较大)")


# 集成示例：与真机控制系统的连接
class SO101RobotPolicyController:
    """
    SO-101 机器人 + DiffusionPolicy 集成控制器
    
    使用说明：
    --------
    controller = SO101RobotPolicyController()
    controller.set_task("lift")
    
    # 在主控制循环中
    while True:
        image = camera.get_frame()  # 获取图像
        state = robot.get_state()    # 获取关节角度
        
        action = controller.predict_action(image, state)
        robot.execute_action(action)
    """
    
    def __init__(
        self,
        # 机器人相关参数
        robot_interface=None,  # 你的真机接口对象
        camera_interface=None,  # 你的相机接口对象
        # 推理相关参数
        checkpoints_dir: str = "checkpoints",
        device: str = "cuda",
        # 安全参数
        action_scale: float = 1.0,  # 1.0 = 100% 速度，0.1 = 10% 安全测试
        action_clip: bool = True,    # 是否限制动作范围
        collision_detection: bool = False,  # 是否启用碰撞检测
    ):
        """
        初始化机器人控制器
        """
        self.robot = robot_interface
        self.camera = camera_interface
        self.policy_controller = RobotPolicyController(
            checkpoints_dir=checkpoints_dir,
            device=device,
            action_scale=action_scale,
        )
        self.action_clip = action_clip
        self.collision_detection = collision_detection
        
        print("✓ SO-101 Robot Policy Controller initialized")
    
    def step(self, task_name: str) -> bool:
        """
        执行一个控制步骤
        
        Returns:
            success: 是否执行成功
        """
        try:
            # 设置任务
            self.policy_controller.set_task(task_name)
            
            # 获取观测
            if self.camera is None or self.robot is None:
                print("⚠ Robot/Camera interface not implemented")
                return False
            
            image = self.camera.get_frame()  # 应该是 (3, 480, 640) float32
            state = self.robot.get_state()   # 应该是 (state_dim,) float32
            
            # 预测动作
            action = self.policy_controller.predict_action(image, state)
            
            # 安全检查
            if self.action_clip:
                action = self._clip_action(action)
            
            if self.collision_detection:
                if self._check_collision(state, action):
                    print("⚠ Collision detected! Stopping.")
                    return False
            
            # 执行动作
            self.robot.execute_action(action)
            
            return True
            
        except Exception as e:
            print(f"✗ Step failed: {e}")
            return False
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        限制动作范围（防止机器人超速）
        """
        # TODO: 根据你的机器人规格调整这些限制
        MAX_VELOCITY = 1.0  # rad/s
        action = np.clip(action, -MAX_VELOCITY, MAX_VELOCITY)
        return action
    
    def _check_collision(self, state: np.ndarray, action: np.ndarray) -> bool:
        """
        检测潜在碰撞
        """
        # TODO: 实现碰撞检测逻辑
        # 这取决于你的机器人模型和传感器
        return False
    
    def run_task(
        self,
        task_name: str,
        max_steps: int = 500,
        verbose: bool = True
    ) -> Dict:
        """
        运行完整任务
        
        Returns:
            {
                "task": task_name,
                "steps": 实际执行的步骤数,
                "success_rate": 成功率,
                "inference_stats": 推理统计,
            }
        """
        self.policy_controller.set_task(task_name)
        
        success_count = 0
        for step in range(max_steps):
            if self.step(task_name):
                success_count += 1
            
            if verbose and (step + 1) % 100 == 0:
                success_rate = success_count / (step + 1) * 100
                print(f"  Step {step+1}: {success_rate:.1f}% success rate")
        
        stats = self.policy_controller.get_inference_stats()
        
        return {
            "task": task_name,
            "steps": max_steps,
            "success_rate": success_count / max_steps * 100,
            "inference_stats": stats,
        }


if __name__ == "__main__":
    # 演示：不需要真实机器人也能测试
    print("RobotPolicyController Demo (without real robot)\n")
    
    controller = RobotPolicyController(action_smoothing=3, action_scale=0.5)
    
    # 选择任务
    controller.set_task("lift")
    
    # 模拟推理循环
    print("Running 10 simulated steps...\n")
    for step in range(10):
        # 模拟观测
        image = np.random.rand(3, 480, 640).astype(np.float32)
        state = np.random.randn(6).astype(np.float32)
        
        # 预测动作
        action = controller.predict_action(image, state)
        
        print(f"Step {step+1}: action = {action}")
    
    # 打印性能统计
    controller.print_stats()
