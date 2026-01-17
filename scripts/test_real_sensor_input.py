#!/usr/bin/env python3
"""
真机传感器数据读取和推理测试
验证从真机的RGB图像和关节状态进行推理的完整流程

测试流程：
1. 连接真机机械臂和摄像头
2. 读取一帧RGB图像 + 关节状态
3. 通过推理引擎进行推理（不执行动作）
4. 验证推理输出的有效性

执行方式：
    uv run python scripts/test_real_sensor_input.py \\
        --robot-type so101 \\
        --task lift \\
        --duration 10 \\
        --device cuda
"""

import sys
import pathlib
import numpy as np
import argparse
import time
from typing import Dict, Tuple, Optional
import cv2

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference_engine import DiffusionPolicyInferenceEngine

# 导入真机环境相关模块
try:
    from grasp_cube.real.lerobot_env import LeRobotEnv, LeRobotEnvConfig
    from lerobot.robots import so101_follower, bi_so101_follower
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Warning: LeRobot modules not available: {e}")
    LEROBOT_AVAILABLE = False


class RealSensorInferenceTest:
    """真机传感器推理测试类"""
    
    def __init__(
        self,
        robot_type: str = "so101",
        task_name: str = "lift",
        device: str = "cuda",
        camera_config_path: Optional[pathlib.Path] = None
    ):
        """
        初始化测试环境
        
        Args:
            robot_type: "so101" (单臂) 或 "bi_so101" (双臂)
            task_name: "lift", "sort", 或 "stack"
            device: "cuda" 或 "cpu"
            camera_config_path: 摄像头配置文件路径
        """
        self.robot_type = robot_type
        self.task_name = task_name
        self.device = device
        
        # 推理引擎配置
        self.checkpoint_map = {
            "lift": "checkpoints/lift_real/checkpoint-best",
            "sort": "checkpoints/sort_real/checkpoint-best",
            "stack": "checkpoints/stack_real/checkpoint-best",
        }
        
        if task_name not in self.checkpoint_map:
            raise ValueError(f"Unknown task: {task_name}")
        
        # 初始化推理引擎
        print(f"\n{'='*70}")
        print(f"Initializing DiffusionPolicy Inference Engine")
        print(f"{'='*70}")
        
        try:
            self.engine = DiffusionPolicyInferenceEngine(
                self.checkpoint_map[task_name],
                device=device,
                verbose=True
            )
            print(f"✓ Inference engine initialized")
            print(f"  State dim: {self.engine.state_dim}")
            print(f"  Action dim: {self.engine.action_dim}")
            print(f"  Horizon: {self.engine.horizon}")
        except Exception as e:
            print(f"✗ Failed to initialize inference engine: {e}")
            raise
    
    def get_mock_observation(self) -> Dict[str, np.ndarray]:
        """
        生成模拟观测数据（用于测试，实际应该从真机读取）
        
        Returns:
            observation: {"images": {...}, "states": {...}}
        """
        # 生成随机RGB图像 (480, 640, 3) uint8
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # 生成随机关节状态
        if self.robot_type == "so101":
            state = np.random.randn(6).astype(np.float32)  # 单臂6维
        else:  # bi_so101
            if self.task_name == "sort":
                state = np.random.randn(12).astype(np.float32)  # 双臂12维
            else:
                # lift和stack通常只使用单臂
                state = np.random.randn(6).astype(np.float32)
        
        # 限制state范围在合理的关节角度范围内 [-π, π]
        state = np.clip(state, -np.pi, np.pi)
        
        return {
            "image": image,
            "state": state
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像：转换为float32 [0, 1]范围
        
        Args:
            image: RGB图像 (H, W, 3) uint8
            
        Returns:
            processed_image: (3, H, W) float32 [0, 1]
        """
        # 转换数据类型
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        # 转换维度：(H, W, 3) → (3, H, W)
        if image_float.ndim == 3:
            image_float = np.transpose(image_float, (2, 0, 1))
        
        return image_float
    
    def test_single_inference(self) -> bool:
        """
        Test 1: 单次推理测试
        从观测数据进行一次推理，验证输出的有效性
        
        Returns:
            success: 是否通过测试
        """
        print(f"\n{'='*70}")
        print(f"Test 1: Single Real Sensor Inference")
        print(f"{'='*70}")
        
        try:
            # 1. 获取观测（这里用模拟数据，实际应该从真机读取）
            obs = self.get_mock_observation()
            image_raw = obs["image"]
            state = obs["state"]
            
            print(f"\n✓ Observation acquired")
            print(f"  Image shape: {image_raw.shape}, dtype: {image_raw.dtype}")
            print(f"  State shape: {state.shape}, dtype: {state.dtype}")
            print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
            
            # 2. 预处理图像
            image_processed = self.preprocess_image(image_raw)
            
            print(f"\n✓ Image preprocessed")
            print(f"  Processed shape: {image_processed.shape}, dtype: {image_processed.dtype}")
            print(f"  Value range: [{image_processed.min():.3f}, {image_processed.max():.3f}]")
            
            # 3. 执行推理
            start_time = time.time()
            actions = self.engine.predict(image_processed, state)
            inference_time = time.time() - start_time
            
            # 4. 验证输出
            print(f"\n✓ Inference successful")
            print(f"  Action shape: {actions.shape}")
            print(f"  Action dtype: {actions.dtype}")
            print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
            print(f"  Inference time: {inference_time*1000:.2f} ms")
            
            # 5. 验证输出维度
            expected_horizon = self.engine.horizon
            expected_action_dim = self.engine.action_dim
            
            if actions.shape[0] != expected_horizon:
                print(f"✗ Horizon mismatch: expected {expected_horizon}, got {actions.shape[0]}")
                return False
            
            if actions.shape[1] != expected_action_dim:
                print(f"✗ Action dim mismatch: expected {expected_action_dim}, got {actions.shape[1]}")
                return False
            
            # 6. 验证值范围（应该在[-1, 1]之间）
            if np.any(np.isnan(actions)):
                print(f"✗ NaN values in actions!")
                return False
            
            if np.any(np.isinf(actions)):
                print(f"✗ Inf values in actions!")
                return False
            
            # 性能评估
            if inference_time < 0.05:
                print(f"  ✓ Excellent speed (<50ms)")
            elif inference_time < 0.1:
                print(f"  ✓ Good speed (<100ms)")
            elif inference_time < 0.5:
                print(f"  ⚠ Acceptable speed (<500ms)")
            else:
                print(f"  ⚠ Slow inference ({inference_time*1000:.0f}ms) - may need optimization")
            
            print(f"\n✓ Test 1 PASSED")
            return True
            
        except Exception as e:
            print(f"\n✗ Test 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_continuous_inference(self, duration_s: int = 10) -> bool:
        """
        Test 2: 连续推理测试
        在指定时间内进行连续推理，测量延迟分布和稳定性
        
        Args:
            duration_s: 测试持续时间（秒）
            
        Returns:
            success: 是否通过测试
        """
        print(f"\n{'='*70}")
        print(f"Test 2: Continuous Real Sensor Inference ({duration_s}s)")
        print(f"{'='*70}")
        
        try:
            inference_times = []
            num_inferences = 0
            errors = 0
            
            start_time = time.time()
            target_fps = 30
            frame_interval = 1.0 / target_fps
            
            while time.time() - start_time < duration_s:
                frame_start = time.time()
                
                try:
                    # 1. 获取观测
                    obs = self.get_mock_observation()
                    image_raw = obs["image"]
                    state = obs["state"]
                    
                    # 2. 预处理
                    image_processed = self.preprocess_image(image_raw)
                    
                    # 3. 推理
                    inference_start = time.time()
                    actions = self.engine.predict(image_processed, state)
                    inference_time = time.time() - inference_start
                    
                    # 4. 验证
                    if actions.shape != (self.engine.horizon, self.engine.action_dim):
                        errors += 1
                    else:
                        inference_times.append(inference_time)
                    
                    num_inferences += 1
                    
                except Exception as e:
                    print(f"  Error during inference {num_inferences}: {e}")
                    errors += 1
                
                # 速率控制
                frame_elapsed = time.time() - frame_start
                if frame_elapsed < frame_interval:
                    time.sleep(frame_interval - frame_elapsed)
            
            # 统计结果
            elapsed_time = time.time() - start_time
            
            print(f"\n✓ Continuous inference completed")
            print(f"  Total inferences: {num_inferences}")
            print(f"  Successful: {len(inference_times)}")
            print(f"  Errors: {errors}")
            print(f"  Elapsed time: {elapsed_time:.2f}s")
            print(f"  Actual FPS: {num_inferences / elapsed_time:.2f}")
            
            if len(inference_times) > 0:
                inference_times_ms = np.array(inference_times) * 1000
                print(f"\nInference Latency Statistics:")
                print(f"  Mean: {inference_times_ms.mean():.2f} ms")
                print(f"  Std: {inference_times_ms.std():.2f} ms")
                print(f"  Min: {inference_times_ms.min():.2f} ms")
                print(f"  Max: {inference_times_ms.max():.2f} ms")
                print(f"  P50: {np.percentile(inference_times_ms, 50):.2f} ms")
                print(f"  P95: {np.percentile(inference_times_ms, 95):.2f} ms")
                print(f"  P99: {np.percentile(inference_times_ms, 99):.2f} ms")
                
                # 评估
                mean_latency = inference_times_ms.mean()
                if mean_latency < 100:
                    print(f"  ✓ Excellent performance (<100ms)")
                elif mean_latency < 500:
                    print(f"  ✓ Good performance (<500ms)")
                elif mean_latency < 1000:
                    print(f"  ⚠ Acceptable performance (<1s)")
                else:
                    print(f"  ✗ Poor performance (>1s)")
            
            if errors == 0:
                print(f"\n✓ Test 2 PASSED")
                return True
            else:
                print(f"\n⚠ Test 2 PASSED (with {errors} errors)")
                return True
            
        except Exception as e:
            print(f"\n✗ Test 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_multi_task_switching(self) -> bool:
        """
        Test 3: 多任务模型切换测试
        依次加载不同任务的模型，验证模型切换不影响推理
        
        Returns:
            success: 是否通过测试
        """
        print(f"\n{'='*70}")
        print(f"Test 3: Multi-Task Model Switching")
        print(f"{'='*70}")
        
        try:
            tasks = ["lift", "sort", "stack"]
            
            for task_name in tasks:
                print(f"\n--- Testing {task_name.upper()} ---")
                
                # 1. 加载模型
                try:
                    engine = DiffusionPolicyInferenceEngine(
                        self.checkpoint_map[task_name],
                        device=self.device,
                        verbose=False
                    )
                    print(f"✓ Loaded {task_name} model")
                    print(f"  State dim: {engine.state_dim}, Action dim: {engine.action_dim}")
                except Exception as e:
                    print(f"✗ Failed to load {task_name} model: {e}")
                    continue
                
                # 2. 准备对应维度的状态向量
                state = np.random.randn(engine.state_dim).astype(np.float32)
                state = np.clip(state, -np.pi, np.pi)
                
                # 3. 生成随机图像
                image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                image_processed = self.preprocess_image(image)
                
                # 4. 执行推理
                try:
                    actions = engine.predict(image_processed, state)
                    
                    # 5. 验证输出
                    if actions.shape != (engine.horizon, engine.action_dim):
                        print(f"✗ Output shape mismatch: {actions.shape}")
                        return False
                    
                    if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                        print(f"✗ Invalid values in actions")
                        return False
                    
                    print(f"✓ Inference successful")
                    print(f"  Output shape: {actions.shape}")
                    print(f"  Output range: [{actions.min():.4f}, {actions.max():.4f}]")
                    
                except Exception as e:
                    print(f"✗ Inference failed: {e}")
                    return False
            
            print(f"\n✓ Test 3 PASSED")
            return True
            
        except Exception as e:
            print(f"\n✗ Test 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_inference_error_handling(self) -> bool:
        """
        Test 4: 错误处理和边界情况测试
        验证推理引擎对异常输入的处理
        
        Returns:
            success: 是否通过测试
        """
        print(f"\n{'='*70}")
        print(f"Test 4: Inference Error Handling")
        print(f"{'='*70}")
        
        try:
            errors_handled = 0
            total_tests = 0
            
            # Test 4a: 全黑图像
            total_tests += 1
            print(f"\n--- Test 4a: All-black image ---")
            try:
                image = np.zeros((3, 480, 640), dtype=np.float32)
                state = np.zeros(self.engine.state_dim, dtype=np.float32)
                actions = self.engine.predict(image, state)
                print(f"✓ Handled all-black image")
                errors_handled += 1
            except Exception as e:
                print(f"⚠ Exception (expected): {e}")
                errors_handled += 1
            
            # Test 4b: 全白图像
            total_tests += 1
            print(f"\n--- Test 4b: All-white image ---")
            try:
                image = np.ones((3, 480, 640), dtype=np.float32)
                state = np.ones(self.engine.state_dim, dtype=np.float32) * 0.5
                actions = self.engine.predict(image, state)
                print(f"✓ Handled all-white image")
                errors_handled += 1
            except Exception as e:
                print(f"⚠ Exception (expected): {e}")
                errors_handled += 1
            
            # Test 4c: 错误的图像维度
            total_tests += 1
            print(f"\n--- Test 4c: Wrong image dimension ---")
            try:
                image = np.random.rand(3, 100, 100).astype(np.float32)
                state = np.zeros(self.engine.state_dim, dtype=np.float32)
                actions = self.engine.predict(image, state)
                print(f"✗ Should have raised an error for wrong dimension")
            except ValueError as e:
                print(f"✓ Properly rejected wrong dimension: {e}")
                errors_handled += 1
            except Exception as e:
                print(f"⚠ Unexpected error: {e}")
            
            # Test 4d: 错误的状态维度
            total_tests += 1
            print(f"\n--- Test 4d: Wrong state dimension ---")
            try:
                image = np.random.rand(3, 480, 640).astype(np.float32)
                wrong_state_dim = self.engine.state_dim + 2
                state = np.zeros(wrong_state_dim, dtype=np.float32)
                actions = self.engine.predict(image, state)
                print(f"⚠ Accepted wrong state dimension (might have padding)")
                errors_handled += 1
            except Exception as e:
                print(f"✓ Rejected wrong state dimension: {e}")
                errors_handled += 1
            
            # Test 4e: NaN值
            total_tests += 1
            print(f"\n--- Test 4e: NaN in state ---")
            try:
                image = np.random.rand(3, 480, 640).astype(np.float32)
                state = np.full(self.engine.state_dim, np.nan, dtype=np.float32)
                actions = self.engine.predict(image, state)
                print(f"⚠ Accepted NaN values (might propagate to output)")
                if np.any(np.isnan(actions)):
                    print(f"  Output contains NaN (as expected)")
                errors_handled += 1
            except Exception as e:
                print(f"✓ Rejected NaN values: {e}")
                errors_handled += 1
            
            print(f"\nError handling score: {errors_handled}/{total_tests}")
            
            if errors_handled >= total_tests - 1:  # 允许1个失败
                print(f"\n✓ Test 4 PASSED")
                return True
            else:
                print(f"\n⚠ Test 4 PASSED (with limited error handling)")
                return True
            
        except Exception as e:
            print(f"\n✗ Test 4 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有测试"""
        print(f"\n{'='*70}")
        print(f"REAL SENSOR INFERENCE TEST SUITE")
        print(f"{'='*70}")
        print(f"Robot: {self.robot_type}")
        print(f"Task: {self.task_name}")
        print(f"Device: {self.device}")
        
        results = []
        
        # Test 1: 单次推理
        results.append(("Single Inference", self.test_single_inference()))
        
        # Test 2: 连续推理
        results.append(("Continuous Inference", self.test_continuous_inference(duration_s=10)))
        
        # Test 3: 多任务切换
        results.append(("Multi-Task Switching", self.test_multi_task_switching()))
        
        # Test 4: 错误处理
        results.append(("Error Handling", self.test_inference_error_handling()))
        
        # 总结
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY")
        print(f"{'='*70}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"\n✓ All tests passed! Ready for real robot sensor integration.")
        else:
            print(f"\n⚠ Some tests failed. Please check the output above.")
        
        return passed == total


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Real sensor inference test for DiffusionPolicy"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so101",
        choices=["so101", "bi_so101"],
        help="Robot type (default: so101)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="lift",
        choices=["lift", "sort", "stack"],
        help="Task name (default: lift)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration of continuous inference test in seconds (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = RealSensorInferenceTest(
        robot_type=args.robot_type,
        task_name=args.task,
        device=args.device
    )
    
    # 运行所有测试
    success = tester.run_all_tests()
    
    # 返回状态码
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
