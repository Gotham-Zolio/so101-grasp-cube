"""
真机DiffusionPolicy推理包装器

此模块提供了一个包装器类，将DiffusionPolicyInferenceEngine集成到真机环境中，
处理观测数据的转换、推理以及动作输出的处理。

核心功能：
1. 从真机观测dict中提取和预处理数据
2. 调用推理引擎获取动作序列
3. 处理多任务（不同state_dim）的情况
4. 提供action chunking支持
"""

import numpy as np
import pathlib
import sys
from typing import Dict, Tuple, Optional, Any
import torch

# 导入推理引擎
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference_engine import DiffusionPolicyInferenceEngine


class RealRobotDiffusionInferenceWrapper:
    """
    真机DiffusionPolicy推理包装器
    
    将DiffusionPolicyInferenceEngine集成到真机环境中，处理：
    - 观测数据的预处理（图像转换、状态维度适配）
    - 推理输出的后处理
    - 任务切换时的模型加载和状态维度适配
    - Action chunking的管理
    """
    
    def __init__(
        self,
        task_name: str,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        初始化推理包装器
        
        Args:
            task_name: 任务名称 ("lift", "sort", "stack")
            device: 计算设备 ("cuda" 或 "cpu")
            verbose: 是否打印详细信息
        """
        self.task_name = task_name
        self.device = device
        self.verbose = verbose
        
        # 任务到模型路径的映射
        self.task_to_checkpoint = {
            "lift": "checkpoints/lift_real/checkpoint-best",
            "sort": "checkpoints/sort_real/checkpoint-best",
            "stack": "checkpoints/stack_real/checkpoint-best",
        }
        
        if task_name not in self.task_to_checkpoint:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.task_to_checkpoint.keys())}")
        
        # 初始化推理引擎
        if self.verbose:
            print(f"Initializing DiffusionPolicy for task: {task_name}")
        
        self.engine = DiffusionPolicyInferenceEngine(
            self.task_to_checkpoint[task_name],
            device=device,
            verbose=verbose
        )
        
        # Action chunk管理
        self.action_chunk = None  # 当前的动作序列
        self.chunk_index = 0  # 当前执行到的位置
        
        if self.verbose:
            print(f"✓ Wrapper initialized for task: {task_name}")
            print(f"  Model state_dim: {self.engine.state_dim}")
            print(f"  Model action_dim: {self.engine.action_dim}")
            print(f"  Horizon: {self.engine.horizon}")
    
    def switch_task(self, new_task_name: str) -> bool:
        """
        切换到不同的任务模型
        
        Args:
            new_task_name: 新任务名称
            
        Returns:
            success: 是否切换成功
        """
        if new_task_name not in self.task_to_checkpoint:
            print(f"✗ Unknown task: {new_task_name}")
            return False
        
        if new_task_name == self.task_name:
            if self.verbose:
                print(f"Already on task: {new_task_name}")
            return True
        
        try:
            if self.verbose:
                print(f"Switching from {self.task_name} to {new_task_name}")
            
            # 加载新模型
            self.engine = DiffusionPolicyInferenceEngine(
                self.task_to_checkpoint[new_task_name],
                device=self.device,
                verbose=self.verbose
            )
            
            self.task_name = new_task_name
            self.action_chunk = None
            self.chunk_index = 0
            
            if self.verbose:
                print(f"✓ Successfully switched to {new_task_name}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to switch task: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: RGB图像，可能是：
                   - (H, W, 3) uint8 从真机摄像头
                   - (3, H, W) float32 已转换
                   
        Returns:
            processed: (3, H, W) float32 [0, 1] 范围
        """
        # 处理数据类型
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        # 处理维度
        if image_float.ndim == 3:
            if image_float.shape[-1] == 3:
                # (H, W, 3) → (3, H, W)
                image_float = np.transpose(image_float, (2, 0, 1))
        elif image_float.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image_float.ndim}D")
        
        return image_float
    
    def extract_state_from_observation(
        self,
        observation: Dict[str, Any]
    ) -> np.ndarray:
        """
        从观测dict中提取状态向量
        
        Args:
            observation: 观测dict，应包含：
                - "states": 包含关节状态的dict
                  - "arm" (单臂) 或
                  - "left_arm"/"right_arm" (双臂)
        
        Returns:
            state: (state_dim,) float32 状态向量
            
        Raises:
            ValueError: 如果无法提取状态
        """
        if "states" not in observation:
            raise ValueError("Observation missing 'states' key")
        
        states_dict = observation["states"]
        
        # 确定robottype
        if "arm" in states_dict:
            # 单臂
            state = states_dict["arm"]
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
        elif "left_arm" in states_dict and "right_arm" in states_dict:
            # 双臂：连接左臂和右臂状态
            left_arm = states_dict["left_arm"]
            right_arm = states_dict["right_arm"]
            
            if not isinstance(left_arm, np.ndarray):
                left_arm = np.array(left_arm, dtype=np.float32)
            if not isinstance(right_arm, np.ndarray):
                right_arm = np.array(right_arm, dtype=np.float32)
            
            state = np.concatenate([left_arm, right_arm], axis=-1)
        else:
            raise ValueError(
                f"Cannot extract state from observation. "
                f"Expected 'arm' or 'left_arm'/'right_arm', got keys: {list(states_dict.keys())}"
            )
        
        return state.astype(np.float32)
    
    def extract_image_from_observation(
        self,
        observation: Dict[str, Any],
        camera_key: str = "front"
    ) -> np.ndarray:
        """
        从观测dict中提取图像
        
        Args:
            observation: 观测dict，应包含：
                - "images": 包含摄像头图像的dict
                  - "front", "wrist", "left_wrist", "right_wrist" 等
            camera_key: 使用哪个摄像头的图像 (默认: "front")
        
        Returns:
            image: (3, H, W) float32 [0, 1] 预处理后的图像
            
        Raises:
            ValueError: 如果无法提取图像
        """
        if "images" not in observation:
            raise ValueError("Observation missing 'images' key")
        
        images_dict = observation["images"]
        
        if camera_key not in images_dict:
            available_keys = list(images_dict.keys())
            # 如果指定的key不存在，尝试使用第一个可用的
            if available_keys:
                camera_key = available_keys[0]
                if self.verbose:
                    print(f"⚠ Camera '{camera_key}' not found, using '{camera_key}' instead")
            else:
                raise ValueError(f"No images found in observation")
        
        image = images_dict[camera_key]
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image is not numpy array, got {type(image)}")
        
        return self.preprocess_image(image)
    
    def predict_from_obs(
        self,
        observation: Dict[str, Any],
        camera_key: str = "front",
        force_new_chunk: bool = False
    ) -> np.ndarray:
        """
        从观测dict预测动作序列
        
        Args:
            observation: 观测dict，包含 "images" 和 "states"
            camera_key: 使用的摄像头 (默认: "front")
            force_new_chunk: 是否强制生成新的动作chunk
                            (默认: False，如果当前chunk未用完则继续用)
        
        Returns:
            action_chunk: (horizon, action_dim) 动作序列
            
        Raises:
            ValueError: 如果观测数据无效
        """
        # 如果有未完成的chunk且不是强制重新生成，直接返回
        if self.action_chunk is not None and not force_new_chunk:
            if self.chunk_index < len(self.action_chunk):
                return self.action_chunk
        
        # 提取和预处理数据
        try:
            image = self.extract_image_from_observation(observation, camera_key)
            state = self.extract_state_from_observation(observation)
        except Exception as e:
            raise ValueError(f"Failed to extract observation data: {e}")
        
        # 推理
        try:
            action_chunk = self.engine.predict(image, state)
        except Exception as e:
            raise ValueError(f"Inference failed: {e}")
        
        # 重置chunk索引
        self.action_chunk = action_chunk
        self.chunk_index = 0
        
        return action_chunk
    
    def get_next_action(
        self,
        observation: Dict[str, Any],
        camera_key: str = "front"
    ) -> Tuple[np.ndarray, int]:
        """
        获取下一个要执行的动作
        
        此方法用于行为策略部署，每次返回chunk中的一个动作。
        
        Args:
            observation: 观测dict
            camera_key: 使用的摄像头
        
        Returns:
            (action, remaining_actions):
                - action: (action_dim,) 单个动作
                - remaining_actions: 当前chunk中还有多少个动作未执行
        """
        # 如果没有chunk或chunk已用完，生成新的
        if self.action_chunk is None or self.chunk_index >= len(self.action_chunk):
            try:
                self.predict_from_obs(observation, camera_key, force_new_chunk=True)
            except Exception as e:
                print(f"✗ Failed to generate action chunk: {e}")
                raise
        
        # 获取当前动作
        action = self.action_chunk[self.chunk_index].copy()
        
        # 更新索引
        self.chunk_index += 1
        remaining = len(self.action_chunk) - self.chunk_index
        
        return action, remaining
    
    def has_pending_actions(self) -> bool:
        """
        检查是否还有未执行的动作
        
        Returns:
            has_pending: 是否还有待执行的动作
        """
        if self.action_chunk is None:
            return False
        return self.chunk_index < len(self.action_chunk)
    
    def reset_chunk(self) -> None:
        """重置动作chunk状态"""
        self.action_chunk = None
        self.chunk_index = 0
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        获取调试信息
        
        Returns:
            debug_info: 包含当前状态的字典
        """
        return {
            "task_name": self.task_name,
            "device": str(self.device),
            "model_state_dim": self.engine.state_dim,
            "model_action_dim": self.engine.action_dim,
            "horizon": self.engine.horizon,
            "has_chunk": self.action_chunk is not None,
            "chunk_index": self.chunk_index,
            "chunk_size": len(self.action_chunk) if self.action_chunk is not None else 0,
            "remaining_actions": (
                len(self.action_chunk) - self.chunk_index 
                if self.action_chunk is not None and self.chunk_index < len(self.action_chunk) 
                else 0
            )
        }


if __name__ == "__main__":
    """简单测试"""
    
    print("Testing RealRobotDiffusionInferenceWrapper...")
    
    # 创建包装器
    wrapper = RealRobotDiffusionInferenceWrapper(
        task_name="lift",
        device="cuda",
        verbose=True
    )
    
    # 生成模拟观测
    mock_observation = {
        "images": {
            "front": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        },
        "states": {
            "arm": np.random.randn(6).astype(np.float32)
        }
    }
    
    # 预测
    print("\nTesting prediction...")
    try:
        actions = wrapper.predict_from_obs(mock_observation)
        print(f"✓ Prediction successful: {actions.shape}")
        
        # 逐个获取动作
        print("\nTesting action sequencing...")
        for i in range(3):
            action, remaining = wrapper.get_next_action(mock_observation)
            print(f"  Step {i}: action shape {action.shape}, remaining {remaining}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试任务切换
    print("\nTesting task switching...")
    wrapper.switch_task("sort")
    print(f"✓ Switched to sort task")
    
    # 获取调试信息
    print("\nDebug info:")
    debug_info = wrapper.get_debug_info()
    for key, value in debug_info.items():
        print(f"  {key}: {value}")
