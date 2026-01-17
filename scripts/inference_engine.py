#!/usr/bin/env python3
"""
DiffusionPolicy 推理引擎
用于从观测预测动作序列
采用自定义归一化避免LeRobot normalizer的初始化问题
"""
import torch
import pathlib
import json
import numpy as np
from typing import Dict, Optional, Tuple
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


class DiffusionPolicyInferenceEngine:
    """
    DiffusionPolicy 推理引擎
    支持多任务推理，自动处理归一化/反归一化
    手动应用归一化，绕过LeRobot的normalizer缓冲区初始化问题
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: 训练好的模型目录路径 
                       例如: "checkpoints/lift_real/checkpoint-best"
            device: "cuda" 或 "cpu"
            verbose: 是否打印加载信息
        """
        self.device = torch.device(device)
        self.model_path = pathlib.Path(model_path)
        self.verbose = verbose
        
        # 加载模型权重（跳过损坏的normalizer缓冲区）
        # 改为直接加载，不使用from_pretrained的normalize初始化
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Missing key.*")
            self.model = DiffusionPolicy.from_pretrained(str(self.model_path))
        
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为推理模式
        
        # 禁用normalize_inputs和unnormalize_outputs以避免infinity错误
        # 我们将手动应用所有的归一化/反归一化
        self.model.normalize_inputs = torch.nn.Identity()
        self.model.unnormalize_outputs = torch.nn.Identity()
        
        if self.verbose:
            print(f"✓ Model loaded from {model_path}")
        
        # 加载统计信息用于手动归一化
        stats_file = self.model_path / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self.stats = json.load(f)
            if self.verbose:
                print(f"✓ Stats loaded from {stats_file}")
                print(f"  State range: {self.stats.get('observation.state', {})}")
                print(f"  Action range: {self.stats.get('action', {})}")
        else:
            self.stats = {}
            print(f"⚠ Warning: No stats.json found in {stats_file}")
    
    
    def _normalize_inputs_manually(self, batch: Dict):
        """
        手动应用归一化到输入batch
        使用stats.json中的mean/std进行标准化
        动态适配不同的tensor shape
        """
        # 归一化 state
        if "observation.state" in batch and "observation.state" in self.stats:
            state_stats = self.stats["observation.state"]
            state_mean = torch.tensor(state_stats.get("mean", []), dtype=torch.float32).to(self.device)
            state_std = torch.tensor(state_stats.get("std", []), dtype=torch.float32).to(self.device)
            
            # 检查state维度是否匹配
            actual_state_dim = batch["observation.state"].shape[-1]
            stats_state_dim = len(state_mean)
            
            # 如果stats中的state维度与实际不匹配，只对前stats_state_dim维进行归一化
            if stats_state_dim != actual_state_dim:
                if self.verbose:
                    print(f"⚠ State dim mismatch: stats has {stats_state_dim}, actual is {actual_state_dim}")
                
                # 只对前stats_state_dim维进行归一化
                if stats_state_dim < actual_state_dim:
                    # 确保mean/std的shape兼容 (1, stats_state_dim)
                    state_mean = state_mean.unsqueeze(0)
                    state_std = state_std.unsqueeze(0)
                    
                    # 克隆state以避免原地修改广播问题
                    batch_state = batch["observation.state"].clone()
                    batch_state[:, :stats_state_dim] = (
                        batch_state[:, :stats_state_dim] - state_mean
                    ) / (state_std + 1e-6)
                    batch["observation.state"] = batch_state
                    # 后续维度保持不变
                else:
                    # 如果stats_state_dim > actual_state_dim，截断stats
                    state_mean = state_mean[:actual_state_dim].unsqueeze(0)
                    state_std = state_std[:actual_state_dim].unsqueeze(0)
                    batch["observation.state"] = (batch["observation.state"] - state_mean) / (state_std + 1e-6)
            else:
                # 维度匹配，正常归一化
                state_mean = state_mean.unsqueeze(0)  # (1, state_dim)
                state_std = state_std.unsqueeze(0)    # (1, state_dim)
                batch["observation.state"] = (batch["observation.state"] - state_mean) / (state_std + 1e-6)
        
        # 归一化 image
        if "observation.images.front" in batch and "observation.images.front" in self.stats:
            image_stats = self.stats["observation.images.front"]
            image_mean = torch.tensor(image_stats.get("mean", []), dtype=torch.float32).to(self.device)
            image_std = torch.tensor(image_stats.get("std", []), dtype=torch.float32).to(self.device)
            
            # 动态处理shape：可能是(B, T, C, H, W)或(B, C, H, W)
            # 需要确保均值和标准差的形状兼容（使用广播）
            current_shape = batch["observation.images.front"].shape
            if len(current_shape) == 5:  # (B, T, C, H, W)
                # reshape to (1, 1, 3, 1, 1) for broadcasting
                image_mean = image_mean.view(1, 1, -1, 1, 1)
                image_std = image_std.view(1, 1, -1, 1, 1)
            elif len(current_shape) == 4:  # (B, C, H, W)
                # reshape to (1, 3, 1, 1) for broadcasting
                image_mean = image_mean.view(1, -1, 1, 1)
                image_std = image_std.view(1, -1, 1, 1)
            
            batch["observation.images.front"] = (batch["observation.images.front"] - image_mean) / (image_std + 1e-6)
        
        return batch
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,      # (3, 480, 640) or (3, 84, 84) float32 [0, 1]
        state: np.ndarray,      # (state_dim,) float32
        return_raw: bool = False  # 是否返回原始归一化值
    ) -> np.ndarray:
        """
        从图像和状态预测动作序列
        
        Args:
            image: RGB 图像 (3, H, W)，范围 [0, 1]
                   支持 (3, 480, 640) 或 (3, 84, 84) 大小
            state: 关节角度或状态向量，单位应与训练数据一致
            return_raw: 如果 True，返回原始的策略输出（已归一化）
                       如果 False，返回反归一化后的实际动作
        
        Returns:
            actions: (horizon, action_dim) 动作序列
                    如果 return_raw=False，已反归一化到实际范围
                    如果 return_raw=True，为原始归一化值
        """
        # 验证并调整图像大小
        if image.dtype != np.float32:
            raise ValueError(f"Expected float32, got {image.dtype}")
        if not (0 <= image.min() and image.max() <= 1.0):
            print(f"⚠ Warning: Image values outside [0, 1] range: [{image.min():.3f}, {image.max():.3f}]")
        
        # 如果图像是480x640，调整到84x84
        if image.shape == (3, 480, 640):
            # 使用torch的双线性插值调整大小
            image_tensor = torch.from_numpy(image).float()
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),  # 添加batch维度
                size=(84, 84),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # 移除batch维度
            image = image_tensor.cpu().numpy()
        elif image.shape != (3, 84, 84):
            raise ValueError(f"Expected image shape (3, 480, 640) or (3, 84, 84), got {image.shape}")
        
        # 构建批次（添加 batch 和 time step 维度）
        # LeRobot期望: (batch, n_obs_steps, C, H, W)
        batch = {
            "observation.images.front": torch.from_numpy(image)
                .float()
                .to(self.device)
                .unsqueeze(0)           # (1, 3, 84, 84)
                .unsqueeze(0),          # (1, 1, 3, 84, 84)
            "observation.state": torch.from_numpy(state)
                .float()
                .to(self.device)
                .unsqueeze(0),          # (1, state_dim)
        }
        
        # 手动应用归一化（因为normalize_inputs已被禁用）
        batch = self._normalize_inputs_manually(batch)
        
        # 关键修复：展平image的batch和time维度供rgb_encoder使用
        # rgb_encoder期望 (B, C, H, W)，但我们有 (B, T, C, H, W)
        # 所以展平为 (B*T, C, H, W)，rgb_encoder会处理它
        B, T, C, H, W = batch["observation.images.front"].shape
        batch["observation.images.front"] = batch["observation.images.front"].reshape(B * T, C, H, W)
        
        # 推理
        try:
            # 注意：normalize_inputs已被替换为Identity，所以select_action不会再做normalization
            # 我们已经在上面手动应用了normalization
            output = self.model.select_action(batch)
            actions = output[0].cpu().numpy()  # (horizon, action_dim)
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            raise
        
        # 如果需要原始值，直接返回
        if return_raw:
            return actions
        
        # 反归一化到实际范围
        actions = self._denormalize_actions(actions)
        
        return actions
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        反归一化动作到实际范围
        支持两种格式：
        1. MIN_MAX格式: min/max - 需要反归一化: x = (normalized + 1) / 2 * (max - min) + min
        2. 标准化格式: mean/std - 需要反归一化: x = normalized * std + mean
        
        如果actions维度与stats不匹配，只对前N维进行反归一化
        """
        if "action" not in self.stats:
            print("⚠ No action stats found, returning raw predictions")
            return actions
        
        stat = self.stats["action"]
        
        # 检查是MIN_MAX还是mean/std格式
        if "mean" in stat and "std" in stat:
            # 使用mean/std反归一化
            mean_val = np.array(stat["mean"], dtype=np.float32)
            std_val = np.array(stat["std"], dtype=np.float32)
            
            # 检查维度是否匹配
            stats_action_dim = len(mean_val)
            actual_action_dim = actions.shape[-1] if actions.ndim > 0 else 1
            
            if stats_action_dim != actual_action_dim:
                if self.verbose:
                    print(f"⚠ Action dim mismatch: stats has {stats_action_dim}, actual is {actual_action_dim}")
                # 只对前N维进行反归一化
                denormalized = actions.copy()
                denormalized[..., :stats_action_dim] = actions[..., :stats_action_dim] * std_val + mean_val
            else:
                denormalized = actions * std_val + mean_val
            
            return denormalized
        elif "min" in stat and "max" in stat:
            # 使用min/max反归一化
            min_val = np.array(stat["min"], dtype=np.float32)
            max_val = np.array(stat["max"], dtype=np.float32)
            
            # 检查维度是否匹配
            stats_action_dim = len(min_val)
            actual_action_dim = actions.shape[-1] if actions.ndim > 0 else 1
            
            if stats_action_dim != actual_action_dim:
                if self.verbose:
                    print(f"⚠ Action dim mismatch: stats has {stats_action_dim}, actual is {actual_action_dim}")
                # 反归一化公式（假设使用 MIN_MAX 归一化到 [-1, 1]）
                # normalized ∈ [-1, 1]
                # x = (normalized + 1) / 2 * (max - min) + min
                denormalized = actions.copy()
                denormalized[..., :stats_action_dim] = (actions[..., :stats_action_dim] + 1) / 2 * (max_val - min_val) + min_val
            else:
                denormalized = (actions + 1) / 2 * (max_val - min_val) + min_val
            
            return denormalized
        else:
            print("⚠ Missing mean/std or min/max in action stats, returning raw predictions")
            return actions
    
    def predict_batch(
        self,
        images: np.ndarray,     # (N, 3, 480, 640)
        states: np.ndarray,     # (N, state_dim)
        return_raw: bool = False
    ) -> np.ndarray:
        """
        批量推理
        
        Args:
            images: (N, 3, 480, 640) 图像批次
            states: (N, state_dim) 状态批次
            
        Returns:
            actions: (N, horizon, action_dim) 动作序列批次
        """
        actions_list = []
        for i in range(len(images)):
            actions = self.predict(images[i], states[i], return_raw=return_raw)
            actions_list.append(actions)
        return np.stack(actions_list)
    
    @property
    def config(self):
        """返回模型配置"""
        return self.model.config
    
    @property
    def state_dim(self) -> int:
        """返回状态维度"""
        state_shape = self.config.input_features["observation.state"].shape
        return state_shape[0]
    
    @property
    def action_dim(self) -> int:
        """返回动作维度"""
        action_shape = self.config.output_features["action"].shape
        return action_shape[0]
    
    @property
    def horizon(self) -> int:
        """返回预测时间步长"""
        return self.config.horizon


# 便捷函数：一次性加载多个任务的模型
def load_multi_task_models(
    checkpoints_dir: str = "checkpoints",
    tasks: list = None,
    device: str = "cuda"
) -> Dict[str, DiffusionPolicyInferenceEngine]:
    """
    加载所有任务的模型
    
    Args:
        checkpoints_dir: 检查点目录
        tasks: 任务列表，默认为 ["lift", "sort", "stack"]
        device: 使用的设备
        
    Returns:
        {task_name: inference_engine}
    """
    if tasks is None:
        tasks = ["lift", "sort", "stack"]
    
    models = {}
    checkpoints_dir = pathlib.Path(checkpoints_dir)
    
    for task in tasks:
        model_path = checkpoints_dir / f"{task}_real" / "checkpoint-best"
        if model_path.exists():
            try:
                models[task] = DiffusionPolicyInferenceEngine(str(model_path), device=device)
                print(f"✓ Loaded {task} model")
            except Exception as e:
                print(f"✗ Failed to load {task} model: {e}")
        else:
            print(f"✗ Model not found: {model_path}")
    
    return models


if __name__ == "__main__":
    # 测试推理引擎
    print("Testing DiffusionPolicy Inference Engine\n")
    
    # 加载 lift 模型
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    print(f"\nModel Info:")
    print(f"  State dim: {engine.state_dim}")
    print(f"  Action dim: {engine.action_dim}")
    print(f"  Horizon: {engine.horizon}")
    
    # 模拟推理
    print(f"\nTesting inference...")
    dummy_image = np.random.rand(3, 480, 640).astype(np.float32)
    dummy_state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    
    import time
    start = time.time()
    actions = engine.predict(dummy_image, dummy_state)
    elapsed = time.time() - start
    
    print(f"  Actions shape: {actions.shape}")
    print(f"  First action: {actions[0]}")
    print(f"  Inference time: {elapsed*1000:.2f} ms")
    
    if elapsed > 0.1:
        print(f"  ⚠ Warning: Inference slow ({elapsed*1000:.0f}ms > 100ms)")
    else:
        print(f"  ✓ Fast enough for real-time control")
    
    # 加载所有模型
    print(f"\n\nLoading all task models...")
    models = load_multi_task_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}")
