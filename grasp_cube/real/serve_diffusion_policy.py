#!/usr/bin/env python3
"""
DiffusionPolicy Server for Real Robot Deployment

此模块将 DiffusionPolicyInferenceEngine 包装为一个可以通过 WebSocket 服务的服务器，
参考 grasp_cube/real/serve_act_policy.py 的架构。

架构：
  Client (run_fake_env_client.py 或真机环境)
    ↓ WebSocket
  Server (此模块)
    ├─ WebsocketPolicyServer (从 env_client 库)
    └─ LeRobotDiffusionPolicy (推理引擎包装)

执行方式：
    python grasp_cube/real/serve_diffusion_policy.py \
        --policy.path checkpoints/lift_real/checkpoint-best \
        --policy.task lift \
        --host 0.0.0.0 \
        --port 8000 \
        --device cuda
"""

import dataclasses
import pathlib
import tyro
import numpy as np
from typing import Any, Literal, TYPE_CHECKING

# 导入环境客户端的服务器
ENV_CLIENT_AVAILABLE = False
WebsocketPolicyServer = None

try:
    from env_client.websocket_policy_server import WebsocketPolicyServer
    ENV_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: env_client not installed: {e}")

# 导入推理引擎和包装器
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from grasp_cube.real.diffusion_inference_wrapper import RealRobotDiffusionInferenceWrapper


@dataclasses.dataclass
class LeRobotDiffusionPolicyConfig:
    """DiffusionPolicy 服务器配置"""
    
    # 模型路径
    path: pathlib.Path
    
    # 任务名称
    task: Literal["lift", "sort", "stack"] = "lift"
    
    # 设备配置
    device: str = "cuda"
    
    # 推理参数
    act_steps: int | None = None  # 动作步数（如果为None则使用模型的horizon）


class LeRobotDiffusionPolicy:
    """
    DiffusionPolicy 推理政策包装器
    
    用于真机部署的推理政策。接收真机的观测数据，输出动作序列的下一个动作。
    
    接口与 LeRobotACTPolicy 兼容，便于与现有的 run_env_client.py 集成。
    """
    
    def __init__(self, config: LeRobotDiffusionPolicyConfig):
        """
        初始化推理政策
        
        Args:
            config: 配置对象，包含模型路径、任务名称、设备等
        """
        if not ENV_CLIENT_AVAILABLE:
            raise ImportError("env_client is required for server mode")
        
        self.config = config
        self.device = config.device
        self.task = config.task
        
        print(f"✓ Initializing LeRobotDiffusionPolicy")
        print(f"  Task: {config.task}")
        print(f"  Model path: {config.path}")
        print(f"  Device: {config.device}")
        
        # 初始化推理包装器
        try:
            self.inference_wrapper = RealRobotDiffusionInferenceWrapper(
                task_name=config.task,
                device=config.device,
                verbose=True
            )
            print(f"✓ Inference wrapper initialized")
        except Exception as e:
            print(f"✗ Failed to initialize inference wrapper: {e}")
            raise
        
        # 动作缓存：保存当前的动作序列和索引
        self.action_chunk = None
        self.chunk_index = 0
        
        # 记录调用次数用于调试
        self.inference_count = 0
        self.action_count = 0
    
    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        从观测获取动作
        
        这是 BasePolicy 接口所需的方法。返回一个动作字典，其中包含 action 序列。
        
        与 env_client 的 BasePolicy 兼容。
        
        Args:
            observation: 观测字典，包含图像和状态
        
        Returns:
            dict 包含 "action" 键，值为 (horizon, action_dim) 的动作序列
        """
        try:
            self.inference_count += 1
            
            print(f"\n[Policy Server] Inference #{self.inference_count}")
            
            # 从推理包装器获取动作序列
            action_sequence = self.inference_wrapper.predict_from_obs(observation)
            
            # 验证输出
            if action_sequence is None or action_sequence.shape[0] == 0:
                raise RuntimeError("Policy returned empty action sequence")
            
            print(f"  → Action type: {type(action_sequence)}")
            print(f"  → Action dtype: {action_sequence.dtype}")
            print(f"  → Action shape: {action_sequence.shape}")
            print(f"  → Action ndim: {action_sequence.ndim}")
            print(f"  → Action range: [{action_sequence.min():.4f}, {action_sequence.max():.4f}]")
            
            # 返回 BasePolicy 接口期望的格式
            # ⚠️ CRITICAL: Convert numpy array to list for JSON serialization
            # The WebSocket will serialize this to JSON, so we must convert to Python list
            # Return just the list, not wrapped in dict - the client will handle wrapping
            action_list = action_sequence.tolist()
            print(f"  → Returning action_list with {len(action_list)} actions")
            return action_list
            
        except Exception as e:
            print(f"✗ [Policy Server] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回零动作作为安全的 fallback
            fallback_action = np.zeros(
                (self.inference_wrapper.engine.horizon, self.inference_wrapper.engine.action_dim),
                dtype=np.float32
            )
            return fallback_action.tolist()
    
    def reset(self) -> None:
        """
        重置政策状态
        
        在每个 episode 开始时调用（如果有状态需要重置）
        """
        print("[Policy Server] Reset called")
        self.action_chunk = None
        self.chunk_index = 0
        self.inference_wrapper.reset_chunk()


@dataclasses.dataclass
class DiffusionPolicyServerConfig:
    """DiffusionPolicy 服务器的主配置"""
    
    # 政策配置
    policy: LeRobotDiffusionPolicyConfig
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None


def create_diffusion_policy_server(
    config: DiffusionPolicyServerConfig
) -> Any:
    """
    创建 DiffusionPolicy WebSocket 服务器
    
    Args:
        config: 服务器配置
    
    Returns:
        server: 可以直接调用 serve_forever() 的服务器对象
    
    Raises:
        RuntimeError: 如果 env_client 未安装
    """
    
    if not ENV_CLIENT_AVAILABLE:
        raise RuntimeError(
            "env_client is not installed. Please install it with:\n"
            "  uv pip install -e packages/env-client"
        )
    
    print("=" * 70)
    print("Creating DiffusionPolicy WebSocket Server")
    print("=" * 70)
    
    # 创建推理政策
    policy = LeRobotDiffusionPolicy(config.policy)
    
    # 创建 WebSocket 服务器
    # 这个 API 与 serve_act_policy.py 中的相同
    server = WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        metadata={
            "policy_type": "DiffusionPolicy",
            "task": config.policy.task,
            "model_path": str(config.policy.path),
            "device": config.policy.device,
        },
    )
    
    print(f"\n✓ Server created successfully")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Task: {config.policy.task}")
    print(f"  Device: {config.policy.device}")
    print(f"\nServer will listen at ws://{config.host}:{config.port}")
    
    return server


def main():
    """主函数：解析命令行参数并启动服务器"""
    
    config = tyro.cli(DiffusionPolicyServerConfig)
    server = create_diffusion_policy_server(config)
    
    print("\n" + "=" * 70)
    print("Starting DiffusionPolicy Policy Server...")
    print("=" * 70)
    print("\nWaiting for client connections at ws://0.0.0.0:8000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped gracefully")
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
