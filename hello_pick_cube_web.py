"""
使用 Web Viewer 在无头模式下查看 PickCubeSO101 环境
在无头服务器或没有显示器的环境中运行时，可以通过浏览器访问 http://localhost:5000 查看仿真画面
"""

import grasp_cube.agents.robots.so101.so_101
import grasp_cube.envs.tasks.pick_cube_so101
from grasp_cube.utils.web_viewer import WebViewer
from dataclasses import dataclass
from typing import Annotated, Optional, Dict, Any
import tyro
import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
import time
import numpy as np

try:
    import torch
except ImportError:  # torch 一定存在，但防御一下
    torch = None

@dataclass
class Args:
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    shader: str = "default"
    random_actions: bool = False
    none_actions: bool = False
    zero_actions: bool = False
    sim_freq: int = 100
    control_freq: int = 20
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None
    port: int = 5000  # Web viewer 端口
    num_episodes: int = 10  # 运行的 episode 数量
    fps: int = 30  # 更新帧率

def main(args: Args):
    # 初始化 Web Viewer
    viewer = WebViewer(port=args.port)
    viewer.start()
    viewer.update_status(mode="Initializing", episode=0, total_episodes=args.num_episodes, task="PickCubeSO101")
    
    print(f"Web viewer started at http://localhost:{args.port}")
    print(f"Open this URL in your browser to view the simulation")
    
    # 创建环境，使用 rgb_array 模式而不是 human 模式
    env = gym.make(
        "PickCubeSO101-v1",
        obs_mode="rgbd",  # 需要 RGB 观测
        reward_mode="dense",
        enable_shadow=True,
        control_mode=args.control_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader, width=960, height=720),
        render_mode="rgb_array",  # 使用 rgb_array 而不是 human
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    
    env.reset(seed=args.seed if args.seed is not None else 0)
    env: BaseEnv = env.unwrapped
    
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    
    debug_counter = 0  # 仅前几帧打印调试信息

    def to_numpy_frame(img: Any):
        if torch is not None and isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            return None
        # 去掉 batch 维度 (1, H, W, C) 或 (1, C, H, W)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # 如果是通道优先 (C, H, W)，转为 (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # 若是浮点，认为 0-1，转为 uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def render_to_frames(render_output: Any, obs: Any) -> Dict[str, np.ndarray]:
        nonlocal debug_counter
        frames: Dict[str, np.ndarray] = {}

        def push_frame(name: str, img: Any):
            arr = to_numpy_frame(img)
            if arr is not None and arr.ndim >= 2:
                frames[name] = arr

        if isinstance(render_output, (np.ndarray, torch.Tensor)):
            push_frame("render", render_output)
        elif isinstance(render_output, dict):
            for k, v in render_output.items():
                push_frame(str(k), v)
        elif isinstance(render_output, (list, tuple)):
            for idx, v in enumerate(render_output):
                push_frame(f"render_{idx}", v)

        # 如果渲染为空，尝试从观测中取一张图
        if not frames and isinstance(obs, dict):
            for k, v in obs.items():
                push_frame(str(k), v)
                if frames:
                    break

        if debug_counter < 5:
            debug_counter += 1
            render_keys = render_output.keys() if isinstance(render_output, dict) else None
            print(f"[debug] render type={type(render_output)} keys={render_keys} frames={ {k: frames[k].shape for k in frames} }")
        return frames

    try:
        for episode in range(args.num_episodes):
            viewer.update_status(
                mode="Running", 
                episode=episode + 1, 
                total_episodes=args.num_episodes, 
                task="PickCubeSO101"
            )
            
            # 重置环境
            obs, info = env.reset()
            # 首帧推送，避免浏览器看到 No Signal
            initial_render = env.render()
            init_frames = render_to_frames(initial_render, obs)
            if init_frames:
                viewer.update_frames(init_frames)
            done = False
            step_count = 0
            
            print(f"\nEpisode {episode + 1}/{args.num_episodes} started")
            
            while not done:
                # 获取动作
                if args.random_actions:
                    action = env.action_space.sample()
                elif args.none_actions:
                    action = None
                elif args.zero_actions:
                    action = env.action_space.sample() * 0
                else:
                    # 默认使用随机动作
                    action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                
                # 渲染并更新到 web viewer
                rgb_frame = env.render()
                frames_to_push = render_to_frames(rgb_frame, obs)
                if frames_to_push:
                    viewer.update_frames(frames_to_push)
                
                # 控制帧率
                time.sleep(1.0 / args.fps)

                # 简单进度输出，避免长时间无输出
                if step_count % (args.control_freq // 2 or 1) == 0:
                    print(f"Episode {episode + 1} step {step_count}")
                
                if done:
                    success = info.get("success", False)
                    print(f"Episode {episode + 1} finished: steps={step_count}, success={success}, reward={reward:.2f}")
            
            # Episode 结束后稍作停留
            time.sleep(1.0)
        
        print(f"\nAll {args.num_episodes} episodes completed!")
        print("Web viewer will keep running. Press Ctrl+C to exit.")
        
        # 保持 web viewer 运行
        viewer.update_status(
            mode="Finished", 
            episode=args.num_episodes, 
            total_episodes=args.num_episodes, 
            task="PickCubeSO101"
        )
        
        # 保持程序运行，直到用户中断
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))
