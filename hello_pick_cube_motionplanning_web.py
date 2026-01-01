"""
使用 Web Viewer 在无头模式下查看 PickCubeSO101 运动规划
在无头服务器或没有显示器的环境中运行时，可以通过浏览器访问 http://localhost:5000 查看仿真画面
"""

import grasp_cube.agents.robots.so101.so_101
import grasp_cube.envs.tasks.pick_cube_so101
from grasp_cube.utils.web_viewer import WebViewer
from grasp_cube.motionplanning.so101.solutions import solvePickCube
from dataclasses import dataclass
from typing import Annotated, Optional, Dict, Any
import tyro
import gymnasium as gym
import mani_skill
from mani_skill.envs.sapien_env import BaseEnv
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

@dataclass
class Args:
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    shader: str = "default"
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None
    port: int = 5000  # Web viewer 端口
    num_episodes: int = 10  # 运行的 episode 数量
    fps: int = 30  # 更新帧率
    vis: bool = False  # 是否显示运动规划的可视化辅助信息

def main(args: Args):
    # 初始化 Web Viewer
    viewer = WebViewer(port=args.port)
    viewer.start()
    viewer.update_status(mode="Initializing", episode=0, total_episodes=args.num_episodes, task="PickCubeSO101-MotionPlanning")
    
    print(f"Web viewer started at http://localhost:{args.port}")
    print(f"Open this URL in your browser to view the simulation")
    
    # 创建环境，使用 rgb_array 模式
    env = gym.make(
        "PickCubeSO101-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader, width=960, height=720),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )
    
    env: BaseEnv = env.unwrapped
    
    print("Running motion planning solution for PickCubeSO101")
    
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
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        # 若是浮点，认为 0-1，转为 uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def render_to_frames(render_output: Any) -> Dict[str, np.ndarray]:
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

        return frames

    try:
        seed = args.seed if args.seed is not None else 0
        successes = []
        
        for episode in range(args.num_episodes):
            viewer.update_status(
                mode="Running", 
                episode=episode + 1, 
                total_episodes=args.num_episodes, 
                task="PickCubeSO101-MotionPlanning"
            )
            
            print(f"\nEpisode {episode + 1}/{args.num_episodes} started (seed={seed})")
            
            try:
                # 运行运动规划解决方案
                result = solvePickCube(env, seed=seed, debug=False, vis=args.vis)
                
                if result == -1:
                    print(f"Episode {episode + 1} failed: Motion planning could not find a solution")
                    successes.append(False)
                else:
                    # result 是一个包含每一步信息的列表
                    success = result[-1]["success"].item() if hasattr(result[-1]["success"], "item") else result[-1]["success"]
                    elapsed_steps = result[-1]["elapsed_steps"].item() if hasattr(result[-1]["elapsed_steps"], "item") else result[-1]["elapsed_steps"]
                    
                    print(f"Episode {episode + 1} completed: success={success}, steps={elapsed_steps}")
                    successes.append(success)
                    
                    # 重放轨迹并推送到 web viewer
                    print(f"Replaying trajectory for visualization...")
                    env.reset(seed=seed)
                    
                    for step_idx, step_data in enumerate(result):
                        # 执行动作
                        if "action" in step_data:
                            env.step(step_data["action"])
                        
                        # 渲染并推送
                        rgb_frame = env.render()
                        frames = render_to_frames(rgb_frame)
                        if frames:
                            viewer.update_frames(frames)
                        
                        # 控制帧率
                        time.sleep(1.0 / args.fps)
                        
                        # 进度输出
                        if step_idx % 10 == 0:
                            print(f"  Replaying step {step_idx}/{len(result)}")
                    
            except Exception as e:
                print(f"Episode {episode + 1} failed with error: {e}")
                successes.append(False)
            
            # Episode 结束后稍作停留
            time.sleep(1.0)
            seed += 1
        
        # 统计结果
        success_rate = sum(successes) / len(successes) if successes else 0.0
        print(f"\n{'='*60}")
        print(f"All {args.num_episodes} episodes completed!")
        print(f"Success rate: {success_rate:.2%} ({sum(successes)}/{len(successes)})")
        print(f"{'='*60}")
        print("Web viewer will keep running. Press Ctrl+C to exit.")
        
        # 保持 web viewer 运行
        viewer.update_status(
            mode="Finished", 
            episode=args.num_episodes, 
            total_episodes=args.num_episodes, 
            task=f"PickCubeSO101-MotionPlanning (Success: {success_rate:.0%})"
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
