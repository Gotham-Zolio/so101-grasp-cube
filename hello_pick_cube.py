"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

import grasp_cube.agents.robots.so101.so_101
import grasp_cube.envs.tasks.pick_cube_so101
from dataclasses import dataclass
from typing import Annotated, Optional
import tyro
import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
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

def main(args: Args):
    env = gym.make(
        "PickCubeSO101-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()
    while True:
        if args.random_actions:
            env.step(env.action_space.sample())
        elif args.none_actions:
            env.step(None)
        elif args.zero_actions:
            env.step(env.action_space.sample() * 0)
        viewer = env.render()

if __name__ == "__main__":
    main(tyro.cli(Args))