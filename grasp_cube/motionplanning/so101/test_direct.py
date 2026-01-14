import sys
sys.path.insert(0, "/home/ubuntu/grasp-cube-sample-main")
from grasp_cube.envs import *
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import gymnasium as gym
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

env = gym.make("SortCubeSO101-v1", obs_mode="none", control_mode="pd_joint_pos", render_mode="rgb_array")
env = env.unwrapped
env.reset(seed=0)

left_agent = env.agent.agents[0]

FINGER_LENGTH = 0.025

print("Step 1: Get OBB and approaching...")
obb_red = get_actor_obb(env.cube_red)
approaching = np.array([0, 0, -1])

print("Step 2: Get tcp_pose and target_closing...")
tcp_raw = left_agent.tcp_pose.raw_pose.cpu().numpy().squeeze()
tcp_pose_left_base = sapien.Pose(tcp_raw[:3], tcp_raw[3:])
tcp_pose_left = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * tcp_pose_left_base
target_closing_left = (tcp_pose_left).to_transformation_matrix()[:3, 1]
print(f"target_closing_left: {target_closing_left}")

print("Step 3: Compute grasp info...")
try:
    grasp_info_red = compute_grasp_info_by_obb(
        obb_red,
        approaching=approaching,
        target_closing=target_closing_left,
        depth=FINGER_LENGTH,
    )
    closing_red = grasp_info_red["closing"]
    print(f"closing_red: {closing_red}")
except Exception as e:
    print(f"ERROR in compute_grasp_info_by_obb: {e}")
    import traceback
    traceback.print_exc()

print("Step 4: Get cube position...")
cube_red_p = env.cube_red.pose.p
cube_red_pos = cube_red_p.cpu().numpy().squeeze()
print(f"cube_red_pos: {cube_red_pos}")

print("Step 5: Build grasp pose...")
try:
    grasp_pose_red = left_agent.build_grasp_pose(approaching, closing_red, cube_red_pos)
    print(f"grasp_pose_red type: {type(grasp_pose_red)}")
    print(f"grasp_pose_red p: {grasp_pose_red.p}")
except Exception as e:
    print(f"ERROR in build_grasp_pose: {e}")
    import traceback
    traceback.print_exc()

env.close()
