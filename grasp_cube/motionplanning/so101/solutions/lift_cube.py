import numpy as np
import sapien
from transforms3d.euler import euler2quat

from grasp_cube.envs.tasks.lift_cube_so101 import LiftCubeSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def _get_agent(env: LiftCubeSO101Env, agent_idx: int):
    agent = env.unwrapped.agent
    if hasattr(agent, "agents"):
        # Multi-agent: check if agent_idx is valid
        if agent_idx < len(agent.agents):
            return agent.agents[agent_idx]
        else:
            # If agent_idx is out of range, use the first (and only) agent
            return agent.agents[0]
    # Single agent: ignore agent_idx and return the agent directly
    return agent


def solve(env: LiftCubeSO101Env, seed=None, debug=False, vis=False, agent_idx: int = 0):
    env.reset(seed=seed)
    agent = _get_agent(env, agent_idx)
    planner = SO101ArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        agent_idx=agent_idx,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.open_gripper()
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])

    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * agent.tcp_pose.sp
    target_closing = (tcp_pose ).to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = grasp_info["closing"]
    grasp_pose = agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))


    # Reach
    reach_pose = sapien.Pose([0, 0.00, 0.05]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # Grasp
    planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose)
    planner.close_gripper(t=12)
    
    # Lift
    # Use world frame translation to ensure vertical lift, keeping the grasp orientation
    
    lift_height = 0.05
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]), grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(lift_pose)
    planner.close()
    return res
