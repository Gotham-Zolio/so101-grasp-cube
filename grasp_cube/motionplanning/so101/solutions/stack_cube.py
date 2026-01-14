import numpy as np
import sapien
from transforms3d.euler import euler2quat

from grasp_cube.envs.tasks.stack_cube_so101 import StackCubeSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def _get_agent(env: StackCubeSO101Env, agent_idx: int):
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


def solve(env: StackCubeSO101Env, seed=None, debug=False, vis=False, agent_idx: int = 0):
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
    
    # Get red cube (the one to be stacked)
    obb = get_actor_obb(env.cube_red)

    approaching = np.array([0, 0, -1])

    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * agent.tcp_pose.sp
    target_closing = (tcp_pose).to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = grasp_info["closing"]
    grasp_pose = agent.build_grasp_pose(approaching, closing, env.cube_red.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))

    # Reach red cube
    reach_pose = sapien.Pose([0, 0.02, 0.03]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # Grasp red cube
    planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose)
    planner.close_gripper(t=12)


    planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0.07]) * grasp_pose)

    
    # Move to stack position (above green cube)
    # Get green cube position
    green_pos = env.cube_green.pose.sp.p
    cube_height = env.cube_half_size * 2  # Full height of cube
    stack_height = 0.13  # Lift height above green cube
    
    # Target position: above green cube center, at height = green_z + cube_height + stack_height
    stack_target_pos = green_pos + np.array([-0.01, 0, cube_height + stack_height])
    
    # Keep the same orientation as grasp pose for stacking
    stack_pose = sapien.Pose(stack_target_pos, grasp_pose.q)
    
    # Move to stack position
    res = planner.move_to_pose_with_RRTConnect(stack_pose)
    
    # Lower to stack position (just above green cube)
    final_stack_pos = green_pos + np.array([-0.01, 0, cube_height])
    final_stack_pose = sapien.Pose(final_stack_pos, grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(final_stack_pose)
    
    # Release the cube
    planner.open_gripper(t=12)
    
    # Lift up slightly to avoid collision
    lift_away_pose = sapien.Pose(final_stack_pos + np.array([0, 0, 0.02]), grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(lift_away_pose)
    
    planner.close()
    return res