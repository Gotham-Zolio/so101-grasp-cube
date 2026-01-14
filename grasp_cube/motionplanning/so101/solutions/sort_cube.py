import numpy as np
import sapien
from transforms3d.euler import euler2quat

from grasp_cube.envs.tasks.sort_cube_so101 import SortCubeSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def _get_agent(env: SortCubeSO101Env, agent_idx: int):
    agent = env.unwrapped.agent
    if hasattr(agent, "agents"):
        return agent.agents[agent_idx]
    return agent


def solve(env: SortCubeSO101Env, seed=None, debug=False, vis=False, agent_idx: int = None):
    """
    Dual-arm solution for sort task:
    - Left arm (agent_idx=0): picks green cube and places at green goal
    - Right arm (agent_idx=1): picks red cube and places at red goal
    When agent_idx is None, both arms are used in coordination.
    """
    env.reset(seed=seed)
    
    # Create planners for both arms
    left_agent = _get_agent(env, 0)
    right_agent = _get_agent(env, 1)
    
    planner_left = SO101ArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=left_agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        agent_idx=0,
    )
    
    planner_right = SO101ArmMotionPlanningSolver(
        env,
        debug=False,
        vis=vis,
        base_pose=right_agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        agent_idx=1,
    )

    FINGER_LENGTH = 0.025
    
    # After creating planners, unwrap env for accessing scene objects
    env = env.unwrapped
    
    # Open both grippers first
    planner_left.open_gripper()
    planner_right.open_gripper()
    
    left_success = True
    right_success = True
    
    # ===========================================================================
    # Left arm: Green cube -> Green goal
    # ===========================================================================
    obb_green = get_actor_obb(env.cube_green)
    approaching = np.array([0, 0, -1])
    
    tcp_pose_left = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * left_agent.tcp_pose.sp
    target_closing_left = (tcp_pose_left).to_transformation_matrix()[:3, 1]
    
    grasp_info_green_left = compute_grasp_info_by_obb(
        obb_green,
        approaching=approaching,
        target_closing=target_closing_left,
        depth=FINGER_LENGTH,
    )
    closing_green_left = grasp_info_green_left["closing"]
    grasp_pose_green_left = left_agent.build_grasp_pose(approaching, closing_green_left, env.cube_green.pose.sp.p)
    grasp_pose_green_left = grasp_pose_green_left * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    # Left arm: Reach green cube
    reach_pose_green_left = sapien.Pose([0, 0.00, 0.05]) * grasp_pose_green_left
    res = planner_left.move_to_pose_with_RRTConnect(reach_pose_green_left)
    if res == -1:
        left_success = False
    
    # Left arm: Grasp green cube (only if reach succeeded)
    if left_success:
        res = planner_left.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose_green_left)
        if res == -1:
            left_success = False
    
    if left_success:
        planner_left.close_gripper(t=12)
        
        # Left arm: Lift green cube
        lift_pose_green_left = sapien.Pose(grasp_pose_green_left.p + np.array([0, 0, 0.05]), grasp_pose_green_left.q)
        res = planner_left.move_to_pose_with_RRTConnect(lift_pose_green_left)
        if res == -1:
            left_success = False
    
    if left_success:
        # Left arm: Move to green goal
        goal_green_pose_left = sapien.Pose(env.goal_green.pose.sp.p, grasp_pose_green_left.q)
        res = planner_left.move_to_pose_with_RRTConnect(goal_green_pose_left)
        if res == -1:
            left_success = False
    
    if left_success:
        # Open left gripper to release
        planner_left.open_gripper(t=12)
        
        # Left arm: Retract
        retract_pose_green_left = sapien.Pose(goal_green_pose_left.p + np.array([0, 0, 0.05]), goal_green_pose_left.q)
        planner_left.move_to_pose_with_RRTConnect(retract_pose_green_left)
    
    # ===========================================================================
    # Right arm: Red cube -> Red goal
    # ===========================================================================
    obb_red = get_actor_obb(env.cube_red)
    
    tcp_pose_right = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * right_agent.tcp_pose.sp
    target_closing_right = (tcp_pose_right).to_transformation_matrix()[:3, 1]
    
    grasp_info_red_right = compute_grasp_info_by_obb(
        obb_red,
        approaching=approaching,
        target_closing=target_closing_right,
        depth=FINGER_LENGTH,
    )
    closing_red_right = grasp_info_red_right["closing"]
    grasp_pose_red_right = right_agent.build_grasp_pose(approaching, closing_red_right, env.cube_red.pose.sp.p)
    grasp_pose_red_right = grasp_pose_red_right * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    # Right arm: Reach red cube
    reach_pose_red_right = sapien.Pose([0, 0.02, 0.03]) * grasp_pose_red_right
    res = planner_right.move_to_pose_with_RRTConnect(reach_pose_red_right)
    if res == -1:
        right_success = False
    
    # Right arm: Grasp red cube (only if reach succeeded)
    if right_success:
        res = planner_right.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose_red_right)
        if res == -1:
            right_success = False
    
    if right_success:
        planner_right.close_gripper(t=12)
        
        # Right arm: Lift red cube
        lift_pose_red_right = sapien.Pose(grasp_pose_red_right.p + np.array([0, 0, 0.05]), grasp_pose_red_right.q)
        res = planner_right.move_to_pose_with_RRTConnect(lift_pose_red_right)
        if res == -1:
            right_success = False
    
    if right_success:
        # Right arm: Move to red goal
        goal_red_pose_right = sapien.Pose(env.goal_red.pose.sp.p, grasp_pose_red_right.q)
        res = planner_right.move_to_pose_with_RRTConnect(goal_red_pose_right)
        if res == -1:
            right_success = False
    
    if right_success:
        # Open right gripper to release
        planner_right.open_gripper(t=12)
        
        # Right arm: Retract
        retract_pose_red_right = sapien.Pose(goal_red_pose_right.p + np.array([0, 0, 0.05]), goal_red_pose_right.q)
        planner_right.move_to_pose_with_RRTConnect(retract_pose_red_right)
    
    planner_left.close()
    planner_right.close()
    
    return res
