import numpy as np
import sapien
from transforms3d.euler import euler2quat

from grasp_cube.envs.tasks.avoid_obstacle_so101 import AvoidObstacleSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def _get_agent(env: AvoidObstacleSO101Env, agent_idx: int):
    agent = env.unwrapped.agent
    if hasattr(agent, "agents"):
        return agent.agents[agent_idx]
    return agent


def solve(env: AvoidObstacleSO101Env, seed=None, debug=False, vis=False, agent_idx: int = None):
    """
    Dual-arm hurdle jump solution:
    - Both arms pick their cubes, jump over the first obstacle, land between obstacles,
      jump over the second obstacle, and place at their respective goals.
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
    env = env.unwrapped
    
    # Open both grippers
    planner_left.open_gripper()
    planner_right.open_gripper()
    
    left_success = True
    right_success = True
    
    # ===========================================================================
    # Left arm: Green cube -> Jump over obstacles -> Green goal
    # ===========================================================================

    # Dynamically calculate the midpoint for the left arm
    y_row1_left = env.obstacles_left[1].pose.sp.p[1]
    y_row2_left = env.obstacles_left_2[1].pose.sp.p[1]
    mid_point_y_left = (y_row1_left + y_row2_left) / 2
    mid_point_x_left = env.obstacles_left[1].pose.sp.p[0]

    obb_green = get_actor_obb(env.cube_green)
    approaching = np.array([0, 0, -1])
    
    tcp_pose_left = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * left_agent.tcp_pose.sp
    target_closing_left = (tcp_pose_left).to_transformation_matrix()[:3, 1]
    
    grasp_info_green = compute_grasp_info_by_obb(
        obb_green,
        approaching=approaching,
        target_closing=target_closing_left,
        depth=FINGER_LENGTH,
    )
    closing_green = grasp_info_green["closing"]
    grasp_pose_green = left_agent.build_grasp_pose(approaching, closing_green, env.cube_green.pose.sp.p)
    grasp_pose_green = grasp_pose_green * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    # Approach green cube
    reach_pose_green = sapien.Pose([0, 0.00, 0.05]) * grasp_pose_green
    res = planner_left.move_to_pose_with_RRTConnect(reach_pose_green)
    if res == -1: left_success = False
    
    # Grasp green cube
    if left_success:
        res = planner_left.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose_green)
        if res == -1: left_success = False
    
    if left_success:
        planner_left.close_gripper(t=12)
        lift_pose_green = sapien.Pose(grasp_pose_green.p + np.array([0, 0, 0.06]), grasp_pose_green.q)
        res = planner_left.move_to_pose_with_RRTConnect(lift_pose_green)
        if res == -1: left_success = False

    # Intermediate landing for the left arm
    if left_success:
        mid_point_p_left = np.array([mid_point_x_left, mid_point_y_left, grasp_pose_green.p[2]])
        mid_point_high_pose = sapien.Pose(mid_point_p_left + np.array([0, 0, 0.06]), grasp_pose_green.q)
        res = planner_left.move_to_pose_with_RRTConnect(mid_point_high_pose)
        if res == -1: left_success = False

    if left_success:
        mid_point_low_pose = sapien.Pose(mid_point_p_left + np.array([0, 0, -0.01]), grasp_pose_green.q)
        res = planner_left.move_to_pose_with_RRTConnect(mid_point_low_pose)
        if res == -1: left_success = False
            
    if left_success:
        mid_point_lift_pose = sapien.Pose(mid_point_low_pose.p + np.array([0, 0, 0.06]), mid_point_low_pose.q)
        res = planner_left.move_to_pose_with_RRTConnect(mid_point_lift_pose)
        if res == -1: left_success = False

    if left_success:
        goal_green_high = sapien.Pose(
            [env.goal_green.pose.sp.p[0], env.goal_green.pose.sp.p[1], grasp_pose_green.p[2] + 0.06],
            grasp_pose_green.q
        )
        res = planner_left.move_to_pose_with_RRTConnect(goal_green_high)
        if res == -1: left_success = False
    
    if left_success:
        goal_green_pose = sapien.Pose(env.goal_green.pose.sp.p, grasp_pose_green.q)
        res = planner_left.move_to_pose_with_RRTConnect(goal_green_pose)
        if res == -1: left_success = False
    
    if left_success:
        planner_left.open_gripper(t=12)
        retract_pose_green = sapien.Pose(goal_green_pose.p + np.array([0, 0, 0.05]), goal_green_pose.q)
        planner_left.move_to_pose_with_RRTConnect(retract_pose_green)
        safe_pose = sapien.Pose([-0.635 + 0.2, 0.15, 0.06], retract_pose_green.q)
        planner_left.move_to_pose_with_RRTConnect(safe_pose)
    
    # ===========================================================================
    # Right arm: Red cube -> Jump over obstacles -> Red goal
    # ===========================================================================

    # --- START: 核心修改 1 (右臂版本) ---
    # 动态计算中间降落点
    y_row1_right = env.obstacles_right[1].pose.sp.p[1]
    y_row2_right = env.obstacles_right_2[1].pose.sp.p[1]
    mid_point_y_right = (y_row1_right + y_row2_right) / 2
    mid_point_x_right = env.obstacles_right[1].pose.sp.p[0]
    # --- END: 核心修改 1 (右臂版本) ---

    obb_red = get_actor_obb(env.cube_red)
    
    tcp_pose_right = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * right_agent.tcp_pose.sp
    target_closing_right = (tcp_pose_right).to_transformation_matrix()[:3, 1]
    
    grasp_info_red = compute_grasp_info_by_obb(
        obb_red,
        approaching=approaching,
        target_closing=target_closing_right,
        depth=FINGER_LENGTH,
    )
    closing_red = grasp_info_red["closing"]
    grasp_pose_red = right_agent.build_grasp_pose(approaching, closing_red, env.cube_red.pose.sp.p)
    grasp_pose_red = grasp_pose_red * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    # Approach red cube
    reach_pose_red = sapien.Pose([0, 0.02, 0.03]) * grasp_pose_red
    res = planner_right.move_to_pose_with_RRTConnect(reach_pose_red)
    if res == -1: right_success = False
    
    # Grasp red cube
    if right_success:
        res = planner_right.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, -0.01]) * grasp_pose_red)
        if res == -1: right_success = False
    
    if right_success:
        planner_right.close_gripper(t=12)
        lift_pose_red = sapien.Pose(grasp_pose_red.p + np.array([0, 0, 0.06]), grasp_pose_red.q)
        res = planner_right.move_to_pose_with_RRTConnect(lift_pose_red)
        if res == -1: right_success = False
    
    # --- START: 核心修改 2 (右臂版本) ---
    # Intermediate landing for the right arm
    if right_success:
        mid_point_p_right = np.array([mid_point_x_right, mid_point_y_right, grasp_pose_red.p[2]])
        mid_point_high_pose = sapien.Pose(mid_point_p_right + np.array([0, 0, 0.06]), grasp_pose_red.q)
        res = planner_right.move_to_pose_with_RRTConnect(mid_point_high_pose)
        if res == -1: right_success = False

    if right_success:
        mid_point_low_pose = sapien.Pose(mid_point_p_right, grasp_pose_red.q)
        res = planner_right.move_to_pose_with_RRTConnect(mid_point_low_pose)
        if res == -1: right_success = False
            
    if right_success:
        mid_point_lift_pose = sapien.Pose(mid_point_low_pose.p + np.array([0, 0, 0.06]), mid_point_low_pose.q)
        res = planner_right.move_to_pose_with_RRTConnect(mid_point_lift_pose)
        if res == -1: right_success = False
    # --- END: 核心修改 2 (右臂版本) ---

    if right_success:
        goal_red_high = sapien.Pose(
            [env.goal_red.pose.sp.p[0], env.goal_red.pose.sp.p[1], grasp_pose_red.p[2] + 0.06],
            grasp_pose_red.q
        )
        res = planner_right.move_to_pose_with_RRTConnect(goal_red_high)
        if res == -1: right_success = False
    
    if right_success:
        goal_red_pose = sapien.Pose(env.goal_red.pose.sp.p, grasp_pose_red.q)
        res = planner_right.move_to_pose_with_RRTConnect(goal_red_pose)
        if res == -1: right_success = False
    
    if right_success:
        planner_right.open_gripper(t=12)
        retract_pose_red = sapien.Pose(goal_red_pose.p + np.array([0, 0, 0.05]), goal_red_pose.q)
        planner_right.move_to_pose_with_RRTConnect(retract_pose_red)
    
    planner_left.close()
    planner_right.close()
    
    return res