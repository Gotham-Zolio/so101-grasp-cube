from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.envs.sapien_env import BaseEnv
from grasp_cube.agents.robots.so101.so_101 import SO101
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
import mani_skill.envs.utils.randomization as randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


@register_env("AvoidObstacleSO101-v1", max_episode_steps=150)
class AvoidObstacleSO101Env(BaseEnv):
    """
    Dual-Arm Hurdle Jump Task for SO101 Robot.
    
    Task Description:
    - Red cube (right column) with 3 obstacle blocks → Right arm lifts over obstacles → Middle column
    - Green cube (left column) with 3 obstacle blocks → Left arm lifts over obstacles → Middle column
    - Each arm must lift its cube high enough to clear the vertical obstacle column
    """
    SUPPORTED_ROBOTS = ["so101"]
    
    cube_half_size = 0.015
    obstacle_half_size = 0.015
    goal_thresh = 0.025
    
    box_inner = 0.16
    thickness = 0.02
    grid_start_x = -0.715
    
    sensor_cam_eye_pos = [-0.27, 0, 0.4]
    sensor_cam_target_pos = [-0.56, 0, -0.25]
    lock_z = True
    
    TASK_PROMPT = "Lift the red cube over obstacles to the middle. Lift the green cube over obstacles to the middle."
    
    def __init__(self, *args, robot_uids=("so101", "so101"), robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """Configure cameras for LeRobot Dataset format.
        - Front camera: 480×640, third-person view
        - Left side camera: 480×640, fixed pose on the left side
        - Right side camera: 480×640, fixed pose on the right side
        
        All tasks use the same three cameras: front + left_side + right_side.
        """
        configs = []
        
        # Front camera: third-person view, 480×640 resolution
        front_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        configs.append(CameraConfig("front", front_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        # Left side camera: fixed pose on the left side, covering top-left and top-middle regions
        left_side_eye = [-0.215, 0.33, 0.2]  # Left side position
        left_side_target = [-0.455, 0.115, 0.05]  # Looking at center between top-left (y=0.17) and top-middle (y=0.0)
        left_side_pose = sapien_utils.look_at(eye=left_side_eye, target=left_side_target)
        configs.append(CameraConfig("left_side", left_side_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        # Right side camera: fixed pose on the right side, covering top-right and top-middle regions
        right_side_eye = [-0.215, -0.33, 0.2]  # Right side position
        right_side_target = [-0.455, -0.115, 0.05]  # Looking at center between top-right (y=-0.17) and top-middle (y=0.0)
        right_side_pose = sapien_utils.look_at(eye=right_side_eye, target=right_side_target)
        configs.append(CameraConfig("right_side", right_side_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0, 0.4], target=[-0.46, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, [
            sapien.Pose(p=[-0.635, 0.15, 0]), 
            sapien.Pose(p=[-0.635, -0.15, 0])
        ])

    def _build_boundary_lines(self):
        """Build 2 rows × 3 columns grid (same as SortCube)."""
        thickness = 0.02
        box_inner = 0.16
        height = 0.001
        color = [0, 0, 0, 1]
        
        total_width = box_inner * 3 + thickness * 4
        total_height = box_inner * 2 + thickness * 3
        
        grid_start_x = -0.715
        grid_end_x = grid_start_x + total_height
        grid_center_x = grid_start_x + total_height / 2
        grid_center_y = 0
        
        row_separator_x = grid_start_x + box_inner + thickness
        z_pos = 0.005
        
        x_shift = -thickness
        row_separator_x_shifted = row_separator_x + x_shift
        grid_end_x_shifted = grid_end_x + x_shift - thickness
        
        # Row separator
        actors.build_box(
            self.scene,
            half_sizes=[thickness/2, total_width/2, height/2],
            color=color,
            name="boundary_row_separator",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([row_separator_x_shifted, grid_center_y, z_pos])
        )
        
        # Top boundary
        actors.build_box(
            self.scene,
            half_sizes=[thickness/2, total_width/2, height/2],
            color=color,
            name="boundary_top",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([grid_end_x_shifted, grid_center_y, z_pos])
        )
        
        actual_vertical_height = grid_end_x_shifted - grid_start_x
        vertical_center_x = (grid_start_x + grid_end_x_shifted) / 2
        
        # Left-middle separator
        left_middle_y = box_inner/2 + thickness/2
        actors.build_box(
            self.scene,
            half_sizes=[actual_vertical_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([vertical_center_x, left_middle_y, z_pos])
        )
        
        # Middle-right separator
        middle_right_y = -(box_inner/2 + thickness/2)
        actors.build_box(
            self.scene,
            half_sizes=[actual_vertical_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_right",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([vertical_center_x, middle_right_y, z_pos])
        )
        
        # Top region edges
        top_region_height = grid_end_x_shifted - row_separator_x_shifted
        top_region_center_x = (row_separator_x_shifted + grid_end_x_shifted) / 2
        
        top_left_y = total_width / 2
        actors.build_box(
            self.scene,
            half_sizes=[top_region_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_top_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([top_region_center_x, top_left_y, z_pos])
        )
        
        top_right_y = -total_width / 2
        actors.build_box(
            self.scene,
            half_sizes=[top_region_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_top_right",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([top_region_center_x, top_right_y, z_pos])
        )

    def initialize_agent(self, env_idx: torch.Tensor):
        """Initialize dual-arm robot qpos."""
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        
        qpos_left = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        qpos_right = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        
        self.agent.agents[0].reset(qpos_left)
        self.agent.agents[1].reset(qpos_right)
        
        self.agent.agents[0].robot.set_pose(sapien.Pose([-0.635, 0.15, 0]))
        self.agent.agents[1].robot.set_pose(sapien.Pose([-0.635, -0.15, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_boundary_lines()
        
        # Cubes
        self.cube_red = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube_red"
        )
        self.cube_green = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cube_green"
        )
        
        # Obstacle blocks (3 for left column, 3 for right column) - first row
        self.obstacles_left = []
        self.obstacles_right = []
        for i in range(3):
            obs_left = actors.build_cube(
                self.scene, half_size=self.obstacle_half_size, 
                color=[0.5, 0, 0.5, 1], name=f"obstacle_left_{i}"
            )
            obs_right = actors.build_cube(
                self.scene, half_size=self.obstacle_half_size,
                color=[0.5, 0, 0.5, 1], name=f"obstacle_right_{i}"
            )
            self.obstacles_left.append(obs_left)
            self.obstacles_right.append(obs_right)

        # Create second row of obstacles (for dual-layer obstacle layout)
        self.obstacles_left_2 = []
        self.obstacles_right_2 = []
        for i in range(3):
            obs_left_2 = actors.build_cube(
                self.scene, half_size=self.obstacle_half_size, 
                color=[0.5, 0, 0.5, 1], name=f"obstacle_left_2_{i}"
            )
            obs_right_2 = actors.build_cube(
                self.scene, half_size=self.obstacle_half_size,
                color=[0.5, 0, 0.5, 1], name=f"obstacle_right_2_{i}"
            )
            self.obstacles_left_2.append(obs_left_2)
            self.obstacles_right_2.append(obs_right_2)

        # Goals
        self.goal_red = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[1, 0, 0, 0.5], name="goal_red",
            body_type="kinematic", add_collision=False
        )
        self.goal_green = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 0.5], name="goal_green",
            body_type="kinematic", add_collision=False
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.initialize_agent(env_idx)
            
            # Grid coordinates
            box_inner = 0.16
            thickness = 0.02
            grid_start_x = -0.715
            
            left_col_y = box_inner + thickness/2  # +0.17
            middle_col_y = 0.0
            right_col_y = -(box_inner + thickness/2)  # -0.17
            
            top_row_x = grid_start_x + box_inner/2 + box_inner + thickness  # Top row
            
            # Place cubes in top row with X-axis randomization for domain randomization
            # Randomize cube positions along X-axis (±4cm range)
            random_x_range = 0.08  # ±4cm randomization range
            x_offsets_red = (torch.rand((b, 1), device=self.device) * random_x_range) - (random_x_range / 2)
            x_offsets_green = (torch.rand((b, 1), device=self.device) * random_x_range) - (random_x_range / 2)
            
            # Red cube (right column) with X-axis randomization
            xyz_red = torch.zeros((b, 3), device=self.device)
            xyz_red[:, 0] = top_row_x + x_offsets_red.squeeze()  # Apply random X offset
            xyz_red[:, 1] = right_col_y - 0.07  # Fixed Y position
            xyz_red[:, 2] = self.cube_half_size
            
            # Green cube (left column) with X-axis randomization
            xyz_green = torch.zeros((b, 3), device=self.device)
            xyz_green[:, 0] = top_row_x + x_offsets_green.squeeze()  # Apply random X offset
            xyz_green[:, 1] = left_col_y + 0.07  # Fixed Y position
            xyz_green[:, 2] = self.cube_half_size
            
            # --- START: 核心修正 ---
            # 不再生成随机旋转，而是创建一个固定的“无旋转”姿态
            # 单位四元数 [w, x, y, z] = [1, 0, 0, 0] 代表无旋转
            no_rotation_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            qs = no_rotation_q.unsqueeze(0).repeat(b, 1)

            # Apply poses to cubes
            self.cube_red.set_pose(Pose.create_from_pq(xyz_red, qs))
            self.cube_green.set_pose(Pose.create_from_pq(xyz_green, qs))
            
            # Place obstacle blocks
            obstacle_spacing = self.obstacle_half_size * 2 + 0.005
            obstacle_start_x = grid_start_x + box_inner/2 + 0.14
            
            for i in range(3):
                # Place first row of obstacles
                obs_xyz_left = torch.zeros((b, 3), device=self.device)
                obs_xyz_left[:, 0] = obstacle_start_x + i * obstacle_spacing
                obs_xyz_left[:, 1] = left_col_y
                obs_xyz_left[:, 2] = self.obstacle_half_size
                self.obstacles_left[i].set_pose(Pose.create_from_pq(obs_xyz_left))
                
                obs_xyz_right = torch.zeros((b, 3), device=self.device)
                obs_xyz_right[:, 0] = obstacle_start_x + i * obstacle_spacing
                obs_xyz_right[:, 1] = right_col_y
                obs_xyz_right[:, 2] = self.obstacle_half_size
                self.obstacles_right[i].set_pose(Pose.create_from_pq(obs_xyz_right))

                # Place second row of obstacles (offset inward from first row)
                y_offset = self.obstacle_half_size * 2  # One cube width

                # Second row left obstacles (offset inward)
                obs_xyz_left_2 = torch.zeros((b, 3), device=self.device)
                obs_xyz_left_2[:, 0] = obstacle_start_x + i * obstacle_spacing
                obs_xyz_left_2[:, 1] = left_col_y - y_offset - 0.04
                obs_xyz_left_2[:, 2] = self.obstacle_half_size
                self.obstacles_left_2[i].set_pose(Pose.create_from_pq(obs_xyz_left_2))

                # Second row right obstacles (offset inward)
                obs_xyz_right_2 = torch.zeros((b, 3), device=self.device)
                obs_xyz_right_2[:, 0] = obstacle_start_x + i * obstacle_spacing
                obs_xyz_right_2[:, 1] = right_col_y + y_offset + 0.04
                obs_xyz_right_2[:, 2] = self.obstacle_half_size
                self.obstacles_right_2[i].set_pose(Pose.create_from_pq(obs_xyz_right_2))

            # Goals in middle column
            goal_red_xyz = torch.zeros((b, 3), device=self.device)
            goal_red_xyz[:, 0] = top_row_x
            goal_red_xyz[:, 1] = middle_col_y - 0.03
            goal_red_xyz[:, 2] = self.cube_half_size
            self.goal_red.set_pose(Pose.create_from_pq(goal_red_xyz))
            
            goal_green_xyz = torch.zeros((b, 3), device=self.device)
            goal_green_xyz[:, 0] = top_row_x
            goal_green_xyz[:, 1] = middle_col_y + 0.03
            goal_green_xyz[:, 2] = self.cube_half_size
            self.goal_green.set_pose(Pose.create_from_pq(goal_green_xyz))

    def evaluate(self):
        red_to_goal = torch.linalg.norm(self.cube_red.pose.p - self.goal_red.pose.p, axis=1)
        green_to_goal = torch.linalg.norm(self.cube_green.pose.p - self.goal_green.pose.p, axis=1)
        
        success = (red_to_goal < self.goal_thresh) & (green_to_goal < self.goal_thresh)
        
        return {
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(len(obs), device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info)