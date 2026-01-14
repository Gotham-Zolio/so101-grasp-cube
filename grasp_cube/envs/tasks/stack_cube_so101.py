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

@register_env("StackCubeSO101-v1", max_episode_steps=100)
class StackCubeSO101Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    
    cube_half_size = 0.015
    goal_thresh = 0.025
    cube_spawn_half_size = 0.08
    cube_spawn_center = (-0.715 + 0.15 + 0.018 + 0.082, 0)
    
    sensor_cam_eye_pos = [-0.27, 0, 0.4]
    sensor_cam_target_pos = [-0.56, 0, -0.25]
    lock_z = True
    
    # Task prompt for Diffusion Policy
    TASK_PROMPT = "Stack the red cube on top of the green cube."
    
    def __init__(self, *args, robot_uids=("so101",), robot_init_qpos_noise=0.02, **kwargs):
        """Initialize with single arm (right arm only)."""
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """Configure cameras for LeRobot Dataset format.
        - Front camera: 480×640, third-person view (matches real-world front camera)
        - Wrist cameras: Will be added via robot link attachment (requires robot URDF modification)
        """
        configs = []
        
        # Front camera: third-person view, 480×640 resolution
        # Using intrinsic parameters from PDF: fx=fy=570, approximate vfov ≈ 50° (0.873 rad)
        front_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        # CameraConfig: name, pose, width, height, fov (vertical), near, far
        configs.append(CameraConfig("front", front_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0, 0.4], target=[-0.46, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        """Load single arm (right arm only) at y = -0.15."""
        # Call BaseEnv._load_agent directly with initial_agent_poses
        super()._load_agent(options, [
            sapien.Pose(p=[-0.665, -0.15, 0])  # Right arm position
        ])

    def _build_boundary_lines(self):
        """
        Build black boundary boxes to create a 2 rows × 3 columns grid.
        Grid layout (same as PickCube):
        - 2 rows: bottom row (robots are here) and top row
        - 3 columns: left, middle, right
        - Grid starts at x = -0.715, robots at x = -0.665, y = ±0.15
        """
        thickness = 0.02  # 2cm thickness
        box_inner = 0.16  # 16cm per cell
        height = 0.001  # 0.1cm height
        color = [0, 0, 0, 1]  # Black
        
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
        
        # Row separator (horizontal)
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
        
        # Top region vertical edges
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
        """Initialize single arm (right arm only) qpos for each episode."""
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        
        # Add noise for randomization (use _episode_rng for proper episode-based randomization)
        qpos_right = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        
        # Reset single agent - check if it's MultiAgent or single agent
        if hasattr(self.agent, "agents"):
            # MultiAgent wrapper (shouldn't happen with single robot_uids, but handle it)
            self.agent.agents[0].reset(qpos_right)
            self.agent.agents[0].robot.set_pose(sapien.Pose([-0.665, -0.15, 0]))
        else:
            # Direct agent object (single robot)
            self.agent.reset(qpos_right)
            self.agent.robot.set_pose(sapien.Pose([-0.665, -0.15, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_boundary_lines()
        
        self.cube_red = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube_red",
            initial_pose=sapien.Pose(p=[0, 0.05, self.cube_half_size])
        )
        self.cube_green = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cube_green",
            initial_pose=sapien.Pose(p=[0, -0.05, self.cube_half_size])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        Initialize episode with red and green cubes in the top-right region.
        - Both cubes spawn in top row, right column
        - Minimum 5cm separation between cubes
        - Cubes should not be too close to edges
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Initialize Agents using initialize_agent method
            self.initialize_agent(env_idx)
            
            # Grid parameters (matching _build_boundary_lines)
            box_inner = 0.16
            thickness = 0.02
            grid_start_x = -0.715
            
            # Row centers: bottom (-0.715 + 0.08), top (-0.715 + 0.08 + 0.18)
            bottom_row_x = grid_start_x + box_inner/2  # -0.635
            top_row_x = bottom_row_x + box_inner + thickness  # -0.455
            
            # Right column center: y = -(box_inner + thickness/2) = -0.17
            right_col_y = -(box_inner + thickness/2)  # -0.17
            
            # Spawn cubes in top row, right column
            # Ensure minimum 5cm separation and avoid edges
            # Right column spans from y = -0.25 to y = -0.09 (approximately)
            # Use a smaller spawn range to avoid edges (keep at least 2cm from boundaries)
            spawn_range = 0.05  # 5cm random range within the cell (reduced from full cell size)
            min_separation = 0.05  # 5cm minimum distance
            edge_margin = 0.02  # 2cm margin from edges
            
            xyz_red = torch.zeros((b, 3))
            xyz_green = torch.zeros((b, 3))
            
            # Generate positions with distance check
            for i in range(b):
                max_attempts = 50
                for attempt in range(max_attempts):
                    # Red cube position in top-right region
                    # X: top row center ± spawn_range
                    red_x = top_row_x + (torch.rand(1).item() * 2 - 1) * spawn_range
                    # Y: right column center ± spawn_range (but avoid edges)
                    red_y = right_col_y + (torch.rand(1).item() * 2 - 1) * spawn_range
                    
                    # Green cube position in top-right region
                    green_x = top_row_x + (torch.rand(1).item() * 2 - 1) * spawn_range
                    green_y = right_col_y + (torch.rand(1).item() * 2 - 1) * spawn_range
                    
                    # Check distance (using float math since these are scalars)
                    distance = ((red_x - green_x)**2 + (red_y - green_y)**2)**0.5
                    
                    # Check if positions are not too close to edges
                    # Right column boundaries: approximately y = -0.25 (left edge) and y = -0.09 (right edge)
                    right_col_left_edge = right_col_y - box_inner/2  # approximately -0.25
                    right_col_right_edge = right_col_y + box_inner/2  # approximately -0.09
                    
                    red_too_close_to_edge = (
                        red_y < (right_col_left_edge + edge_margin) or 
                        red_y > (right_col_right_edge - edge_margin)
                    )
                    green_too_close_to_edge = (
                        green_y < (right_col_left_edge + edge_margin) or 
                        green_y > (right_col_right_edge - edge_margin)
                    )
                    
                    if distance >= min_separation and not red_too_close_to_edge and not green_too_close_to_edge:
                        xyz_red[i, 0] = red_x
                        xyz_red[i, 1] = red_y
                        xyz_green[i, 0] = green_x
                        xyz_green[i, 1] = green_y
                        break
                else:
                    # Fallback: place with fixed offset if max attempts exceeded
                    # Place green cube first, then red cube with min_separation offset
                    xyz_green[i, 0] = top_row_x
                    xyz_green[i, 1] = right_col_y - min_separation / 2
                    xyz_red[i, 0] = top_row_x
                    xyz_red[i, 1] = right_col_y + min_separation / 2
            
            xyz_red[:, 2] = self.cube_half_size
            xyz_green[:, 2] = self.cube_half_size
            
            qs_red = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            qs_green = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            
            self.cube_red.set_pose(Pose.create_from_pq(xyz_red, qs_red))
            self.cube_green.set_pose(Pose.create_from_pq(xyz_green, qs_green))

    def evaluate(self):
        """
        Success Criteria: The robot successfully stacks the red cube securely and centrally 
        on top of the green cube, maintaining stability.
        """
        # Check if red cube is on top of green cube
        # 1. Horizontal alignment: red cube should be close to green cube's center (within threshold)
        red_pos = self.cube_red.pose.p
        green_pos = self.cube_green.pose.p
        
        # Horizontal distance (xy plane)
        horizontal_dist = torch.linalg.norm(red_pos[:, :2] - green_pos[:, :2], axis=1)
        
        # 2. Vertical position: red cube should be on top of green cube
        # Red cube bottom should be at green cube top (green_z + cube_height)
        cube_height = self.cube_half_size * 2  # Full height of cube
        expected_red_z = green_pos[:, 2] + cube_height
        vertical_dist = torch.abs(red_pos[:, 2] - expected_red_z)
        
        # 3. Check if red cube is grasped (should not be grasped when stacked)
        if hasattr(self.agent, "agents"):
            agent = self.agent.agents[0]
        else:
            agent = self.agent
        is_grasped = agent.is_grasping(self.cube_red)
        
        # Success conditions:
        # - Horizontal distance < threshold (centered)
        # - Vertical distance < threshold (properly stacked)
        # - Not grasped (released)
        horizontal_thresh = 0.02  # 2cm tolerance for centering
        vertical_thresh = 0.01  # 1cm tolerance for stacking height
        
        is_stacked = (
            (horizontal_dist < horizontal_thresh) & 
            (vertical_dist < vertical_thresh) & 
            (~is_grasped)
        )
        
        return {
            "success": is_stacked,
            "horizontal_dist": horizontal_dist,
            "vertical_dist": vertical_dist,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(len(obs), device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # Simple normalized reward based on distance to goals
        return self.compute_dense_reward(obs, action, info)