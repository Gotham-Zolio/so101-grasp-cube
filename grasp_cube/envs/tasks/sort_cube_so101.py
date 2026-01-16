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

@register_env("SortCubeSO101-v1", max_episode_steps=100)
class SortCubeSO101Env(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    
    cube_half_size = 0.015
    goal_thresh = 0.025
    cube_spawn_half_size = 0.08
    cube_spawn_center = (-0.715 + 0.15 + 0.018 + 0.082, 0)
    
    sensor_cam_eye_pos = [-0.27, 0, 0.4]
    sensor_cam_target_pos = [-0.56, 0, -0.25]
    lock_z = True
    
    # Task prompt for Diffusion Policy
    TASK_PROMPT = "Move the red cube to the left region and the green cube to the right region."
    
    def __init__(self, *args, robot_uids=("so101", "so101"), robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """Configure cameras for LeRobot Dataset format.
        - Front camera: 480×640, third-person view
        - Wrist cameras: 480×640, attached to each arm's camera_link
        """
        configs = []
        
        # Front camera: third-person view, 480×640 resolution
        front_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        configs.append(CameraConfig("front", front_pose, 640, 480, np.deg2rad(50), 0.01, 100))
        
        # Wrist cameras for dual-arm setup
        # Note: We'll set the mount in _load_agent after agents are loaded
        # For now, create configs without mount - they'll be updated in _load_agent
        configs.append(CameraConfig(
            uid="left_wrist",
            pose=sapien.Pose(),  # Identity pose - will be mounted to camera_link
            width=640,
            height=480,
            fov=np.deg2rad(50),
            near=0.01,
            far=100,
        ))
        configs.append(CameraConfig(
            uid="right_wrist",
            pose=sapien.Pose(),  # Identity pose - will be mounted to camera_link
            width=640,
            height=480,
            fov=np.deg2rad(50),
            near=0.01,
            far=100,
        ))
        
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
        
        # Mount wrist cameras to agent camera_links after agents are loaded
        # agents[0] is left arm, agents[1] is right arm
        if hasattr(self, 'agent') and hasattr(self.agent, 'agents'):
            if len(self.agent.agents) >= 2:
                # Left wrist camera (agent 0)
                left_agent = self.agent.agents[0]
                if hasattr(left_agent, 'robot') and 'camera_link' in left_agent.robot.links_map:
                    left_camera_link = left_agent.robot.links_map['camera_link']
                    if hasattr(self, '_sensors') and 'left_wrist' in self._sensors:
                        self._sensors['left_wrist'].mount = left_camera_link
                
                # Right wrist camera (agent 1)
                right_agent = self.agent.agents[1]
                if hasattr(right_agent, 'robot') and 'camera_link' in right_agent.robot.links_map:
                    right_camera_link = right_agent.robot.links_map['camera_link']
                    if hasattr(self, '_sensors') and 'right_wrist' in self._sensors:
                        self._sensors['right_wrist'].mount = right_camera_link

    

    
    def _build_boundary_lines(self):
        """
        Build black boundary boxes to create a 2 rows × 3 columns grid.
        Grid layout (same as PickCube):
        - 2 rows: bottom row (robots are here) and top row
        - 3 columns: left, middle, right
        - Robots at x = -0.715, y = ±0.15
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
        """Initialize dual-arm robot qpos for each episode."""
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        
        # Add noise for randomization (use _episode_rng for proper episode-based randomization)
        qpos_left = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        qpos_right = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        
        # Reset each agent individually
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
        
        self.cube_red = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube_red",
            initial_pose=sapien.Pose(p=[0, 0.05, self.cube_half_size])
        )
        self.cube_green = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cube_green",
            initial_pose=sapien.Pose(p=[0, -0.05, self.cube_half_size])
        )
        
        self.goal_red = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[1, 0, 0, 0.5], name="goal_red",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose()
        )
        self.goal_green = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 0.5], name="goal_green",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose()
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Initialize Agents using initialize_agent method
            self.initialize_agent(env_idx)
            
            # Grid parameters (matching _build_boundary_lines)
            box_inner = 0.16
            thickness = 0.02
            grid_start_x = -0.715
            
            # Column centers: left (+0.17), middle (0), right (-0.17)
            middle_col_y = 0.0
            left_col_y = box_inner + thickness/2  # +0.17
            right_col_y = -(box_inner + thickness/2)  # -0.17
            
            # Row centers: bottom (-0.715 + 0.08), top (-0.715 + 0.08 + 0.18)
            bottom_row_x = grid_start_x + box_inner/2  # -0.635
            top_row_x = bottom_row_x + box_inner + thickness  # -0.455
            
            # Spawn cubes in middle column, TOP ROW
            # Ensure minimum 5cm separation between cubes
            spawn_range = 0.03  # 4cm random range within the cell
            min_separation = 0.05  # 5cm minimum distance
            
            xyz_red = torch.zeros((b, 3))
            xyz_green = torch.zeros((b, 3))
            
            # Generate positions with distance check
            for i in range(b):
                max_attempts = 50
                for attempt in range(max_attempts):
                    # Red cube position
                    red_x = top_row_x + (torch.rand(1).item() * 2 - 1) * spawn_range
                    # red_x = bottom_row_x + torch.rand(1).item() * spawn_range
                    red_y = middle_col_y + (torch.rand(1).item() * 2 - 1) * spawn_range
                    
                    # Green cube position
                    green_x = top_row_x + (torch.rand(1).item() * 2 - 1) * spawn_range
                    # green_x = bottom_row_x + torch.rand(1).item() * spawn_range
                    green_y = middle_col_y + (torch.rand(1).item() * 2 - 1) * spawn_range
                    
                    # Check distance (using float math since these are scalars)
                    distance = ((red_x - green_x)**2 + (red_y - green_y)**2)**0.5
                    
                    if distance >= min_separation:
                        xyz_red[i, 0] = red_x
                        xyz_red[i, 1] = red_y
                        xyz_green[i, 0] = green_x
                        xyz_green[i, 1] = green_y
                        break
                else:
                    # Fallback: place with fixed offset if max attempts exceeded
                    xyz_red[i, 0] = top_row_x
                    xyz_red[i, 1] = middle_col_y + min_separation
                    xyz_green[i, 0] = top_row_x
                    xyz_green[i, 1] = middle_col_y - min_separation
            
            xyz_red[:, 2] = self.cube_half_size
            xyz_green[:, 2] = self.cube_half_size
            
            qs_red = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            qs_green = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            
            self.cube_red.set_pose(Pose.create_from_pq(xyz_red, qs_red))
            self.cube_green.set_pose(Pose.create_from_pq(xyz_green, qs_green))
            
            # Goals in side columns (top row), positioned closer to center for better reachability
            # Each box is 0.16m wide, left column range ~[0.09, 0.25], right column range ~[-0.25, -0.09]
            # Place goals at y=±0.10 (still inside boxes but closer to center)
            goal_offset_y = 0.17  # At inner edge of each box, closest to center
            
            # Red goal in right column
            goal_red_xyz = torch.zeros((b, 3))
            goal_red_xyz[:, 0] = top_row_x
            goal_red_xyz[:, 1] = -goal_offset_y  # -0.10 instead of -0.17
            goal_red_xyz[:, 2] = self.cube_half_size
            self.goal_red.set_pose(Pose.create_from_pq(goal_red_xyz))
            
            # Green goal in left column
            goal_green_xyz = torch.zeros((b, 3))
            goal_green_xyz[:, 0] = top_row_x
            goal_green_xyz[:, 1] = goal_offset_y  # +0.10 instead of +0.17
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
        # Simple normalized reward based on distance to goals
        return self.compute_dense_reward(obs, action, info)
