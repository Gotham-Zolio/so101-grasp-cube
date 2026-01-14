from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from grasp_cube.agents.robots.so101.so_101 import SO101


@register_env("PickCubeSO101-v1", max_episode_steps=50)
class PickCubeSO101Env(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
    capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        "so101",
    ]
    agent: SO101
    cube_half_size = 0.015
    goal_thresh = 0.015 * 1.25
    cube_spawn_half_size = 0.08
    cube_spawn_center = (-0.715 + 0.15 + 0.018 + 0.082, 0)
    sensor_cam_eye_pos = [-0.27, 0, 0.4]
    sensor_cam_target_pos = [-0.56, 0, -0.25]
    human_cam_eye_pos = [-0.1, 0.3, 0.4]
    human_cam_target_pos = [-0.46, 0.0, 0.1]
    max_goal_height = 0.07
    lock_z = True

    def __init__(self, *args, robot_uids=("so101", "so101"), robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        """Configure cameras for LeRobot Dataset format.
        - Front camera: 480×640, third-person view (matches real-world front camera)
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
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Two robots: Left (y > 0) and Right (y < 0)
        super()._load_agent(options, [
            sapien.Pose(p=[-0.665, 0.15, 0]), 
            sapien.Pose(p=[-0.665, -0.15, 0])
        ])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._build_boundary_lines()
        # visualize cube spawn region with a very thin, non-colliding cube
        spawn_center = [self.cube_spawn_center[0], self.cube_spawn_center[1], self.cube_half_size]
        spawn_half_size = [self.cube_spawn_half_size, self.cube_spawn_half_size, 1e-4]
        self.cube_spawn_vis = actors.build_box(
            self.scene,
            half_sizes=spawn_half_size,
            color=[0, 0, 1, 0.2],
            name="cube_spawn_region",
            add_collision=False,
            initial_pose=sapien.Pose(p=spawn_center),
        )
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
    
    def _build_boundary_lines(self):
        """
        Build black boundary boxes to create a 2 rows × 3 columns grid in front of the robots.
        Robots are positioned at x = -0.715, y = ±0.15
        Grid layout:
        - 2 rows: bottom row (robots are here) and top row
        - 3 columns: left, middle, right
        - Left robot (y=0.15) is in bottom-left region
        - Right robot (y=-0.15) is in bottom-right region
        
        Grid starts at robot base (x = -0.715) and extends forward.
        """
        thickness = 0.02  # 2cm thickness
        box_inner = 0.16  # 16cm per cell
        height = 0.001  # 0.1cm height for visibility
        color = [0, 0, 0, 1]  # Black color
        
        # Grid dimensions
        # 3 columns total width (in y direction)
        total_width = box_inner * 3 + thickness * 4  # 0.16*3 + 0.02*4 = 0.56m
        # 2 rows total height (in x direction, forward from robot)
        total_height = box_inner * 2 + thickness * 3  # 0.16*2 + 0.02*3 = 0.38m
        
        # Grid position: starts at robot base x = -0.715
        grid_start_x = -0.715  # Robot base x position (bottom of grid)
        grid_end_x = grid_start_x + total_height  # Top of grid
        grid_center_x = grid_start_x + total_height / 2  # Center of grid in x direction
        grid_center_y = 0  # Center of grid in y direction (symmetric)
        
        # Row separator position (between bottom and top rows)
        row_separator_x = grid_start_x + box_inner + thickness  # Between the two rows
        
        z_pos = 0.005  # Slightly above table surface
        
        # After removing bottom boundary, shift remaining boundaries by -thickness in x direction
        # This maintains the grid structure while removing the bottom line
        x_shift = -thickness  # -0.02m shift
        
        # Horizontal lines (separate rows and form grid boundaries)
        # 1. Row separator line (between bottom and top rows)
        # Shifted by -thickness since bottom boundary is removed
        row_separator_x_shifted = row_separator_x + x_shift
        actors.build_box(
            self.scene,
            half_sizes=[thickness/2, total_width/2, height/2],
            color=color,
            name="boundary_row_separator",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([row_separator_x_shifted, grid_center_y, z_pos])
        )
        
        # 2. Top boundary line
        # Shifted by -thickness since bottom boundary is removed
        # Further shift down by thickness to align with top of vertical lines
        grid_end_x_shifted = grid_end_x + x_shift - thickness  # Additional -thickness to align with vertical lines
        actors.build_box(
            self.scene,
            half_sizes=[thickness/2, total_width/2, height/2],
            color=color,
            name="boundary_top",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([grid_end_x_shifted, grid_center_y, z_pos])
        )
        
        # Vertical lines (separate columns)
        # Column boundaries:
        # - Left column center: y = +box_inner + thickness/2 = +0.17
        # - Middle column center: y = 0
        # - Right column center: y = -(box_inner + thickness/2) = -0.17
        # Separators at:
        # - Left-middle: y = +box_inner/2 + thickness/2 = +0.09
        # - Middle-right: y = -(box_inner/2 + thickness/2) = -0.09
        
        # Updated grid center after x shift
        grid_center_x_shifted = grid_center_x + x_shift
        
        # Calculate actual vertical line height (from grid_start_x to grid_end_x_shifted)
        # After removing bottom boundary and shifting, the vertical lines should extend
        # from grid_start_x (robot base) to grid_end_x_shifted (top boundary)
        actual_vertical_height = grid_end_x_shifted - grid_start_x
        vertical_center_x = (grid_start_x + grid_end_x_shifted) / 2
        
        # Left-middle separator (full height)
        left_middle_y = box_inner/2 + thickness/2  # +0.09
        actors.build_box(
            self.scene,
            half_sizes=[actual_vertical_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([vertical_center_x, left_middle_y, z_pos])
        )
        
        # Middle-right separator (full height)
        middle_right_y = -(box_inner/2 + thickness/2)  # -0.09
        actors.build_box(
            self.scene,
            half_sizes=[actual_vertical_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_right",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([vertical_center_x, middle_right_y, z_pos])
        )
        
        # Short vertical lines to close the top-left and top-right regions
        # Top-left: from row separator to top boundary, at left edge (y = +total_width/2)
        top_left_y = total_width / 2  # +0.28 (left edge)
        # Top region height: from row separator to top boundary
        top_region_height = grid_end_x_shifted - row_separator_x_shifted
        top_region_center_x = (row_separator_x_shifted + grid_end_x_shifted) / 2
        actors.build_box(
            self.scene,
            half_sizes=[top_region_height/2, thickness/2, height/2],
            color=color,
            name="boundary_v_top_left",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose([top_region_center_x, top_left_y, z_pos])
        )
        
        # Top-right: from row separator to top boundary, at right edge (y = -total_width/2)
        top_right_y = -total_width / 2  # -0.28 (right edge)
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
        b = len(env_idx)
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        
        # Add noise for randomization
        qpos_left = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        qpos_right = qpos + self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
        
        # Reset each agent individually (MultiAgent expects per-agent reset)
        self.agent.agents[0].reset(qpos_left)
        self.agent.agents[1].reset(qpos_right)
        
        self.agent.agents[0].robot.set_pose(sapien.Pose([-0.665, 0.15, 0]))
        self.agent.agents[1].robot.set_pose(sapien.Pose([-0.665, -0.15, 0]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.initialize_agent(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5