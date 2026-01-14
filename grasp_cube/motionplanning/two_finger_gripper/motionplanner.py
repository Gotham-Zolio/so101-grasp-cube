import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from grasp_cube.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from transforms3d import quaternions


class TwoFingerGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        agent_idx: int = 0,
    ):
        super().__init__(env, debug, vis, base_pose, print_env_info, joint_vel_limits, joint_acc_limits, agent_idx=agent_idx)
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.env_agent.tcp_pose)

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)
    
    def _get_full_dual_arm_action(self, active_qpos, active_gripper_state, active_qvel=None):
        """
        Construct full dual-arm action for multi-agent environments.
        For single-arm environments, returns single-arm action.
        Returns action with batch dimension: (1, action_dim)
        """
        # Check if we're in a multi-agent environment
        if hasattr(self.env_agent_full, "agents") and len(self.env_agent_full.agents) > 1:
            # Get the other arm's current state
            other_agent_idx = 1 - self.agent_idx
            other_agent = self.env_agent_full.agents[other_agent_idx]
            other_robot = other_agent.robot
            
            # Get other arm's qpos
            other_qpos_full = other_robot.get_qpos()
            if hasattr(other_qpos_full, 'cpu'):
                other_qpos_full = other_qpos_full.cpu().numpy()
            if other_qpos_full.ndim > 1:
                other_qpos_full = other_qpos_full[0]
            
            # Get number of joints for other arm
            num_joints_other = len(other_robot.get_active_joints())
            other_qpos = other_qpos_full[:num_joints_other]
            
            # Get other arm's gripper state (track it or use default)
            if not hasattr(self, '_other_arm_gripper_state'):
                self._other_arm_gripper_state = self.OPEN
            other_gripper_state = self._other_arm_gripper_state
            
            # Construct actions for both arms
            if self.control_mode == "pd_joint_pos_vel":
                active_action = np.hstack([active_qpos, active_qvel if active_qvel is not None else np.zeros(len(active_qpos)), active_gripper_state])
                other_action = np.hstack([other_qpos, np.zeros(len(other_qpos)), other_gripper_state])
            else:
                active_action = np.hstack([active_qpos, active_gripper_state])
                other_action = np.hstack([other_qpos, other_gripper_state])
            
            # Combine: order depends on agent_idx (0=left/first, 1=right/second)
            if self.agent_idx == 0:
                full_action = np.hstack([active_action, other_action])
            else:
                full_action = np.hstack([other_action, active_action])
        else:
            # Single arm environment
            if self.control_mode == "pd_joint_pos_vel":
                full_action = np.hstack([active_qpos, active_qvel if active_qvel is not None else np.zeros(len(active_qpos)), active_gripper_state])
            else:
                full_action = np.hstack([active_qpos, active_gripper_state])
        
        # Ensure batch dimension: (1, action_dim)
        if full_action.ndim == 1:
            full_action = full_action[None, :]
        
        return full_action
    
    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            # Get qvel if available (for pd_joint_pos_vel mode)
            qvel = None
            if self.control_mode == "pd_joint_pos_vel" and "velocity" in result:
                qvel = result["velocity"][min(i, n_step - 1)]
            # Use helper method to construct full dual-arm action
            action = self._get_full_dual_arm_action(qpos, self.gripper_state, qvel)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        
        # Get current qpos
        qpos_full = self.robot.get_qpos()
        if hasattr(qpos_full, 'cpu'):
            qpos_full = qpos_full.cpu().numpy()
        if qpos_full.ndim > 1:
            qpos_full = qpos_full[0]
        
        # Get number of joints (excluding gripper if it's separate)
        num_joints = len(self.robot.get_active_joints())
        qpos = qpos_full[:num_joints]
        
        for i in range(t):
            # Use helper method to construct full dual-arm action
            action = self._get_full_dual_arm_action(qpos, gripper_state)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        
        # Get current qpos
        qpos_full = self.robot.get_qpos()
        if hasattr(qpos_full, 'cpu'):
            qpos_full = qpos_full.cpu().numpy()
        if qpos_full.ndim > 1:
            qpos_full = qpos_full[0]
        
        # Get number of joints (excluding gripper if it's separate)
        num_joints = len(self.robot.get_active_joints())
        qpos = qpos_full[:num_joints]
        
        for i in range(t):
            # Use helper method to construct full dual-arm action
            action = self._get_full_dual_arm_action(qpos, gripper_state)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


def build_two_finger_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual