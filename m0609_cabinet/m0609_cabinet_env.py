# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform


@configclass
class M0609CabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 7  # 6 arm joints + 1 binary gripper (open/close)
    observation_space = 24  # 주석은 그대로 두지만 실제 obs dim은 env에서 결정됨
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot - M0609 with xArm gripper
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"C:\Users\HBK\Desktop\usd\doosan-robot-isaac-driver-main\description\m0609_isaac_sim\m0609_xarm.usda",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint_1": 0.0,
                "joint_2": -0.8,  # More bent towards drawer
                "joint_3": 0.0,
                "joint_4": -2.0,  # More bent
                "joint_5": 0.0,
                "joint_6": 1.5,   # Point towards drawer
                "drive_joint": 0.0,  # Fully open
            },
            pos=(0.8, 0.0, 0.0),  # Closer to cabinet (was 1.0)
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    # 거리/자세는 보조, 서랍 여는 쪽을 메인 보상으로
    dist_reward_scale = 0.2
    rot_reward_scale = 0.3
    open_reward_scale = 30.0     # 절대 열림량 보상
    action_penalty_scale = 0.01
    finger_reward_scale = 0.5
    gripper_close_reward_scale = 1.0  # 지금은 직접 쓰진 않지만 인자 형태는 유지


class M0609CabinetEnv(DirectRLEnv):
    cfg: M0609CabinetEnvCfg

    def __init__(self, cfg: M0609CabinetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates."""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # controllable joint indices (arm + gripper drive joint)
        arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        gripper_joint_name = "drive_joint"

        self.arm_joint_indices = [self._robot.find_joints(name)[0][0] for name in arm_joint_names]
        self.gripper_joint_idx = self._robot.find_joints(gripper_joint_name)[0][0]

        # joint limits (전체 DOF 기준)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # speed scales (팔 관절만)
        self.robot_dof_speed_scales = torch.ones(len(self.arm_joint_indices), device=self.device)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # gripper open/close pos
        self.gripper_open_pos = 0.0
        self.gripper_close_pos = 48.7 * (3.14159 / 180.0)  # deg -> rad

        stage = get_current_stage()

        # xArm gripper structure from USDA
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm_gripper/xarm_gripper_base_link")
            ),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm_gripper/left_finger")
            ),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/xarm_gripper/right_finger")
            ),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        # body link indices
        self.hand_link_idx = self._robot.find_bodies("xarm_gripper_base_link")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("left_finger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("right_finger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # 서랍 열림 정도(prev)를 저장해서 progress 보상에 사용
        self.prev_drawer_open = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # arm actions (first 6)
        arm_actions = self.actions[:, :6]
        arm_targets = self.robot_dof_targets[:, self.arm_joint_indices].clone()
        arm_targets += self.robot_dof_speed_scales * self.dt * arm_actions * self.cfg.action_scale

        # clamp arm joints
        arm_targets = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[self.arm_joint_indices],
            self.robot_dof_upper_limits[self.arm_joint_indices],
        )
        self.robot_dof_targets[:, self.arm_joint_indices] = arm_targets

        # binary gripper: > 0 close, <= 0 open
        gripper_action = self.actions[:, 6]
        gripper_target = torch.where(
            gripper_action > 0.0,
            torch.full_like(gripper_action, self.gripper_close_pos),
            torch.full_like(gripper_action, self.gripper_open_pos),
        )
        self.robot_dof_targets[:, self.gripper_joint_idx] = gripper_target

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            self.robot_grasp_pos,
            self.drawer_grasp_pos,
            self.robot_grasp_rot,
            self.drawer_grasp_rot,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self.cfg.gripper_close_reward_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # init all joints with default
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()

        # randomize arm joints
        joint_pos[:, self.arm_joint_indices] += sample_uniform(
            -0.125, 0.125, (len(env_ids), len(self.arm_joint_indices)), self.device
        )

        # clamp arm joints
        joint_pos[:, self.arm_joint_indices] = torch.clamp(
            joint_pos[:, self.arm_joint_indices],
            self.robot_dof_lower_limits[self.arm_joint_indices],
            self.robot_dof_upper_limits[self.arm_joint_indices],
        )

        # gripper starts open
        joint_pos[:, self.gripper_joint_idx] = self.gripper_open_pos

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # drawer progress baseline 초기화
        self.prev_drawer_open[env_ids] = self._cabinet.data.joint_pos[env_ids, 3]

        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # arm joints only
        arm_dof_pos = self._robot.data.joint_pos[:, self.arm_joint_indices]
        arm_dof_vel = self._robot.data.joint_vel[:, self.arm_joint_indices]
        gripper_pos = self._robot.data.joint_pos[:, self.gripper_joint_idx].unsqueeze(-1)
        gripper_vel = self._robot.data.joint_vel[:, self.gripper_joint_idx].unsqueeze(-1)

        arm_dof_pos_scaled = (
            2.0
            * (arm_dof_pos - self.robot_dof_lower_limits[self.arm_joint_indices])
            / (self.robot_dof_upper_limits[self.arm_joint_indices] - self.robot_dof_lower_limits[self.arm_joint_indices])
            - 1.0
        )

        # gripper pos to [-1, 1]
        gripper_pos_scaled = (
            (gripper_pos - self.gripper_open_pos) / (self.gripper_close_pos - self.gripper_open_pos) * 2.0 - 1.0
        )

        to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                arm_dof_pos_scaled,                               # 6
                arm_dof_vel * self.cfg.dof_velocity_scale,       # 6
                to_target,                                        # 3
                gripper_pos_scaled,                               # 1
                gripper_vel * self.cfg.dof_velocity_scale,       # 1
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),  # 1
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),  # 1
                self.actions[:, :6],                              # 6
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]

        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.drawer_grasp_rot[env_ids],
            self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot[env_ids],
            self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        gripper_close_reward_scale,
        joint_positions,
    ):
        # 1) 거리 보상 (가이드 역할)
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)

        DIST_SIGMA = 0.10  # 10cm 정도
        dist_reward = torch.exp(-(d / DIST_SIGMA) ** 2)

        near_handle = d < 0.03
        dist_reward = torch.where(near_handle, dist_reward * 0.5, dist_reward)

        # 2) 회전 보상
        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # 3) 서랍 열림: 절대 열림 + progress
        drawer_open = cabinet_dof_pos[:, 3].clamp(min=0.0)
        open_reward = drawer_open

        drawer_progress = (drawer_open - self.prev_drawer_open).clamp(min=0.0)
        self.prev_drawer_open = drawer_open.detach()

        # 4) 손가락-손잡이 보조 신호
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        # 5) grasp 판단: 그리퍼가 닫힌 상태 + 손잡이 근처
        gripper_pos = joint_positions[:, self.gripper_joint_idx]
        GRIPPER_CLOSED_THRESH = 0.7 * self.gripper_close_pos
        gripper_closed = gripper_pos > GRIPPER_CLOSED_THRESH

        HANDLE_DIST_THRESH = 0.04
        near_handle_strict = d < HANDLE_DIST_THRESH

        grasped = gripper_closed & near_handle_strict
        grasp_reward = grasped.float()

        # 6) progress 보상: 잡고 당길 때 크게
        OPEN_PROGRESS_SCALE_WHEN_FREE = 10.0
        OPEN_PROGRESS_SCALE_WHEN_GRASP = 80.0

        drawer_progress_reward = drawer_progress * torch.where(
            grasped,
            torch.full_like(drawer_progress, OPEN_PROGRESS_SCALE_WHEN_GRASP),
            torch.full_like(drawer_progress, OPEN_PROGRESS_SCALE_WHEN_FREE),
        )

        # 7) success bonus: 충분히 열렸을 때
        SUCCESS_THRESH = 0.30
        success = drawer_open > SUCCESS_THRESH
        SUCCESS_BONUS = 10.0
        success_bonus = success.float() * SUCCESS_BONUS

        # 8) action penalty (arm only)
        action_penalty = torch.sum(actions[:, :6] ** 2, dim=-1)

        # 9) 최종 보상
        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + drawer_progress_reward
            + 5.0 * grasp_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
            + success_bonus
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "open_progress_reward": drawer_progress_reward.mean(),
            "grasp_reward": grasp_reward.mean(),
            "success_bonus": success_bonus.mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )
        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
