"""End-effector pose control: a resolved-rate (Jacobian) Cartesian action term.

The default action space drives the arm in *joint* space (one target per joint).
This module adds the **OSC** mode: the policy instead emits a 6-D end-effector
pose *delta* (3 position + 3 axis-angle orientation), and we convert it to joint
targets with one damped-least-squares step of resolved-rate control each physics
substep:

    e   = [pos_err ; ori_err]          # current eef pose -> frozen target pose
    dq  = Jᵀ (J Jᵀ + λ²I)⁻¹ e          # damped least squares; J is the 6×7 site Jacobian
    q  := q + clip(dq)                 # written to the position actuators

This is the same idea as mjlab's ``DifferentialIKAction`` (we read its
``mujoco_warp`` Jacobian the same way), kept small and inline so it is easy to
read and modify. Gravity/inertia are handled by the existing position servos, so
we never compute torques. ``resolved_rate_dq`` and ``FrameJacobian`` are shared
with the scripted expert, which needs the same math to turn Cartesian waypoints
into joint demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

# Maps a raw action in [-1, 1] to a per-step pose delta. Shared by the OSC action
# term and the scripted expert so demos and the controller speak the same units.
OSC_POS_SCALE = 0.05  # meters
OSC_ORI_SCALE = 0.10  # radians


def resolved_rate_dq(
    jacp: torch.Tensor,
    jacr: torch.Tensor,
    pos_err: torch.Tensor,
    ori_err: torch.Tensor,
    damping: float = 0.05,
    max_dq: float = 0.5,
) -> torch.Tensor:
    """One damped-least-squares resolved-rate step: Cartesian error -> joint delta.

    Args:
        jacp, jacr: position / rotation site Jacobians, shape ``(N, 3, n)``.
        pos_err, ori_err: task-space error, each shape ``(N, 3)``.
        damping: DLS regularizer ``λ`` (keeps the solve stable near singularities).
        max_dq: per-step joint clamp (rad), so a large error can't kick the arm.

    Returns:
        Joint displacement ``dq`` of shape ``(N, n)``.
    """
    jac = torch.cat([jacp, jacr], dim=1)  # (N, 6, n)
    err = torch.cat([pos_err, ori_err], dim=1).unsqueeze(-1)  # (N, 6, 1)
    eye = torch.eye(6, device=jac.device).expand(jac.shape[0], 6, 6)
    # dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
    y = torch.linalg.solve(jac @ jac.transpose(1, 2) + damping**2 * eye, err)
    dq = (jac.transpose(1, 2) @ y).squeeze(-1)  # (N, n)
    return dq.clamp(-max_dq, max_dq)


class FrameJacobian:
    """Batched world-frame Jacobian of an entity *site*, via ``mujoco_warp``.

    Allocates the warp scratch buffers once and refills them on each ``compute``,
    mirroring how ``DifferentialIKAction`` calls ``mjwarp.jac``. Returns only the
    columns for the controlled joints.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        entity: Entity,
        site_name: str,
        joint_ids: torch.Tensor,
    ):
        self._env = env
        site_local = entity.find_sites(site_name)[0][0]
        self._site_id = int(entity.indexing.site_ids[site_local].item())
        self._body_id = int(env.sim.mj_model.site_bodyid[self._site_id])
        self._dof_ids = entity.indexing.joint_v_adr[joint_ids].long()

        nworld, nv = env.num_envs, env.sim.mj_model.nv
        with wp.ScopedDevice(env.sim.wp_device):
            self._jacp = wp.zeros((nworld, 3, nv), dtype=float)
            self._jacr = wp.zeros((nworld, 3, nv), dtype=float)
            self._point = wp.zeros(nworld, dtype=wp.vec3)
            self._body = wp.zeros(nworld, dtype=wp.int32)
            self._body.fill_(self._body_id)
        self._jacp_t = wp.to_torch(self._jacp)
        self._jacr_t = wp.to_torch(self._jacr)
        self._point_t = wp.to_torch(self._point).view(nworld, 3)

    def compute(self, point_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Jacobian columns for the controlled joints, evaluated at ``point_w``."""
        self._point_t[:] = point_w
        with wp.ScopedDevice(self._env.sim.wp_device):
            mjwarp.jac(
                self._env.sim.wp_model,
                self._env.sim.wp_data,
                self._jacp,
                self._jacr,
                self._point,
                self._body,
            )
        return self._jacp_t[:, :, self._dof_ids], self._jacr_t[:, :, self._dof_ids]


@dataclass(kw_only=True)
class OscPoseActionCfg(ActionTermCfg):
    """6-D end-effector pose-delta control producing joint position targets."""

    actuator_names: tuple[str, ...]
    """Actuator/joint name patterns selecting the controlled arm joints."""
    frame_name: str
    """Site whose pose is the end-effector reference (e.g. the grasp site)."""
    pos_scale: float = OSC_POS_SCALE
    ori_scale: float = OSC_ORI_SCALE
    damping: float = 0.05
    max_dq: float = 0.5

    def build(self, env: ManagerBasedRlEnv) -> OscPoseAction:
        return OscPoseAction(self, env)


class OscPoseAction(ActionTerm):
    """Policy emits a 6-D eef pose delta; we track it with resolved-rate IK."""

    cfg: OscPoseActionCfg

    def __init__(self, cfg: OscPoseActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        joint_ids, _ = self._entity.find_joints(cfg.actuator_names)
        self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
        self._site_local = self._entity.find_sites(cfg.frame_name)[0][0]
        self._jac = FrameJacobian(env, self._entity, cfg.frame_name, self._joint_ids)

        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat[:, 0] = 1.0

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def _eef_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._entity.data
        return data.site_pos_w[:, self._site_local], data.site_quat_w[
            :, self._site_local
        ]

    def process_actions(self, actions: torch.Tensor) -> None:
        # Freeze the target pose once per policy step: current eef pose ⊕ delta.
        self._raw[:] = actions
        delta = torch.empty_like(actions)
        delta[:, :3] = actions[:, :3] * self.cfg.pos_scale
        delta[:, 3:] = actions[:, 3:] * self.cfg.ori_scale
        pos, quat = self._eef_pose()
        self._target_pos[:], self._target_quat[:] = apply_delta_pose(pos, quat, delta)

    def apply_actions(self) -> None:
        # Re-track the frozen target every substep (closed loop within the step).
        pos, quat = self._eef_pose()
        pos_err, ori_err = compute_pose_error(
            pos, quat, self._target_pos, self._target_quat
        )
        jacp, jacr = self._jac.compute(pos)
        dq = resolved_rate_dq(
            jacp, jacr, pos_err, ori_err, self.cfg.damping, self.cfg.max_dq
        )
        q_target = self._entity.data.joint_pos[:, self._joint_ids] + dq
        self._entity.set_joint_position_target(q_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._target_pos[env_ids] = 0.0
        self._target_quat[env_ids] = 0.0
        self._target_quat[env_ids, 0] = 1.0
