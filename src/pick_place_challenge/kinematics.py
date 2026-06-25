"""Resolved-rate (Jacobian) inverse kinematics for the scripted expert.

The scripted pick-and-place expert plans in Cartesian space — a sequence of
end-effector waypoints relative to the ball and bowl — but the arm is commanded in
*joint* space. This module bridges the two: given the current end-effector pose
error, it returns the joint displacement that moves toward the target with one
damped-least-squares step of resolved-rate control,

    e   = [pos_err ; ori_err]          # current eef pose -> target pose
    dq  = Jᵀ (J Jᵀ + λ²I)⁻¹ e          # damped least squares; J is the 6×7 site Jacobian
    q  := q + clip(dq)                 # written to the position actuators

This is the same idea as mjlab's ``DifferentialIKAction`` (we read its
``mujoco_warp`` Jacobian the same way), kept small and inline so it is easy to read
and modify. Gravity/inertia are handled by the existing position servos, so we
never compute torques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


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
