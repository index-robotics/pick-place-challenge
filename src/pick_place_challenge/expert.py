"""A scripted pick-and-place expert that drives demonstrations.

This is the data source for behavior cloning. It is a small finite-state machine
over Cartesian waypoints, using *privileged* state read straight from the sim
(ball and bowl positions, current end-effector pose) — exactly what a teleoperator
or a motion planner would give you, but deterministic and reproducible.

Every step it picks a target eef pose for the current phase, then emits the **raw
action for the active control mode**:

    osc:   the 6-D pose delta toward the target (+ gripper).
    joint: the same Cartesian target solved to joint targets via one resolved-rate
           step (+ gripper) — i.e. ``q_target - home_q``.

So the two modes follow the *same* plan; only the action representation differs.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import compute_pose_error

from pick_place_challenge import scene
from pick_place_challenge.control import (
    OSC_ORI_SCALE,
    OSC_POS_SCALE,
    FrameJacobian,
    resolved_rate_dq,
)

# Waypoint plan: (reference object, x, y, z offset [m], gripper, pos tol [m], dwell).
# gripper: -1 = open, +1 = closed. dwell = min steps to hold once within tolerance
# (lets the gripper actually open/close). The arm tracks the grasp site to each
# target in turn; phases advance when reached (or after a failsafe step cap).
_PLAN = (
    # ref,   dx,  dy,    dz,  grip,  tol,  dwell
    ("ball", 0.0, 0.0, 0.120, -1.0, 0.030, 2),  # 0: hover above the ball
    ("ball", 0.0, 0.0, 0.005, -1.0, 0.020, 2),  # 1: descend onto the ball
    ("ball", 0.0, 0.0, 0.005, +1.0, 0.060, 12),  # 2: close the gripper (hold firm)
    ("bowl", 0.0, 0.0, 0.180, +1.0, 0.030, 2),  # 3: lift and carry above the bowl
    ("bowl", 0.0, 0.0, 0.060, +1.0, 0.030, 4),  # 4: lower into the bowl (centered)
    ("bowl", 0.0, 0.0, 0.060, -1.0, 0.040, 12),  # 5: release inside the bowl
    ("bowl", 0.0, 0.0, 0.200, -1.0, 0.100, 0),  # 6: retreat and hold
)
_MAX_STEPS_PER_PHASE = 90  # failsafe so a stuck env still advances/ends


class ScriptedExpert:
    """Vectorized waypoint pick-and-place expert for ``num_envs`` parallel envs."""

    def __init__(self, env, control: str = "joint"):
        self.env = env
        self.control = control
        self.robot = env.scene["robot"]
        self.site_local = self.robot.find_sites(scene.GRASP_SITE)[0][0]

        joint_ids, _ = self.robot.find_joints((r"joint[1-7]",))
        self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
        self.home_q = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        # The Jacobian (for joint demos) is only needed in joint mode.
        self.jac = (
            FrameJacobian(env, self.robot, scene.GRASP_SITE, self.joint_ids)
            if control == "joint"
            else None
        )

        # Plan tables as device tensors, indexed by per-env phase.
        self._use_bowl = torch.tensor(
            [r == "bowl" for r, *_ in _PLAN], device=env.device
        )
        self._offset = torch.tensor([p[1:4] for p in _PLAN], device=env.device)
        self._grip = torch.tensor([p[4] for p in _PLAN], device=env.device)
        self._tol = torch.tensor([p[5] for p in _PLAN], device=env.device)
        self._dwell = torch.tensor([p[6] for p in _PLAN], device=env.device)
        self._last_phase = len(_PLAN) - 1

        self.reset()

    def reset(self) -> None:
        n = self.env.num_envs
        self.phase = torch.zeros(n, dtype=torch.long, device=self.env.device)
        self.steps_in_phase = torch.zeros(n, dtype=torch.long, device=self.env.device)
        self.home_quat: torch.Tensor | None = None  # captured on first act()

    def act(self) -> torch.Tensor:
        """Return the raw action ``(num_envs, action_dim)`` for this step."""
        data = self.robot.data
        eef_pos = data.site_pos_w[:, self.site_local]
        eef_quat = data.site_quat_w[:, self.site_local]
        if self.home_quat is None:
            self.home_quat = eef_quat.clone()  # hold the (downward) home orientation

        ball = self.env.scene["ball"].data.root_link_pos_w
        bowl = self.env.scene["bowl"].data.root_link_pos_w
        ref = torch.where(self._use_bowl[self.phase].unsqueeze(-1), bowl, ball)
        target = ref + self._offset[self.phase]
        grip = self._grip[self.phase]

        pos_err, ori_err = compute_pose_error(eef_pos, eef_quat, target, self.home_quat)

        # Advance the FSM where the target is reached (after its dwell) or stuck.
        reached = pos_err.norm(dim=-1) < self._tol[self.phase]
        ready = reached & (self.steps_in_phase >= self._dwell[self.phase])
        stuck = self.steps_in_phase >= _MAX_STEPS_PER_PHASE
        advance = ready | stuck
        self.steps_in_phase = torch.where(
            advance, torch.zeros_like(self.steps_in_phase), self.steps_in_phase + 1
        )
        self.phase = torch.clamp(self.phase + advance.long(), max=self._last_phase)

        if self.control == "osc":
            raw_pos = (pos_err / OSC_POS_SCALE).clamp(-1.0, 1.0)
            raw_ori = (ori_err / OSC_ORI_SCALE).clamp(-1.0, 1.0)
            return torch.cat([raw_pos, raw_ori, grip.unsqueeze(-1)], dim=-1)

        # joint: take the *same* small Cartesian step OSC would (clamped to the per-
        # step scale), solve it to a joint target, and record it as q - home_q. The
        # bounded step keeps joint demos as gentle and reliable as the OSC ones.
        assert self.jac is not None
        jacp, jacr = self.jac.compute(eef_pos)
        step_pos = pos_err.clamp(-OSC_POS_SCALE, OSC_POS_SCALE)
        step_ori = ori_err.clamp(-OSC_ORI_SCALE, OSC_ORI_SCALE)
        dq = resolved_rate_dq(jacp, jacr, step_pos, step_ori)
        q_target = data.joint_pos[:, self.joint_ids] + dq
        return torch.cat([q_target - self.home_q, grip.unsqueeze(-1)], dim=-1)
