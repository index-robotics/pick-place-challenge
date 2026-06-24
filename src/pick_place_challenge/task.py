"""The "place the ball in the bowl" task: MDP, env config, and registration.

This is the one place the environment and its (single) MDP are wired together.
Importing this module registers two variants that differ only in observation:

    Mjlab-PlaceBall-Franka-State-v0   — low-dim state (joint state, EE→ball,
                                        ball→bowl, bowl position).
    Mjlab-PlaceBall-Franka-Pixels-v0  — wrist + scene RGB; the ball is not given
                                        as state (you must see it).

Built on mjlab's ``lift_cube`` manipulation skeleton (actions / sim / viewer),
with the lift command + rewards replaced by the reach-and-place MDP below. The
reward is intentionally simple — rip it out and write your own.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manip_mdp
from mjlab.tasks.manipulation.config.yam.rl_cfg import (
    yam_lift_cube_ppo_runner_cfg,
    yam_lift_cube_vision_ppo_runner_cfg,
)
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pick_place_challenge import scene
from pick_place_challenge.control import OscPoseActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Arm joints (the 7 Panda joints) — selected for both control modes.
_ARM_JOINTS = (r"joint[1-7]",)

_ROBOT = SceneEntityCfg("robot")


# ============================================================================
# MDP — the single task MDP (goal = the bowl's position; no command term)
# ============================================================================


def _target_pos_w(env: ManagerBasedRlEnv, target_name: str, z_offset: float):
    """World position of the bowl opening (its origin lifted by ``z_offset``)."""
    pos = env.scene[target_name].data.root_link_pos_w.clone()
    pos[:, 2] += z_offset
    return pos


def object_to_target_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Vector from the ball to the bowl opening, in the robot base frame."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    vec_w = _target_pos_w(env, target_name, z_offset) - obj.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def target_position(
    env: ManagerBasedRlEnv,
    target_name: str,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Bowl opening position in the robot base frame."""
    robot: Entity = env.scene[asset_cfg.name]
    rel = _target_pos_w(env, target_name, z_offset) - robot.data.root_link_pos_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), rel)


def reach_and_place_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    reaching_std: float = 0.1,
    placing_std: float = 0.1,
    z_offset: float = 0.04,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """reaching(ee→ball) * (1 + placing(ball→bowl)) — gates placing on reaching."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    obj_pos = obj.data.root_link_pos_w
    reaching = torch.exp(-torch.sum((ee - obj_pos) ** 2, dim=-1) / reaching_std**2)
    place_err = torch.sum(
        (_target_pos_w(env, target_name, z_offset) - obj_pos) ** 2, dim=-1
    )
    placing = torch.exp(-place_err / placing_std**2)
    return reaching * (1.0 + placing)


def placed_in_bowl(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    radius: float = 0.06,
    max_height: float = 0.07,
) -> torch.Tensor:
    """True when the ball is within the bowl's rim radius and below its rim."""
    obj_pos = env.scene[object_name].data.root_link_pos_w
    bowl_pos = env.scene[target_name].data.root_link_pos_w
    horizontal = torch.linalg.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=-1)
    return (horizontal < radius) & (obj_pos[:, 2] < bowl_pos[:, 2] + max_height)


# ============================================================================
# Env config (state + pixels)
# ============================================================================

_GRIPPER_BODY = "base"  # 2F-85 base body (parent for the wrist camera)
_BOWL_POS = (0.40, 0.16, 0.0)  # fixed bowl location on the table
_BALL_POS = (0.30, -0.10, 0.06)  # nominal ball spawn (randomized at reset)


def _grasp_site() -> SceneEntityCfg:
    return SceneEntityCfg("robot", site_names=(scene.GRASP_SITE,))


def state_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """State-based place-ball-in-bowl."""
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {
        "robot": scene.franka_robotiq_cfg(),
        "ball": scene.EntityCfg(
            spec_fn=scene.ball_spec,
            init_state=scene.EntityCfg.InitialStateCfg(pos=_BALL_POS),
        ),
        "bowl": scene.EntityCfg(
            spec_fn=scene.bowl_spec,
            init_state=scene.EntityCfg.InitialStateCfg(pos=_BOWL_POS),
        ),
    }
    cfg.scene.terrain = None
    cfg.scene.spec_fn = scene.add_studio
    cfg.scene.sensors = ()  # drop the base task's YAM-specific contact sensor

    # Actions: 7 arm joints (position) + gripper tendon (0=open..255=closed).
    arm = cfg.actions["joint_pos"]
    assert isinstance(arm, JointPositionActionCfg)
    arm.actuator_names = (".*",)
    arm.scale = scene.ARM_ACTION_SCALE
    cfg.actions["gripper"] = TendonLengthActionCfg(
        entity_name="robot",
        actuator_names=(scene.GRIPPER_TENDON,),
        scale=scene.GRIPPER_CTRL_MAX / 2.0,
        offset=scene.GRIPPER_CTRL_MAX / 2.0,
    )

    actor = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "ee_to_ball": ObservationTermCfg(
            func=manip_mdp.ee_to_object_distance,
            params={"object_name": "ball", "asset_cfg": _grasp_site()},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ball_to_bowl": ObservationTermCfg(
            func=object_to_target_distance,
            params={"object_name": "ball", "target_name": "bowl"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "bowl_position": ObservationTermCfg(
            func=target_position, params={"target_name": "bowl"}
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(actor, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor), enable_corruption=False),
    }

    cfg.commands = {}
    cfg.rewards = {
        "reach_place": RewardTermCfg(
            func=reach_and_place_reward,
            weight=1.0,
            params={
                "object_name": "ball",
                "target_name": "bowl",
                "reaching_std": 0.1,
                "placing_std": 0.1,
                "asset_cfg": _grasp_site(),
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "placed": TerminationTermCfg(
            func=placed_in_bowl,
            params={"object_name": "ball", "target_name": "bowl"},
        ),
    }
    cfg.curriculum = {}
    cfg.events = {
        "reset_ball": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.05, 0.08), "y": (-0.08, 0.08)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("ball"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }
    cfg.viewer.body_name = "link0"
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 400)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg


# ============================================================================
# Behavior-cloning env — same task, action space selected by control mode
# ============================================================================


def build_bc_env_cfg(
    control: str = "joint",
    play: bool = False,
    cameras: bool = False,
    res: int = 96,
) -> ManagerBasedRlEnvCfg:
    """State env wired for behavior cloning under one of two control modes.

    ``control="joint"``: 7 joint-position targets + gripper (action dim 8). The
        joint action is unscaled with a default-pose offset, so a recorded raw
        action is simply ``q_target - home_q`` — a clean expert/BC round-trip.
    ``control="osc"``: 6-D end-effector pose delta + gripper (action dim 7), via
        :class:`~pick_place_challenge.control.OscPoseAction`.

    The gripper term and the rest of the MDP are identical across modes, so demos
    differ only in how the arm is commanded. Observation corruption is disabled
    for clean, reproducible demonstrations. ``cameras=True`` adds scene + wrist RGB
    (a separate obs group, leaving the state obs unchanged) for saving demo videos;
    ``res`` is the square camera resolution (used by image-based BC + the future
    dataloading benchmark, where larger frames cost more to decode).
    """
    cfg = state_env_cfg(play=play)
    gripper = cfg.actions["gripper"]

    if control == "joint":
        arm = cfg.actions["joint_pos"]
        assert isinstance(arm, JointPositionActionCfg)
        arm.actuator_names = _ARM_JOINTS
        arm.scale = 1.0  # raw action = q_target - home_q (use_default_offset=True)
        cfg.actions = {"joint_pos": arm, "gripper": gripper}
    elif control == "osc":
        osc = OscPoseActionCfg(
            entity_name="robot",
            actuator_names=_ARM_JOINTS,
            frame_name=scene.GRASP_SITE,
        )
        cfg.actions = {"osc": osc, "gripper": gripper}
    else:
        raise ValueError(f"control must be 'joint' or 'osc', got {control!r}")

    if cameras:
        _add_cameras(cfg, res=res)
    for group in cfg.observations.values():
        group.enable_corruption = False
    return cfg


def _look_at_quat(eye, target):
    """w,x,y,z quat orienting a MuJoCo camera (looks −z, up +y) at ``target``."""
    fx, fy, fz = (t - e for e, t in zip(eye, target, strict=True))
    fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / fn, fy / fn, fz / fn
    rx, ry, rz = fy, -fx, 0.0
    rn = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    m = [[rx, ux, -fx], [ry, uy, -fy], [rz, uz, -fz]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x, y, z = 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x, y, z = (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x, y, z = (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return (w, x, y, z)


def _camera(name, eye, target, parent_body=None, res=84):
    return CameraSensorCfg(
        name=name,
        camera_name=None,
        parent_body=parent_body,
        pos=eye,
        quat=_look_at_quat(eye, target),
        width=res,
        height=res,
        data_types=("rgb",),
        use_shadows=False,
        use_textures=True,
    )


def _add_cameras(cfg: ManagerBasedRlEnvCfg, res: int = 96) -> None:
    """Add the scene + wrist RGB cameras (and their obs group) to ``cfg`` in place.

    Used by both ``pixels_env_cfg`` and BC demo collection. ``res`` is a multiple
    of 16 so the mp4 encoder doesn't pad the saved frames.
    """
    cfg.scene.sensors = (
        _camera("scene_cam", (1.05, 0.55, 0.6), (0.33, 0.02, 0.05), res=res),
        _camera(
            "wrist_cam",
            (0.14, 0.0, 0.04),
            (0.0, 0.0, 0.13),
            f"robot/{_GRIPPER_BODY}",
            res=res,
        ),
    )
    cfg.observations["camera"] = ObservationGroupCfg(
        terms={
            "scene_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "scene_cam"}
            ),
            "wrist_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "wrist_cam"}
            ),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )


def pixels_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Image-based variant: wrist + scene RGB; ball not given as state."""
    cfg = state_env_cfg(play=play)
    cfg.scene.sensors = (
        _camera("scene_cam", (1.05, 0.55, 0.6), (0.33, 0.02, 0.05)),
        _camera(
            "wrist_cam", (0.14, 0.0, 0.04), (0.0, 0.0, 0.13), f"robot/{_GRIPPER_BODY}"
        ),
    )
    cfg.observations["camera"] = ObservationGroupCfg(
        terms={
            "scene_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "scene_cam"}
            ),
            "wrist_rgb": ObservationTermCfg(
                func=manip_mdp.camera_rgb, params={"sensor_name": "wrist_cam"}
            ),
        },
        enable_corruption=False,
        concatenate_terms=False,
    )
    # Drop privileged ball state; keep the (fixed) bowl position.
    for group in ("actor", "critic"):
        cfg.observations[group].terms.pop("ee_to_ball", None)
        cfg.observations[group].terms.pop("ball_to_bowl", None)
    return cfg


# ============================================================================
# Registration
# ============================================================================


def _rl_cfg(vision: bool, name: str):
    cfg = (
        yam_lift_cube_vision_ppo_runner_cfg()
        if vision
        else yam_lift_cube_ppo_runner_cfg()
    )
    cfg.experiment_name = name
    return cfg


register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-State-v0",
    env_cfg=state_env_cfg(),
    play_env_cfg=state_env_cfg(play=True),
    rl_cfg=_rl_cfg(False, "franka_place_ball_state"),
    runner_cls=ManipulationOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-PlaceBall-Franka-Pixels-v0",
    env_cfg=pixels_env_cfg(),
    play_env_cfg=pixels_env_cfg(play=True),
    rl_cfg=_rl_cfg(True, "franka_place_ball_pixels"),
    runner_cls=ManipulationOnPolicyRunner,
)
