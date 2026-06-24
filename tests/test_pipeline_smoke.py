"""Smoke tests for the imitation-learning pipeline: control modes + BC.

These build a tiny env and step it, so they need MuJoCo-Warp (CPU backend is fine;
a GPU is faster). The scripted expert, both control modes, and the BC policy are
all exercised end-to-end on 2 envs for a handful of steps — enough to catch wiring
breakage without a full training run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pick_place_challenge.task as task  # noqa: F401  (registers tasks)
from pick_place_challenge import bc, episode_io
from pick_place_challenge.expert import ScriptedExpert

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _env(control: str, num_envs: int = 2):
    from mjlab.envs import ManagerBasedRlEnv

    cfg = task.build_bc_env_cfg(control)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1e6
    return ManagerBasedRlEnv(cfg=cfg, device=_DEVICE)


@pytest.mark.parametrize("control,action_dim", [("joint", 8), ("osc", 7)])
def test_control_mode_expert_steps(control: str, action_dim: int) -> None:
    """Each control mode builds, has the right action dim, and the expert drives it."""
    env = _env(control)
    try:
        assert env.action_space.shape[-1] == action_dim
        expert = ScriptedExpert(env, control)
        obs, _ = env.reset()
        expert.reset()
        for _ in range(3):
            action = expert.act()
            assert action.shape == (2, action_dim)
            assert torch.isfinite(action).all()
            obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_bc_train_and_act_tiny() -> None:
    """Collect a few expert transitions, fit the BC MLP, and produce an action."""
    env = _env("joint")
    try:
        expert = ScriptedExpert(env, "joint")
        obs, _ = env.reset()
        expert.reset()
        obs_log, act_log = [], []
        for _ in range(20):
            action = expert.act()
            obs_log.append(obs["actor"].clone())
            act_log.append(action.clone())
            obs, *_ = env.step(action)
        obs_t = torch.cat(obs_log)
        act_t = torch.cat(act_log)
    finally:
        env.close()

    policy, stats = bc.fit(obs_t, act_t, epochs=2, device=_DEVICE)
    out = bc.act(policy, stats, obs_t[:4].to(_DEVICE))
    assert out.shape == (4, act_t.shape[1])
    assert torch.isfinite(out).all()


def test_episode_io_roundtrip(tmp_path) -> None:
    """Parquet streams and mp4 videos written by collect_demos round-trip."""
    ep = tmp_path / "episode_000"
    ep.mkdir()
    obs = np.random.rand(10, 5).astype(np.float32)
    frames = (np.random.rand(10, 96, 96, 3) * 255).astype(np.uint8)
    episode_io.write_stream(ep, "observations", obs)
    episode_io.write_video(ep, "scene_camera", frames, fps=50)
    episode_io.write_metadata(ep, {"control": "joint", "num_steps": 10})

    back = episode_io.read_stream(ep, "observations")
    assert back.shape == (10, 5) and np.allclose(back, obs)
    vid = episode_io.read_video(ep, "scene_camera")
    assert vid.shape[0] == 10 and vid.shape[-1] == 3
    assert episode_io.list_episodes(tmp_path) == [ep]
    assert episode_io.read_metadata(ep)["control"] == "joint"
