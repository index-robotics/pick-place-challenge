"""Collect scripted pick-and-place demonstrations for behavior cloning.

Runs the :class:`ScriptedExpert` across ``num_demos`` parallel envs (one demo per
env, each with a different random ball spawn) and saves every successful episode
as a directory of parquet streams + mp4 videos (see
:mod:`pick_place_challenge.episode_io`). The control mode selects the action space
the demos are recorded in.

    uv run python scripts/collect_demos.py --control joint --num-demos 20
    uv run python scripts/collect_demos.py --control osc   --num-demos 20 --device cpu
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import pick_place_challenge.task as task  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from pick_place_challenge import episode_io, scene
from pick_place_challenge.expert import ScriptedExpert


@dataclass(frozen=True)
class Args:
    control: str = "joint"
    """Control mode: 'joint' or 'osc'."""
    num_demos: int = 20
    """Number of parallel envs == max demos to collect."""
    out: str | None = None
    """Output dir (default: demos/<control>)."""
    max_steps: int = 450
    """Max env steps to give the expert per episode."""
    device: str = "cuda"
    """'cuda' or 'cpu'."""


def _frames(rgb: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) float in [0, 1] -> (N, H, W, 3) uint8."""
    return (rgb.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8).cpu()


def main(args: Args) -> None:
    out = Path(args.out or f"demos/{args.control}")
    if out.exists():
        shutil.rmtree(out)  # start clean so demo counts are exact
    out.mkdir(parents=True, exist_ok=True)

    cfg = task.build_bc_env_cfg(args.control, cameras=True)
    cfg.scene.num_envs = args.num_demos
    cfg.episode_length_s = 1e6  # never time out; we cut each demo at its success
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    expert = ScriptedExpert(env, args.control)
    robot = env.scene["robot"]
    site = robot.find_sites(scene.GRASP_SITE)[0][0]
    fps = round(1.0 / env.step_dt)

    obs, _ = env.reset()
    expert.reset()
    n = args.num_demos
    done = torch.zeros(n, dtype=torch.bool, device=args.device)
    success_step = torch.full((n,), -1, dtype=torch.long, device=args.device)
    log: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("obs", "act", "eef", "grip", "scene", "wrist")
    }

    for t in range(args.max_steps):
        action = expert.act()
        log["obs"].append(obs["actor"].clone())
        log["act"].append(action.clone())
        eef = torch.cat(
            [robot.data.site_pos_w[:, site], robot.data.site_quat_w[:, site]], dim=-1
        )
        log["eef"].append(eef.clone())
        log["grip"].append(action[:, -1:].clone())
        log["scene"].append(_frames(obs["camera"]["scene_rgb"]))
        log["wrist"].append(_frames(obs["camera"]["wrist_rgb"]))
        obs, _, terminated, _, _ = env.step(action)
        success_step[terminated & ~done] = (
            t  # huge episode_length -> terminated == placed
        )
        done = done | terminated
        if bool(done.all()):
            break

    stacked = {k: torch.stack(v) for k, v in log.items()}  # each (T, n, ...)

    saved = 0
    for i in range(n):
        s = int(success_step[i].item())
        if s < 0:
            continue
        ep = out / f"episode_{saved:03d}"
        ep.mkdir(parents=True, exist_ok=True)
        sl = slice(0, s + 1)
        episode_io.write_stream(ep, "observations", stacked["obs"][sl, i].cpu().numpy())
        episode_io.write_stream(ep, "actions", stacked["act"][sl, i].cpu().numpy())
        episode_io.write_stream(ep, "eef_states", stacked["eef"][sl, i].cpu().numpy())
        episode_io.write_stream(
            ep, "gripper_states", stacked["grip"][sl, i].cpu().numpy()
        )
        episode_io.write_video(ep, "scene_camera", stacked["scene"][sl, i].numpy(), fps)
        episode_io.write_video(ep, "wrist_camera", stacked["wrist"][sl, i].numpy(), fps)
        episode_io.write_metadata(
            ep,
            {
                "control": args.control,
                "num_steps": s + 1,
                "fps": fps,
                "obs_dim": int(stacked["obs"].shape[-1]),
                "act_dim": int(stacked["act"].shape[-1]),
            },
        )
        saved += 1

    meta = {
        "control": args.control,
        "obs_dim": int(stacked["obs"].shape[-1]),
        "act_dim": int(stacked["act"].shape[-1]),
        "fps": fps,
        "num_saved": saved,
        "num_attempted": n,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {saved}/{n} successful demos to {out} ({meta}).")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
