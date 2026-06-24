"""Collect scripted pick-and-place demonstrations for behavior cloning.

Runs the :class:`ScriptedExpert` across ``num_demos`` parallel envs (one demo per
env, each with a different random ball spawn) and saves every successful episode
as ``episode_XXX.npz`` (``obs``, ``action``). The control mode selects the action
space the demos are recorded in.

    uv run python scripts/collect_demos.py --control joint --num-demos 20
    uv run python scripts/collect_demos.py --control osc   --num-demos 20 --device cpu
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import pick_place_challenge.task as task  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from pick_place_challenge.expert import ScriptedExpert


@dataclass(frozen=True)
class Args:
    control: str = "joint"
    """Control mode: 'joint' or 'osc'."""
    num_demos: int = 20
    """Number of parallel envs == max demos to collect."""
    out: str | None = None
    """Output dir (default: demos/<control>)."""
    max_steps: int = 300
    """Max env steps to give the expert per episode."""
    device: str = "cuda"
    """'cuda' or 'cpu'."""


def main(args: Args) -> None:
    out = Path(args.out or f"demos/{args.control}")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("episode_*.npz"):  # start clean so counts are exact
        stale.unlink()

    cfg = task.build_bc_env_cfg(args.control)
    cfg.scene.num_envs = args.num_demos
    cfg.episode_length_s = 1e6  # never time out; we cut each demo at its success
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    expert = ScriptedExpert(env, args.control)

    obs, _ = env.reset()
    expert.reset()
    n = args.num_demos
    done = torch.zeros(n, dtype=torch.bool, device=args.device)
    success_step = torch.full((n,), -1, dtype=torch.long, device=args.device)
    obs_log: list[torch.Tensor] = []
    act_log: list[torch.Tensor] = []

    for t in range(args.max_steps):
        action = expert.act()
        obs_log.append(obs["actor"].clone())
        act_log.append(action.clone())
        obs, _, terminated, _, _ = env.step(action)
        newly = terminated & ~done  # episode_length is huge -> terminated == placed
        success_step[newly] = t
        done = done | terminated
        if bool(done.all()):
            break

    obs_log = torch.stack(obs_log)  # (T, n, obs_dim)
    act_log = torch.stack(act_log)  # (T, n, act_dim)

    saved = 0
    for i in range(n):
        s = int(success_step[i].item())
        if s < 0:
            continue
        np.savez(
            out / f"episode_{saved:03d}.npz",
            obs=obs_log[: s + 1, i].cpu().numpy(),
            action=act_log[: s + 1, i].cpu().numpy(),
        )
        saved += 1

    meta = {
        "control": args.control,
        "obs_dim": int(obs_log.shape[-1]),
        "act_dim": int(act_log.shape[-1]),
        "num_saved": saved,
        "num_attempted": n,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {saved}/{n} successful demos to {out} ({meta}).")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
