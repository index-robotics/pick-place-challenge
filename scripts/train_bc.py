"""Train a behavior-cloning policy on collected demonstrations.

Loads every ``episode_*`` directory in a demo dir, concatenates them, and fits the
small MLP in :mod:`pick_place_challenge.bc`. The trained policy (with its
normalization stats and chunk size) is saved to a single ``.pt`` file.

    uv run python scripts/train_bc.py --demos demos --out policies/bc.pt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from pick_place_challenge import bc, episode_io, run_dir


@dataclass(frozen=True)
class Args:
    demos: str = "demos"
    """Directory of episode_* demos (with meta.json)."""
    out: str = "policies/bc.pt"
    """Output policy path."""
    epochs: int = 200
    hidden: int = 256
    batch: int = 256
    lr: float = 1e-3
    chunk: int = bc.CHUNK
    """Actions predicted per inference (executed open-loop at rollout)."""
    seed: int = 0
    """RNG seed for reproducible weight init + minibatch shuffling."""
    device: str = "cpu"


def main(args: Args) -> None:
    torch.manual_seed(args.seed)  # reproducible weight init + minibatch shuffling
    np.random.seed(args.seed)
    demos = Path(args.demos)
    meta = json.loads((demos / "meta.json").read_text())
    episodes = episode_io.list_episodes(demos)
    if not episodes:
        raise SystemExit(f"No demos found in {demos}")

    obs = np.concatenate([episode_io.read_stream(d, "observations") for d in episodes])
    # Target = the next `chunk` actions at each step (clamped per episode).
    act = np.concatenate(
        [
            bc.chunk_targets(episode_io.read_stream(d, "actions"), args.chunk)
            for d in episodes
        ]
    )
    print(
        f"Loaded {len(episodes)} demos -> {obs.shape[0]} transitions "
        f"(chunk={args.chunk})."
    )

    policy, stats = bc.fit(
        torch.as_tensor(obs, dtype=torch.float32),
        torch.as_tensor(act, dtype=torch.float32),
        hidden=args.hidden,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
    )

    policy_meta = {
        "obs_dim": meta["obs_dim"],
        "act_dim": meta["act_dim"],
        "chunk": args.chunk,
        "out_dim": meta["act_dim"] * args.chunk,
        "hidden": args.hidden,
    }

    # Archive the run (checkpoint + config) under exp_local/, and also write the
    # flat --out path as the convenient "latest" pointer for eval.
    run = run_dir.new_run_dir("train")
    bc.save(str(run / "policy.pt"), policy, stats, policy_meta)
    run_dir.write_json(
        run,
        "config",
        {
            **policy_meta,
            "demos": str(demos),
            "num_demos": len(episodes),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
        },
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bc.save(str(out), policy, stats, policy_meta)
    print(f"Saved policy to {out} and archived run to {run}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
