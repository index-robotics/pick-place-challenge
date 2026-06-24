"""Train a behavior-cloning policy on collected demonstrations.

Loads every ``episode_*`` demo in a demo dir and fits a policy from
:mod:`pick_place_challenge.bc`. With ``--obs state`` (default) it concatenates the
low-dim observation streams and fits the small MLP; with ``--obs image`` it streams
scene+wrist RGB through the naive mp4 dataset and fits the CNN policy. The trained
policy (with its normalization stats and control mode) is saved to a single ``.pt``
file.

    uv run python scripts/train_bc.py --demos demos/joint --out policies/joint.pt
    uv run python scripts/train_bc.py --demos demos/joint --obs image
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
    demos: str = "demos/joint"
    """Directory of episode_* demos (with meta.json)."""
    obs: str = "state"
    """Policy input: 'state' (low-dim obs MLP) or 'image' (scene+wrist RGB CNN)."""
    out: str | None = None
    """Output policy path (default: policies/<control>.pt)."""
    epochs: int = 200
    hidden: int = 256
    batch: int = 256
    lr: float = 1e-3
    chunk: int = bc.CHUNK
    """Actions predicted per inference (executed open-loop at rollout)."""
    num_workers: int = 0
    """DataLoader workers for image mode (0 keeps the naive loader single-process)."""
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

    policy_meta = {
        "control": meta["control"],
        "obs_mode": args.obs,
        "obs_dim": meta["obs_dim"],
        "act_dim": meta["act_dim"],
        "chunk": args.chunk,
        "out_dim": meta["act_dim"] * args.chunk,
        "hidden": args.hidden,
    }

    if args.obs == "image":
        # Naive mp4-backed dataset: each sample re-decodes its episode's videos (see
        # image_dataset). The future challenge is to make this loader fast.
        from torch.utils.data import DataLoader

        from pick_place_challenge.image_dataset import DemoImageDataset

        dataset = DemoImageDataset(demos, chunk=args.chunk)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
        )
        act_mean, act_std = dataset.action_stats()
        print(
            f"Loaded {len(episodes)} demos -> {len(dataset)} transitions "
            f"({meta['control']}, image res={meta['res']}, chunk={args.chunk})."
        )
        policy, stats = bc.fit_image(
            loader,
            out_dim=dataset.out_dim,
            act_mean=act_mean,
            act_std=act_std,
            hidden=args.hidden,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
        )
        policy_meta["res"] = meta["res"]
    elif args.obs == "state":
        obs = np.concatenate(
            [episode_io.read_stream(d, "observations") for d in episodes]
        )
        # Target = the next `chunk` actions at each step (clamped per episode).
        act = np.concatenate(
            [
                bc.chunk_targets(episode_io.read_stream(d, "actions"), args.chunk)
                for d in episodes
            ]
        )
        print(
            f"Loaded {len(episodes)} demos -> {obs.shape[0]} transitions "
            f"({meta['control']}, chunk={args.chunk})."
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
    else:
        raise SystemExit(f"--obs must be 'state' or 'image', got {args.obs!r}")

    # Archive the run (checkpoint + config) under exp_local/, and also write the
    # flat --out path as the convenient "latest" pointer for eval/compare.
    run = run_dir.new_run_dir(f"train_{meta['control']}")
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

    out = Path(args.out or f"policies/{meta['control']}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    bc.save(str(out), policy, stats, policy_meta)
    print(f"Saved policy to {out} and archived run to {run}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
