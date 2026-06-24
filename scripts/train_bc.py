"""Train a behavior-cloning policy on collected demonstrations.

Loads every ``episode_*.npz`` in a demo dir, concatenates them, and fits the small
MLP in :mod:`pick_place_challenge.bc`. The trained policy (with its normalization
stats and control mode) is saved to a single ``.pt`` file.

    uv run python scripts/train_bc.py --demos demos/joint --out policies/joint.pt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from pick_place_challenge import bc


@dataclass(frozen=True)
class Args:
    demos: str = "demos/joint"
    """Directory of episode_*.npz demos (with meta.json)."""
    out: str | None = None
    """Output policy path (default: policies/<control>.pt)."""
    epochs: int = 200
    hidden: int = 256
    batch: int = 256
    lr: float = 1e-3
    chunk: int = bc.CHUNK
    """Actions predicted per inference (executed open-loop at rollout)."""
    device: str = "cpu"


def main(args: Args) -> None:
    demos = Path(args.demos)
    meta = json.loads((demos / "meta.json").read_text())
    files = sorted(demos.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"No demos found in {demos}")

    episodes = [np.load(f) for f in files]
    obs = np.concatenate([e["obs"] for e in episodes])
    # Target = the next `chunk` actions at each step (clamped per episode).
    act = np.concatenate([bc.chunk_targets(e["action"], args.chunk) for e in episodes])
    print(
        f"Loaded {len(files)} demos -> {obs.shape[0]} transitions ({meta['control']}, "
        f"chunk={args.chunk})."
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

    out = Path(args.out or f"policies/{meta['control']}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    bc.save(
        str(out),
        policy,
        stats,
        {
            "control": meta["control"],
            "obs_dim": meta["obs_dim"],
            "act_dim": meta["act_dim"],
            "chunk": args.chunk,
            "out_dim": meta["act_dim"] * args.chunk,
            "hidden": args.hidden,
        },
    )
    print(f"Saved policy to {out}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
