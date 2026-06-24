"""Compare joint vs OSC control: evaluate both trained policies and print a table.

Assumes you have already collected demos and trained a policy for each mode:

    for c in joint osc; do
      uv run python scripts/collect_demos.py --control $c --num-demos 20
      uv run python scripts/train_bc.py --demos demos/$c --out policies/$c.pt
    done
    uv run python scripts/compare.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import tyro

sys.path.insert(0, os.path.dirname(__file__))
from eval_policy import evaluate  # noqa: E402


@dataclass(frozen=True)
class Args:
    episodes: int = 50
    max_steps: int = 300
    seed: int = 0
    """RNG seed — both modes evaluate on the same ball spawns for a fair compare."""
    device: str = "cuda"


def main(args: Args) -> None:
    print(f"{'control':>8} | {'success':>8} | {'mean reward':>11}")
    print("-" * 33)
    for control in ("joint", "osc"):
        res = evaluate(
            control,
            f"policies/{control}.pt",
            args.episodes,
            args.max_steps,
            args.device,
            args.seed,
        )
        print(
            f"{control:>8} | {res['success_rate']:>7.0%} | {res['mean_reward']:>11.2f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
