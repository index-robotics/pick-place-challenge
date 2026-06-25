"""Evaluate a trained BC policy: roll it out and report the success rate.

Runs ``episodes`` parallel rollouts and counts how many reach the bowl (the
``placed_in_bowl`` termination).

    uv run python scripts/eval_policy.py --policy policies/bc.pt
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

import pick_place_challenge.task as task  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from pick_place_challenge import bc, run_dir


def evaluate(
    policy_path: str,
    episodes: int = 50,
    max_steps: int = 300,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Roll out the policy and return ``{success_rate, mean_reward, episodes}``.

    Archives the result to ``exp_local/<date>/<time>_eval/`` (config + metrics) so
    runs are saved rather than just printed.
    """
    policy, stats, meta = bc.load(policy_path, device)

    cfg = task.build_bc_env_cfg()
    cfg.scene.num_envs = episodes
    cfg.seed = seed  # seeds numpy/torch/warp -> reproducible ball spawns
    cfg.episode_length_s = 1e6  # we stop at success or max_steps
    env = ManagerBasedRlEnv(cfg=cfg, device=device)

    chunk, act_dim = meta["chunk"], meta["act_dim"]
    obs, _ = env.reset()
    done = torch.zeros(episodes, dtype=torch.bool, device=device)
    success = torch.zeros(episodes, dtype=torch.bool, device=device)
    reward_sum = torch.zeros(episodes, device=device)
    actions: torch.Tensor | None = None
    for t in range(max_steps):
        if actions is None or t % chunk == 0:  # predict a fresh chunk, exec open-loop
            actions = bc.act(policy, stats, obs["actor"]).view(episodes, chunk, act_dim)
        obs, reward, terminated, _, _ = env.step(actions[:, t % chunk])
        reward_sum += reward * (~done)
        success = success | (terminated & ~done)
        done = done | terminated
        if bool(done.all()):
            break

    env.close()
    metrics: dict[str, object] = {
        "success_rate": float(success.float().mean()),
        "mean_reward": float(reward_sum.mean()),
        "successes": success.cpu().tolist(),
        "episodes": episodes,
    }
    run = run_dir.new_run_dir("eval")
    run_dir.write_json(
        run,
        "config",
        {
            "policy": policy_path,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "device": device,
        },
    )
    run_dir.write_json(run, "metrics", metrics)
    metrics["run_dir"] = str(run)
    return metrics


@dataclass(frozen=True)
class Args:
    policy: str = "policies/bc.pt"
    """Policy path."""
    episodes: int = 50
    max_steps: int = 300
    seed: int = 0
    """RNG seed for reproducible eval ball spawns."""
    device: str = "cuda"


def main(args: Args) -> None:
    res = evaluate(args.policy, args.episodes, args.max_steps, args.device, args.seed)
    print(
        f"success {res['success_rate']:.0%} "
        f"over {res['episodes']} episodes  (mean reward {res['mean_reward']:.2f})"
    )
    print(f"  saved to {res['run_dir']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
