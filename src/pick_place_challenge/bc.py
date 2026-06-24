"""A minimal behavior-cloning policy: a small MLP with action chunking.

Deliberately tiny — the point of the challenge is the *control mode*, not the
learning algorithm. The MLP maps one observation to the next ``chunk`` actions;
at rollout we execute a chunk open-loop before predicting again. Chunking lets the
policy commit to multi-step motions (notably the descend→close→lift grasp), which
a single-step policy tends to smear away. Observations and actions are z-scored;
the policy predicts normalized actions, which :func:`act` denormalizes back into
the env's raw actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

CHUNK = 16  # actions predicted (and executed open-loop) per inference


def chunk_targets(actions: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """Stack the next ``chunk`` actions for each step: ``(T, A) -> (T, chunk*A)``.

    The window is clamped at the episode end (the last action repeats), so the
    policy learns to "hold" once the task is done.
    """
    t, a = actions.shape
    idx = np.clip(np.arange(t)[:, None] + np.arange(chunk)[None, :], 0, t - 1)
    return actions[idx].reshape(t, chunk * a)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return self.net(obs_norm)


@dataclass
class Stats:
    """Per-dimension normalization for observations and actions."""

    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    act_mean: torch.Tensor
    act_std: torch.Tensor


def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.mean(0), x.std(0).clamp_min(1e-6)


def fit(
    obs: torch.Tensor,
    act: torch.Tensor,
    *,
    hidden: int = 256,
    epochs: int = 200,
    batch: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[MLPPolicy, Stats]:
    """Train the MLP to regress actions from observations (MSE)."""
    obs, act = obs.to(device), act.to(device)
    om, os_ = _stats(obs)
    am, as_ = _stats(act)
    stats = Stats(om, os_, am, as_)
    obs_n, act_n = (obs - om) / os_, (act - am) / as_

    policy = MLPPolicy(obs.shape[1], act.shape[1], hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = obs.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss = nn.functional.mse_loss(policy(obs_n[idx]), act_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  mse {total / n:.5f}")
    return policy, stats


@torch.no_grad()
def act(policy: MLPPolicy, stats: Stats, obs: torch.Tensor) -> torch.Tensor:
    """Map a batch of raw observations to raw env actions."""
    obs_n = (obs - stats.obs_mean) / stats.obs_std
    return policy(obs_n) * stats.act_std + stats.act_mean


def save(path: str, policy: MLPPolicy, stats: Stats, meta: dict) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "stats": vars(stats),
            "meta": meta,  # control, obs_dim, act_dim, hidden
        },
        path,
    )


def load(path: str, device: str = "cpu") -> tuple[MLPPolicy, Stats, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    policy = MLPPolicy(meta["obs_dim"], meta["out_dim"], meta["hidden"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    stats = Stats(**{k: v.to(device) for k, v in ckpt["stats"].items()})
    return policy, stats, meta
