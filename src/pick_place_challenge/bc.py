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
from tqdm.auto import tqdm

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


def _cnn_encoder() -> nn.Sequential:
    """Tiny conv stack ending in a global pool, so the feature dim is independent of
    the input resolution (lets one architecture train/eval at any ``--res``)."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),  # -> (B, 64, 1, 1), resolution-independent
        nn.Flatten(),  # -> (B, 64)
    )


class ImagePolicy(nn.Module):
    """Image-only BC policy: a per-camera CNN encoder + MLP head over the chunked action.

    Consumes scene + wrist RGB as ``(B, 3, H, W)`` float in ``[0, 1]`` (use
    :func:`prep_frames` to convert the dataset's uint8 ``(B, H, W, 3)`` frames). The
    adaptive-pool encoders make the head's input size fixed regardless of ``H, W``.
    """

    FEAT = 64  # per-camera feature width (matches _cnn_encoder's final channels)

    def __init__(self, out_dim: int, hidden: int = 256):
        super().__init__()
        self.scene_enc = _cnn_encoder()
        self.wrist_enc = _cnn_encoder()
        self.head = nn.Sequential(
            nn.Linear(2 * self.FEAT, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, scene: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([self.scene_enc(scene), self.wrist_enc(wrist)], dim=-1)
        return self.head(feats)


def prep_frames(frames: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` uint8 -> ``(B, 3, H, W)`` float in ``[0, 1]`` for the encoder."""
    return frames.permute(0, 3, 1, 2).float() / 255.0


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
    bar = tqdm(range(epochs), desc="train (state)", unit="epoch")
    for epoch in bar:
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss = nn.functional.mse_loss(policy(obs_n[idx]), act_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        bar.set_postfix(mse=f"{total / n:.5f}")
    return policy, stats


@torch.no_grad()
def act(policy: MLPPolicy, stats: Stats, obs: torch.Tensor) -> torch.Tensor:
    """Map a batch of raw observations to raw env actions."""
    obs_n = (obs - stats.obs_mean) / stats.obs_std
    return policy(obs_n) * stats.act_std + stats.act_mean


def fit_image(
    loader,
    *,
    out_dim: int,
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    hidden: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[ImagePolicy, Stats]:
    """Train :class:`ImagePolicy` to regress chunked actions from camera frames.

    Mirrors :func:`fit` but pulls minibatches from a ``DataLoader`` (frames stay on
    disk and are decoded per sample) instead of an in-memory tensor. Actions are
    z-scored with the precomputed ``act_mean``/``act_std``; images are scaled in
    :func:`prep_frames`. The returned :class:`Stats` carries dummy obs fields (unused
    in image mode) so save/load stay uniform with the state policy.
    """
    am, as_ = act_mean.to(device), act_std.to(device)
    policy = ImagePolicy(out_dim, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for epoch in range(epochs):
        total, count = 0.0, 0
        # Per-batch bar (advances one batch per tick). We surface images/sec in the
        # postfix — that throughput is the metric the dataloading challenge is judged
        # on. `leave=False` collapses the bar at the end of each epoch.
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
        for scene, wrist, target in bar:
            scene = prep_frames(scene).to(device)
            wrist = prep_frames(wrist).to(device)
            target_n = (target.to(device) - am) / as_
            loss = nn.functional.mse_loss(policy(scene, wrist), target_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(target)
            count += len(target)
            rate = bar.format_dict["rate"]  # batches/sec from tqdm's timing
            img_s = rate * loader.batch_size if rate else 0.0
            bar.set_postfix(mse=f"{total / max(count, 1):.5f}", img_s=f"{img_s:.0f}")
        tqdm.write(f"  epoch {epoch + 1:4d}/{epochs}  mse {total / max(count, 1):.5f}")
    stats = Stats(torch.zeros(1, device=device), torch.ones(1, device=device), am, as_)
    return policy, stats


@torch.no_grad()
def act_image(
    policy: ImagePolicy, stats: Stats, scene: torch.Tensor, wrist: torch.Tensor
) -> torch.Tensor:
    """Map a batch of env camera frames to raw env actions.

    ``scene``/``wrist`` are ``(N, 3, H, W)`` float in ``[0, 1]`` (the form
    ``manip_mdp.camera_rgb`` already produces at rollout), so no decode is needed here.
    """
    return policy(scene, wrist) * stats.act_std + stats.act_mean


def save(
    path: str, policy: MLPPolicy | ImagePolicy, stats: Stats, meta: dict
) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "stats": vars(stats),
            "meta": meta,  # control, obs_dim, act_dim, hidden
        },
        path,
    )


def load(
    path: str, device: str = "cpu"
) -> tuple[MLPPolicy | ImagePolicy, Stats, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    if meta.get("obs_mode", "state") == "image":
        policy: MLPPolicy | ImagePolicy = ImagePolicy(meta["out_dim"], meta["hidden"])
    else:
        policy = MLPPolicy(meta["obs_dim"], meta["out_dim"], meta["hidden"])
    policy = policy.to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    stats = Stats(**{k: v.to(device) for k, v in ckpt["stats"].items()})
    return policy, stats, meta
