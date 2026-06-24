"""A deliberately naive MP4-backed dataset for image behavior cloning.

Each sample is one timestep of one demo: the scene + wrist camera frames at step
``t`` paired with the chunked expert-action target. The catch — and the whole point
of the future challenge — is that :meth:`DemoImageDataset.__getitem__` **re-decodes
the entire mp4 of both cameras on every single access** (via
:func:`episode_io.read_video`) just to pull out one frame. That is wildly wasteful:
fetching all ``T`` steps of an episode decodes each video ``T`` times.

This naivety is intentional. It establishes a slow, obviously-correct baseline whose
job is to be optimized: caching decoded clips, seeking to a single frame, batching by
episode, prefetching, a smarter on-disk format, etc. **Do not "fix" the redundant
decode here** — that is the exercise.

Only the small per-step streams (actions) are loaded up front; image bytes stay on
disk and are decoded lazily per sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pick_place_challenge import bc, episode_io


class DemoImageDataset(Dataset):
    """Timestep-level (scene, wrist, action-chunk) samples decoded naively from mp4.

    Args:
        demos_dir: directory of ``episode_*`` demos (as written by ``collect_demos``).
        chunk: action-chunk length, matched to the policy (see :func:`bc.chunk_targets`).
    """

    def __init__(self, demos_dir: str | Path, chunk: int = bc.CHUNK):
        self.episodes = episode_io.list_episodes(Path(demos_dir))
        if not self.episodes:
            raise FileNotFoundError(f"No episode_* demos found in {demos_dir}")
        self.chunk = chunk

        # Load only the (small) action streams now; precompute chunked targets and a
        # flat (episode_idx, t) index. Images are NOT touched here — they are decoded
        # lazily, one mp4 per __getitem__, on purpose.
        self._targets: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            actions = episode_io.read_stream(ep, "actions")
            targets = bc.chunk_targets(actions, chunk)  # (T, chunk*act_dim)
            self._targets.append(targets.astype(np.float32))
            self._index.extend((ep_idx, t) for t in range(targets.shape[0]))

        self.act_dim = int(episode_io.read_stream(self.episodes[0], "actions").shape[-1])

    @property
    def out_dim(self) -> int:
        """Flattened action-chunk dimension the policy regresses (``chunk * act_dim``)."""
        return self.chunk * self.act_dim

    def action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean/std of the chunked action targets (for normalization)."""
        all_targets = torch.as_tensor(np.concatenate(self._targets), dtype=torch.float32)
        return all_targets.mean(0), all_targets.std(0).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, t = self._index[i]
        ep = self.episodes[ep_idx]
        # NAIVE: decode the whole mp4 for each camera, every call, to grab one frame.
        scene = episode_io.read_video(ep, "scene_camera")[t]  # (H, W, 3) uint8
        wrist = episode_io.read_video(ep, "wrist_camera")[t]  # (H, W, 3) uint8
        target = self._targets[ep_idx][t]  # (chunk*act_dim,) float32
        return (
            torch.from_numpy(np.ascontiguousarray(scene)),
            torch.from_numpy(np.ascontiguousarray(wrist)),
            torch.from_numpy(target),
        )
