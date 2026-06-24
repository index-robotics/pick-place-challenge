"""Episode-frame datasets for image behavior cloning, incl. the naive mp4 baseline.

A sample is one timestep of one demo: the scene + wrist camera frames at step ``t``
paired with the chunked expert-action target. :class:`EpisodeFrameDataset` holds the
shared bookkeeping (episode list, chunked action targets, flat index, action stats);
subclasses only implement :meth:`_load_frames`, i.e. *how* the two RGB frames at
``(episode, t)`` are fetched. Alternative fetch strategies (seek, transcoded stores)
live in :mod:`pick_place_challenge.loaders` and are compared by
``scripts/benchmark_dataloading.py``.

:class:`DemoImageDataset` is the **deliberately naive baseline**: every
``__getitem__`` re-decodes the *entire* mp4 of both cameras (via
:func:`episode_io.read_video`) just to pull out one frame, so fetching all ``T`` steps
of an episode decodes each video ``T`` times. That waste is intentional — it is the
slow, obviously-correct baseline the dataloading challenge optimizes against. **Do not
"fix" the redundant decode here**; write a faster backend instead.

Only the small per-step streams (actions) are loaded up front; image bytes stay on
disk and are fetched lazily per sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pick_place_challenge import bc, episode_io


class EpisodeFrameDataset(Dataset):
    """Shared base: builds the ``(episode, t)`` index and chunked action targets.

    Subclasses implement :meth:`_load_frames` to return the scene + wrist RGB frames
    at one timestep; everything else (indexing, targets, normalization stats) is
    common, so backends differ *only* in how they read pixels.

    Args:
        demos_dir: directory of ``episode_*`` demos (parquet streams + frame storage).
        chunk: action-chunk length, matched to the policy (see :func:`bc.chunk_targets`).
    """

    def __init__(self, demos_dir: str | Path, chunk: int = bc.CHUNK):
        self.demos_dir = Path(demos_dir)
        self.episodes = episode_io.list_episodes(self.demos_dir)
        if not self.episodes:
            raise FileNotFoundError(f"No episode_* demos found in {demos_dir}")
        self.chunk = chunk

        # Load only the (small) action streams now; precompute chunked targets and a
        # flat (episode_idx, t) index. Image bytes are fetched lazily in _load_frames.
        self._targets: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            actions = episode_io.read_stream(ep, "actions")
            targets = bc.chunk_targets(actions, chunk)  # (T, chunk*act_dim)
            self._targets.append(targets.astype(np.float32))
            self._index.extend((ep_idx, t) for t in range(targets.shape[0]))

        self.act_dim = int(
            episode_io.read_stream(self.episodes[0], "actions").shape[-1]
        )

    @property
    def out_dim(self) -> int:
        """Flattened action-chunk dimension the policy regresses (``chunk * act_dim``)."""
        return self.chunk * self.act_dim

    def action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension mean/std of the chunked action targets (for normalization)."""
        all_targets = torch.as_tensor(
            np.concatenate(self._targets), dtype=torch.float32
        )
        return all_targets.mean(0), all_targets.std(0).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self._index)

    def _load_frames(self, ep: Path, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scene, wrist)`` ``(H, W, 3)`` uint8 frames at step ``t``."""
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx, t = self._index[i]
        scene, wrist = self._load_frames(self.episodes[ep_idx], t)
        target = self._targets[ep_idx][t]  # (chunk*act_dim,) float32
        return (
            torch.from_numpy(np.ascontiguousarray(scene)),
            torch.from_numpy(np.ascontiguousarray(wrist)),
            torch.from_numpy(target),
        )


class DemoImageDataset(EpisodeFrameDataset):
    """Naive baseline: re-decode the whole mp4 of both cameras on every sample.

    This is wildly redundant (each frame is decoded ``T`` times across an epoch) and
    that is the point — it is the slow baseline the dataloading challenge beats. See
    the module docstring; faster backends live in :mod:`pick_place_challenge.loaders`.
    """

    def _load_frames(self, ep: Path, t: int) -> tuple[np.ndarray, np.ndarray]:
        # NAIVE: decode the entire mp4 for each camera, every call, to grab one frame.
        scene = episode_io.read_video(ep, "scene_camera")[t]
        wrist = episode_io.read_video(ep, "wrist_camera")[t]
        return scene, wrist
