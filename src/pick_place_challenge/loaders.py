"""Faster image-loader backends to compare against the naive mp4 baseline.

Each backend is an :class:`~pick_place_challenge.image_dataset.EpisodeFrameDataset`
that differs only in how it fetches the two RGB frames at ``(episode, t)``. They span
the storage/speed tradeoff the dataloading challenge is about:

- ``naive``  — re-decode the whole mp4 per sample (the baseline; in image_dataset).
- ``seek``   — keep an open ffmpeg reader per file and seek to a single frame. No
               transcode; pays one GOP decode per random access.
- ``jpeg``   — transcode each frame to a standalone JPEG; one tiny decode per sample.
- ``memmap`` — transcode to a raw uint8 ``.npy`` per camera and ``mmap`` it; *zero*
               decode, frame access is a slice. Largest on disk, but the OS page cache
               bounds resident memory (we never load whole videos into RAM).

``jpeg``/``memmap`` need a one-time transcode; :func:`build_dataset` does it lazily into
a hidden ``.loader_cache/<backend>`` under the demos dir and copies the action streams
so each store is self-contained.
"""

from __future__ import annotations

import shutil
from collections import OrderedDict
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np

from pick_place_challenge import bc, episode_io
from pick_place_challenge.image_dataset import DemoImageDataset, EpisodeFrameDataset

_CAMERAS = ("scene_camera", "wrist_camera")
_SIDECARS = ("actions.parquet", "metadata.json")


class _LRU(OrderedDict):
    """Tiny LRU that calls ``close_fn`` on evicted values (for open file handles)."""

    def __init__(self, capacity: int, close_fn=lambda v: None):
        super().__init__()
        self.capacity = capacity
        self.close_fn = close_fn

    def get_or_create(self, key, create_fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = create_fn(key)
        self[key] = val
        if len(self) > self.capacity:
            _, evicted = self.popitem(last=False)
            self.close_fn(evicted)
        return val


# ---------------------------------------------------------------------------
# seek backend — no transcode
# ---------------------------------------------------------------------------


class Mp4SeekDataset(EpisodeFrameDataset):
    """Seek to a single frame via a cached ffmpeg reader (one GOP decode per access)."""

    READER_CACHE = 64  # max simultaneously-open ffmpeg readers (per worker process)

    def __init__(self, demos_dir, chunk: int = bc.CHUNK):
        super().__init__(demos_dir, chunk)
        # Created lazily so the (unpicklable) readers live in each DataLoader worker.
        self._readers: _LRU | None = None

    def _reader(self, path: Path):
        if self._readers is None:
            self._readers = _LRU(self.READER_CACHE, close_fn=lambda r: r.close())
        return self._readers.get_or_create(
            str(path), lambda p: imageio.get_reader(p, format="ffmpeg")
        )

    def _load_frames(self, ep: Path, t: int):
        scene = self._reader(ep / "scene_camera.mp4").get_data(t)
        wrist = self._reader(ep / "wrist_camera.mp4").get_data(t)
        return np.asarray(scene), np.asarray(wrist)


# ---------------------------------------------------------------------------
# jpeg backend — transcode to per-frame JPEGs
# ---------------------------------------------------------------------------


def transcode_to_jpeg(demos_dir: Path, out_dir: Path, quality: int = 95) -> None:
    """Write each frame of every demo as ``<cam>/<t>.jpg`` (+ copy action streams)."""
    for ep in episode_io.list_episodes(demos_dir):
        dst = out_dir / ep.name
        for cam in _CAMERAS:
            frames = episode_io.read_video(ep, cam)  # (T, H, W, 3) uint8
            cam_dir = dst / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            for t, frame in enumerate(frames):
                iio.imwrite(cam_dir / f"{t:04d}.jpg", frame, quality=quality)
        for name in _SIDECARS:
            shutil.copy(ep / name, dst / name)


class JpegFrameDataset(EpisodeFrameDataset):
    """One small JPEG decode per camera per sample."""

    def _load_frames(self, ep: Path, t: int):
        scene = iio.imread(ep / "scene_camera" / f"{t:04d}.jpg")
        wrist = iio.imread(ep / "wrist_camera" / f"{t:04d}.jpg")
        return scene, wrist


# ---------------------------------------------------------------------------
# memmap backend — transcode to raw uint8 .npy, mmap, slice (no decode)
# ---------------------------------------------------------------------------


def transcode_to_memmap(demos_dir: Path, out_dir: Path) -> None:
    """Write each camera's full ``(T, H, W, 3)`` uint8 array as ``<cam>.npy``."""
    for ep in episode_io.list_episodes(demos_dir):
        dst = out_dir / ep.name
        dst.mkdir(parents=True, exist_ok=True)
        for cam in _CAMERAS:
            np.save(dst / f"{cam}.npy", episode_io.read_video(ep, cam))
        for name in _SIDECARS:
            shutil.copy(ep / name, dst / name)


class MemmapDataset(EpisodeFrameDataset):
    """Slice a memory-mapped raw array — no decode; page cache bounds resident RAM."""

    def __init__(self, demos_dir, chunk: int = bc.CHUNK):
        super().__init__(demos_dir, chunk)
        self._maps: dict[str, np.ndarray] | None = None  # lazy, per worker

    def _map(self, path: Path) -> np.ndarray:
        if self._maps is None:
            self._maps = {}
        key = str(path)
        if key not in self._maps:
            self._maps[key] = np.load(path, mmap_mode="r")
        return self._maps[key]

    def _load_frames(self, ep: Path, t: int):
        # np.array(...) materializes just this frame out of the mmap (a copy).
        scene = np.array(self._map(ep / "scene_camera.npy")[t])
        wrist = np.array(self._map(ep / "wrist_camera.npy")[t])
        return scene, wrist


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

BACKENDS = ("naive", "seek", "jpeg", "memmap")
_TRANSCODE = {"jpeg": transcode_to_jpeg, "memmap": transcode_to_memmap}


def store_dir(demos_dir: Path, backend: str) -> Path:
    """Where a transcoded backend caches its store (hidden, so list_episodes skips it)."""
    return Path(demos_dir) / ".loader_cache" / backend


def build_dataset(
    backend: str, demos_dir: str | Path, chunk: int = bc.CHUNK, rebuild: bool = False
) -> EpisodeFrameDataset:
    """Construct a backend dataset, transcoding into a cache dir first if needed."""
    demos_dir = Path(demos_dir)
    if backend == "naive":
        return DemoImageDataset(demos_dir, chunk)
    if backend == "seek":
        return Mp4SeekDataset(demos_dir, chunk)
    if backend not in _TRANSCODE:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    out = store_dir(demos_dir, backend)
    n_src = len(episode_io.list_episodes(demos_dir))
    # Rebuild unless the store already has an episode dir per source demo.
    if rebuild or len(episode_io.list_episodes(out)) != n_src:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        _TRANSCODE[backend](demos_dir, out)
    return (JpegFrameDataset if backend == "jpeg" else MemmapDataset)(out, chunk)


def store_bytes(backend: str, demos_dir: str | Path) -> int:
    """Total on-disk bytes the backend reads from (source mp4s, or its transcoded store)."""
    demos_dir = Path(demos_dir)
    if backend in ("naive", "seek"):
        roots = [
            ep / f"{cam}.mp4"
            for ep in episode_io.list_episodes(demos_dir)
            for cam in _CAMERAS
        ]
        return sum(p.stat().st_size for p in roots if p.exists())
    out = store_dir(demos_dir, backend)
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
