"""Episode storage: parquet streams + mp4 videos, mechacarpal-style.

Each demonstration is a *directory* of per-stream files (mirroring mechacarpal's
layout), one row per environment step:

    episode_000/
        metadata.json          # step count, fps, stream dims
        observations.parquet    # (T, obs_dim)  — the BC policy input
        actions.parquet         # (T, act_dim)  — the recorded expert action
        eef_states.parquet      # (T, 7)        — end-effector position + quaternion
        gripper_states.parquet  # (T, 1)        — gripper command
        scene_camera.mp4        # (T, H, W, 3)  — fixed scene view
        wrist_camera.mp4        # (T, H, W, 3)  — wrist view

Arrays are stored with pyarrow's ``FixedShapeTensorArray`` so per-step vectors
round-trip without flattening. Unlike mechacarpal we keep no per-stream timestamp
column: everything runs at one fixed control rate, so the row index *is* the time
index (streams are already aligned).
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_DATA_COLUMN = "data"


def write_stream(episode_dir: Path, name: str, data: np.ndarray) -> None:
    """Write ``(T, ...)`` per-step data to ``{episode_dir}/{name}.parquet``."""
    arr = np.asarray(data)
    table = pa.table({_DATA_COLUMN: pa.FixedShapeTensorArray.from_numpy_ndarray(arr)})
    pq.write_table(table, episode_dir / f"{name}.parquet")


def read_stream(episode_dir: Path, name: str) -> np.ndarray:
    """Read a parquet stream back into a ``(T, ...)`` array."""
    column = pq.read_table(episode_dir / f"{name}.parquet")[_DATA_COLUMN]
    return column.combine_chunks().to_numpy_ndarray()


def write_video(episode_dir: Path, name: str, frames: np.ndarray, fps: int) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames to ``{episode_dir}/{name}.mp4``."""
    iio.imwrite(episode_dir / f"{name}.mp4", frames, fps=fps, codec="libx264")


def read_video(episode_dir: Path, name: str) -> np.ndarray:
    """Read an mp4 back into ``(T, H, W, 3)`` uint8 frames."""
    return iio.imread(episode_dir / f"{name}.mp4")


def write_metadata(episode_dir: Path, meta: dict) -> None:
    (episode_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def read_metadata(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "metadata.json").read_text())


def list_episodes(demos_dir: Path) -> list[Path]:
    """Sorted ``episode_*`` directories under a demos dir."""
    return sorted(d for d in Path(demos_dir).glob("episode_*") if d.is_dir())
