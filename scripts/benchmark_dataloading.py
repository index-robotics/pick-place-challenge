"""Benchmark image-dataloader backends: how fast can we feed frames to training?

Iterates each backend through a ``DataLoader`` exactly as image BC would and reports
throughput (images/sec) plus per-batch latency and on-disk size, so the storage/speed
tradeoff is concrete rather than theoretical. The naive mp4 loader is the baseline the
challenge optimizes against; ``seek``/``jpeg``/``memmap`` are reference faster backends
(see :mod:`pick_place_challenge.loaders`).

The ``dali`` backend is GPU-native: it reads the same per-frame JPEG store as ``jpeg``
but decodes on the GPU (nvJPEG) via an NVIDIA DALI pipeline, so frames land on-device
ready for the model. It needs the optional ``dali`` extra + a GPU; if either is missing
the row is skipped. Enable it with ``uv sync --extra dali``.

    uv run python scripts/benchmark_dataloading.py --demos demos/joint
    uv run python scripts/benchmark_dataloading.py --demos demos/joint --num-workers 4 --device cuda
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import tyro
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pick_place_challenge import bc, episode_io, loaders

# dali (GPU nvJPEG on the JPEG store) and dali_video (GPU NVDEC straight from the
# mp4s, no transcode) are GPU-native and measured on their own path.
ALL_BACKENDS = (*loaders.BACKENDS, "dali", "dali_video")


@dataclass(frozen=True)
class Args:
    demos: str = "demos/joint"
    """Demo dir to benchmark against (its mp4s feed every backend)."""
    backends: tuple[str, ...] = ALL_BACKENDS
    """Backends to compare (subset of naive/seek/jpeg/memmap/dali/dali_video)."""
    batch: int = 64
    num_workers: int = 0
    """DataLoader workers. 0 = single-process (matches the naive default)."""
    epochs: int = 3
    """Timed passes over the dataset (after a short warmup pass)."""
    chunk: int = bc.CHUNK
    device: str = "cpu"
    """If 'cuda', also include the H2D copy + prep_frames in the measured pipeline."""
    seed: int = 0


def _run_epoch(loader: DataLoader, device: str, transfer: bool) -> int:
    """Iterate one epoch, optionally pushing each batch to the GPU. Returns #images."""
    seen = 0
    for scene, wrist, _ in loader:
        if transfer:
            bc.prep_frames(scene).to(device, non_blocking=True)
            bc.prep_frames(wrist).to(device, non_blocking=True)
        seen += scene.shape[0]
    if transfer and device == "cuda":
        torch.cuda.synchronize()
    return seen


def _benchmark(backend: str, args: Args) -> dict:
    transfer = args.device == "cuda"
    t0 = time.perf_counter()
    dataset = loaders.build_dataset(backend, args.demos, chunk=args.chunk)
    prep_s = time.perf_counter() - t0  # transcode (jpeg/memmap) happens here, once

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=transfer,
        persistent_workers=args.num_workers > 0,
    )

    _run_epoch(loader, args.device, transfer)  # warmup (primes page cache / readers)

    t0 = time.perf_counter()
    total = 0
    for _ in tqdm(range(args.epochs), desc=f"{backend:7s}", unit="epoch", leave=False):
        total += _run_epoch(loader, args.device, transfer)
    elapsed = time.perf_counter() - t0

    n_batches = -(-len(dataset) // args.batch) * args.epochs
    return {
        "backend": backend,
        "img_s": total / elapsed,
        "ms_batch": elapsed / n_batches * 1e3,
        "mb": loaders.store_bytes(backend, args.demos) / 1e6,
        "prep_s": prep_s,
        "n": len(dataset),
    }


def _jpeg_pipe(scene_files, wrist_files, args):
    """DALI: read paired JPEGs, decode both on the GPU (nvJPEG, device='mixed')."""
    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def
    def pipe():
        s, _ = fn.readers.file(
            files=scene_files, random_shuffle=True, seed=args.seed, name="scene"
        )
        w, _ = fn.readers.file(
            files=wrist_files, random_shuffle=True, seed=args.seed, name="wrist"
        )
        dec = dict(device="mixed", output_type=types.RGB)
        return fn.decoders.image(s, **dec), fn.decoders.image(w, **dec)

    return pipe(
        batch_size=args.batch, num_threads=max(2, args.num_workers), device_id=0
    )


def _video_pipe(scene_mp4s, wrist_mp4s, args):
    """DALI: NVDEC-decode one random frame per sample straight from the mp4s."""
    from nvidia.dali import fn, pipeline_def

    @pipeline_def
    def pipe():
        rd = dict(device="gpu", sequence_length=1, random_shuffle=True, seed=args.seed)
        # sequence_length=1 => one frame per sample; output is (batch, 1, H, W, C).
        return (
            fn.readers.video(filenames=scene_mp4s, name="scene", **rd),
            fn.readers.video(filenames=wrist_mp4s, name="wrist", **rd),
        )

    return pipe(
        batch_size=args.batch, num_threads=max(2, args.num_workers), device_id=0
    )


def _time_dali(pipe, n: int, args: Args) -> tuple[float, float]:
    """Warm up, then time ``args.epochs`` shuffled passes. Returns (img/s, ms/batch)."""
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    it = DALIGenericIterator(
        [pipe],
        ["scene", "wrist"],
        reader_name="scene",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )
    for batch in it:  # warmup (pipeline build, index, prime caches)
        batch[0]["scene"]
    it.reset()

    t0 = time.perf_counter()
    total = 0
    for _ in tqdm(range(args.epochs), desc="dali   ", unit="epoch", leave=False):
        for batch in it:
            total += batch[0]["scene"].shape[0]
        it.reset()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_batches = -(-n // args.batch) * args.epochs
    return total / elapsed, elapsed / n_batches * 1e3


def _dali_unavailable(backend: str) -> dict | None:
    try:
        import nvidia.dali  # noqa: F401
    except ImportError:
        return {"backend": backend, "img_s": None, "skip": "nvidia-dali not installed"}
    if not torch.cuda.is_available():
        return {"backend": backend, "img_s": None, "skip": "no CUDA device"}
    return None


def _benchmark_dali(args: Args) -> dict:
    """nvJPEG GPU decode of the per-frame JPEG store (same files as the jpeg backend)."""
    if skip := _dali_unavailable("dali"):
        return skip
    t0 = time.perf_counter()
    dataset = loaders.build_dataset(
        "jpeg", args.demos, chunk=args.chunk
    )  # ensure store
    prep_s = time.perf_counter() - t0
    scene_files = [
        str(p)
        for ep in dataset.episodes
        for p in sorted((ep / "scene_camera").glob("*.jpg"))
    ]
    wrist_files = [p.replace("scene_camera", "wrist_camera") for p in scene_files]

    img_s, ms_batch = _time_dali(
        _jpeg_pipe(scene_files, wrist_files, args), len(dataset), args
    )
    return {
        "backend": "dali",
        "img_s": img_s,
        "ms_batch": ms_batch,
        "mb": loaders.store_bytes("jpeg", args.demos) / 1e6,
        "prep_s": prep_s,
        "n": len(dataset),
    }


def _benchmark_dali_video(args: Args) -> dict:
    """NVDEC GPU decode straight from the source mp4s — no transcode, tiny on-disk."""
    if skip := _dali_unavailable("dali_video"):
        return skip
    episodes = episode_io.list_episodes(args.demos)
    scene_mp4s = [str(ep / "scene_camera.mp4") for ep in episodes]
    wrist_mp4s = [str(ep / "wrist_camera.mp4") for ep in episodes]
    n = len(
        loaders.build_dataset("naive", args.demos, chunk=args.chunk)
    )  # cheap: actions only

    img_s, ms_batch = _time_dali(_video_pipe(scene_mp4s, wrist_mp4s, args), n, args)
    return {
        "backend": "dali_video",
        "img_s": img_s,
        "ms_batch": ms_batch,
        "mb": loaders.store_bytes("naive", args.demos) / 1e6,
        "prep_s": 0.0,
        "n": n,
    }


_DISPATCH = {"dali": _benchmark_dali, "dali_video": _benchmark_dali_video}


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    results = [
        _DISPATCH.get(b, lambda a, b=b: _benchmark(b, a))(args) for b in args.backends
    ]

    base = next(
        (r["img_s"] for r in results if r["backend"] == "naive" and r["img_s"]), None
    )
    n = next((r["n"] for r in results if r.get("n")), "?")
    print(
        f"\nbackend    |  img/s  | ms/batch | disk MB | prep s | vs naive   "
        f"(batch={args.batch}, workers={args.num_workers}, "
        f"device={args.device}, n={n})"
    )
    print("-" * 82)
    for r in results:
        if r["img_s"] is None:
            print(f"{r['backend']:10s} | (skipped: {r['skip']})")
            continue
        speedup = f"{r['img_s'] / base:6.1f}x" if base else "      -"
        note = "  [GPU]" if r["backend"].startswith("dali") else ""
        print(
            f"{r['backend']:10s} | {r['img_s']:7.0f} | {r['ms_batch']:8.1f} | "
            f"{r['mb']:7.1f} | {r['prep_s']:6.2f} | {speedup}{note}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
