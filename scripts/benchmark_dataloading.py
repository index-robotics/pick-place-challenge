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

from pick_place_challenge import bc, loaders

ALL_BACKENDS = (*loaders.BACKENDS, "dali")  # dali is GPU-native, measured separately


@dataclass(frozen=True)
class Args:
    demos: str = "demos/joint"
    """Demo dir to benchmark against (its mp4s feed every backend)."""
    backends: tuple[str, ...] = ALL_BACKENDS
    """Which backends to compare (subset of naive/seek/jpeg/memmap/dali)."""
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


def _build_dali_iterator(scene_files, wrist_files, args):
    """A DALI pipeline that reads paired JPEGs and decodes both on the GPU (nvJPEG)."""
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    @pipeline_def
    def pipe():
        # Same seed on both readers keeps scene/wrist paired; one decode batch =
        # `batch` samples = 2*batch frames, matching the torch backends' unit.
        s, _ = fn.readers.file(
            files=scene_files, random_shuffle=True, seed=args.seed, name="scene"
        )
        w, _ = fn.readers.file(
            files=wrist_files, random_shuffle=True, seed=args.seed, name="wrist"
        )
        dec = dict(device="mixed", output_type=types.RGB)
        return fn.decoders.image(s, **dec), fn.decoders.image(w, **dec)

    p = pipe(
        batch_size=args.batch,
        num_threads=max(2, args.num_workers),
        device_id=0,
        seed=args.seed,
    )
    return DALIGenericIterator(
        [p],
        ["scene", "wrist"],
        reader_name="scene",
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )


def _benchmark_dali(args: Args) -> dict:
    try:
        import nvidia.dali  # noqa: F401
    except ImportError:
        return {"backend": "dali", "img_s": None, "skip": "nvidia-dali not installed"}
    if not torch.cuda.is_available():
        return {"backend": "dali", "img_s": None, "skip": "no CUDA device"}

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
    n = len(dataset)

    it = _build_dali_iterator(scene_files, wrist_files, args)
    for batch in it:  # warmup (JIT/build, prime caches)
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
    return {
        "backend": "dali",
        "img_s": total / elapsed,
        "ms_batch": elapsed / n_batches * 1e3,
        "mb": loaders.store_bytes("jpeg", args.demos) / 1e6,  # reads the jpeg store
        "prep_s": prep_s,
        "n": n,
    }


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    results = [
        _benchmark_dali(args) if b == "dali" else _benchmark(b, args)
        for b in args.backends
    ]

    base = next(
        (r["img_s"] for r in results if r["backend"] == "naive" and r["img_s"]), None
    )
    n = next((r["n"] for r in results if r.get("n")), "?")
    print(
        f"\nbackend |  img/s  | ms/batch | disk MB | prep s | vs naive   "
        f"(batch={args.batch}, workers={args.num_workers}, "
        f"device={args.device}, n={n})"
    )
    print("-" * 78)
    for r in results:
        if r["img_s"] is None:
            print(f"{r['backend']:7s} | (skipped: {r['skip']})")
            continue
        speedup = f"{r['img_s'] / base:6.1f}x" if base else "      -"
        note = "  [GPU]" if r["backend"] == "dali" else ""
        print(
            f"{r['backend']:7s} | {r['img_s']:7.0f} | {r['ms_batch']:8.1f} | "
            f"{r['mb']:7.1f} | {r['prep_s']:6.2f} | {speedup}{note}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
