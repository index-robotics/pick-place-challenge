"""Benchmark image-dataloader backends: how fast can we feed frames to training?

Iterates each backend through a ``DataLoader`` exactly as image BC would and reports
throughput (images/sec) plus per-batch latency and on-disk size, so the storage/speed
tradeoff is concrete rather than theoretical. The naive mp4 loader is the baseline the
challenge optimizes against; ``seek``/``jpeg``/``memmap`` are reference faster backends
(see :mod:`pick_place_challenge.loaders`).

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


@dataclass(frozen=True)
class Args:
    demos: str = "demos/joint"
    """Demo dir to benchmark against (its mp4s feed every backend)."""
    backends: tuple[str, ...] = loaders.BACKENDS
    """Which backends to compare (subset of naive/seek/jpeg/memmap)."""
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


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    results = [_benchmark(b, args) for b in args.backends]

    base = next((r["img_s"] for r in results if r["backend"] == "naive"), None)
    print(
        f"\nbackend |  img/s  | ms/batch | disk MB | prep s | vs naive   "
        f"(batch={args.batch}, workers={args.num_workers}, "
        f"device={args.device}, n={results[0]['n']})"
    )
    print("-" * 78)
    for r in results:
        speedup = f"{r['img_s'] / base:5.1f}x" if base else "    -"
        print(
            f"{r['backend']:7s} | {r['img_s']:7.0f} | {r['ms_batch']:8.1f} | "
            f"{r['mb']:7.1f} | {r['prep_s']:6.2f} | {speedup}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
