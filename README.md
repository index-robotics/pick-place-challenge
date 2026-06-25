# pick-place-challenge — data loading

![A Franka arm taking random actions in the scene, camera orbiting](docs/showcase.gif)

> Random actions, camera orbiting — `uv run python scripts/showcase.py`.

A small, self-contained robotics sandbox: a **Franka Panda + Robotiq 2F-85** that
must **pick up a ball and place it in a bowl** on a table, inside a real modeled
room, built on [mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo,
Isaac-Lab-style manager API).

## The challenge

This repo ships a complete **collect → train → eval** pipeline for the pick-and-place
task, including **image-based behavior cloning**: a scripted expert collects demos
(saving scene + wrist camera video to mp4), a small CNN clones the task from pixels,
and the policy is rolled out and scored. See
[docs/imitation_learning.md](docs/imitation_learning.md) for the full walkthrough.

The image dataloader is **deliberately naive**: `DemoImageDataset`
(`src/pick_place_challenge/image_dataset.py`) re-decodes an entire episode's mp4 on
*every single sample fetch*, so image training is badly bottlenecked on data loading.

**Your task: make image data loading as fast as you can — ideally close to the
theoretical maximum for feeding this training loop — and build a benchmark that proves
it.** Concretely:

1. Get the image pipeline running: collect a dataset, train an image policy.
2. Profile the data loading — find where the time actually goes.
3. Build a **benchmarking mechanism** that measures loader throughput (images/sec) and
   lets you compare approaches in one table you can walk us through: the naive baseline
   at one end, your best at the other, and ideally an estimate of the *ceiling* (how
   fast loading could ever be) to measure against.
4. Push throughput as close to that ceiling as you can.

Assume the dataset is **too large to simply preload into memory** — your approach
should still hold when the data doesn't fit in RAM/VRAM.

**Stretch (if we get there):** once data loading is no longer the bottleneck, speed up
the rest of the training loop as much as possible.

There's no single right answer. We care about how you profile, the tradeoffs you reason
through, and that the benchmark backs up your claims.

## Quickstart (~60 seconds)

```bash
uv sync                                   # install everything (locked)
uv run python scripts/view_scene.py       # look at the robot + table (CPU, no GPU needed)
```

Assets are fetched on first use into `~/.cache` (the room mesh is ~tens of MB, so the
first launch takes a moment). Pre-fetch everything with `uv run pick-place-fetch-assets`.
A GPU is recommended for `collect_demos` / `eval_policy` and for image training. List the
registered tasks with `uv run pick-place-envs`.
