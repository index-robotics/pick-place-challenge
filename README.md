# pick-place-challenge

![A Franka arm taking random actions in the scene, camera orbiting](docs/showcase.gif)

> Random actions, camera orbiting — `uv run python scripts/showcase.py`.

A small, self-contained robotics sandbox: a **Franka Panda + Robotiq 2F-85** that
must **pick up a ball and place it in a bowl** on a table, inside a real modeled
room, built on [mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo,
Isaac-Lab-style manager API).

## The problem

This repository is set up for **joint-space robot control**: a complete
**data collection → training → evaluation** pipeline for the pick-and-place task,
with the arm driven entirely in joint action space. A scripted expert collects
demonstrations, a small MLP clones them, and the trained policy is rolled out and
scored by success rate (ball in the bowl).

**Your task:** add **OSC (operational-space) end-effector pose control** to the
repository as a **secondary action mode**, and run the same
data-collection → training → evaluation pipeline with it. The end state: the env can
be driven in either **joint** or **OSC** mode, and you can collect demos, train a
policy, and evaluate it in each.

To get oriented, read these two docs:

- **[docs/env.md](docs/env.md)** — the environment: the task, the two observation
  variants, the joint action space, the reward, how to drive it, and how the repo
  is built.
- **[docs/imitation_learning.md](docs/imitation_learning.md)** — the joint-space
  **collect → train → eval** pipeline (scripted expert, BC MLP, on-disk layout)
  that you'll be extending to OSC.

## Quickstart (~60 seconds)

```bash
uv sync                                   # install everything (locked)
uv run python scripts/view_scene.py       # look at the robot + table (CPU, no GPU needed)
```

Assets are fetched on first use into `~/.cache` (the room mesh is ~tens of MB, so
the first launch takes a moment). Pre-fetch everything with
`uv run pick-place-fetch-assets`. For driving the env and the full command list, see
[docs/env.md](docs/env.md).
