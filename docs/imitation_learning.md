# Imitation learning: collect → train → eval

This walks through the full behavior-cloning pipeline — **collect demos → train →
eval** — and where everything lands on disk. A scripted expert generates the demos,
a small MLP clones them, and we score the policy by success rate (ball placed in
the bowl).

The arm is commanded with 7 joint-position targets plus a 1-D gripper command
(action dim 8). The expert plans Cartesian waypoints and solves each to joint
targets with one resolved-rate IK step, recording the raw action the env expects
(`q_target - home_q`, + gripper), so demos and the cloned policy speak the same
units.

## Prerequisites

```bash
uv sync
```

A GPU is recommended for `collect_demos` and `eval_policy` (they step the mjlab env;
collection also renders the demo videos). Training is tiny and runs fine on CPU.
Every script accepts `--device {cuda,cpu}`.

## TL;DR — run the whole pipeline

```bash
uv run python scripts/collect_demos.py --num-demos 30 --device cuda
uv run python scripts/train_bc.py      --epochs 300  --device cuda
uv run python scripts/eval_policy.py   --episodes 50 --device cuda
```

That prints something like:

```
success ~50% over 50 episodes  (mean reward ~-20)
```

(Exact numbers vary by seed and demo count — see [Reproducibility](#reproducibility).)

## Step by step

### 1. Collect demonstrations

```bash
uv run python scripts/collect_demos.py --num-demos 30 --device cuda
```

Runs the scripted expert across `--num-demos` parallel envs (one demo per env, each
with a different random ball spawn) and saves every **successful** episode. Useful
flags:

- `--num-demos N` — number of parallel envs (= max demos kept). Yield is ~80–95%,
  so ~30 envs gives ~26 demos; raise it for more.
- `--max-steps 450` — per-episode step budget for the expert.
- `--out demos` — output dir (cleared first so counts are exact).
- `--seed 0`, `--device cuda`.

### 2. Train the BC policy

```bash
uv run python scripts/train_bc.py --demos demos --epochs 300 --device cuda
```

Loads the demos, trains the small MLP with **action chunking** (predicts the next
`--chunk` actions, default 16), and saves the policy. Useful flags:

- `--demos demos` — demo directory to train on.
- `--out policies/bc.pt` — output checkpoint.
- `--epochs 300` (default 200), `--hidden 256`, `--batch 256`, `--lr 1e-3`.
- `--chunk 16` — actions predicted per inference, executed open-loop at rollout.
  Chunking is what makes the grasp work; `--chunk 1` (single-step) tends to fail.
- `--seed 0`, `--device cpu` (training default; CPU is fine).

### 3. Evaluate

```bash
uv run python scripts/eval_policy.py --episodes 50 --device cuda
```

Rolls the policy out over `--episodes` parallel envs and reports success rate
(`placed_in_bowl`) and mean reward. Flags: `--policy` (default `policies/bc.pt`),
`--episodes 50`, `--max-steps 300`, `--seed 0`, `--device cuda`.

## Where the outputs go

```
demos/
  meta.json
  episode_000/
    metadata.json
    observations.parquet  actions.parquet
    eef_states.parquet    gripper_states.parquet
    scene_camera.mp4      wrist_camera.mp4
  episode_001/ ...

policies/bc.pt                 # "latest" checkpoint eval defaults to

exp_local/<YYYY.MM.DD>/         # per-run archive (mechacarpal layout)
  <HHMMSS>_train/  policy.pt + config.json
  <HHMMSS>_eval/   config.json + metrics.json
```

Demos are step-aligned parquet streams plus mp4 videos (see
`src/pick_place_challenge/episode_io.py`); training reads only `observations` +
`actions` (the videos are for inspection). `demos/`, `policies/`, and `exp_local/`
are git-ignored.

## Reproducibility

All scripts take `--seed` (default `0`), which pins the ball spawns, weight init,
and minibatch shuffling, so a run repeats. Caveat: mujoco_warp GPU physics isn't
fully deterministic yet, so the spawns and success *rate* repeat but the exact
reward can drift slightly (CPU is tighter). For a trustworthy number, sweep a few
seeds and average:

```bash
for s in 0 1 2; do uv run python scripts/eval_policy.py --episodes 50 --seed $s; done
```

## Pushing it further

The baseline is deliberately simple. Some directions to raise the success rate:
more demos, observation history, receding-horizon (closed-loop) execution instead
of open-loop chunks, or a stronger policy. Diagnosing *why* the policy fails (watch
the saved `scene_camera.mp4` / `wrist_camera.mp4`) is half the work.
