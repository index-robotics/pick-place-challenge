# Imitation learning: joint vs OSC control

This walks through the full behavior-cloning pipeline — **collect demos → train →
eval → compare** — for the two arm control modes, and where everything lands on
disk. A scripted expert generates the demos, a small MLP clones them, and we score
the policy by success rate (ball placed in the bowl).

The only thing that differs between the two runs is the **control mode** (how the
arm is commanded); the task, expert plan, reward, and policy architecture are
identical, so the comparison isolates the effect of the action space.

| Mode | Arm action | Total dim |
|---|---|---|
| `joint` | 7 joint-position targets | 8 (+ gripper) |
| `osc` | 6-D end-effector pose delta, tracked by a resolved-rate Jacobian controller | 7 (+ gripper) |

## Prerequisites

```bash
uv sync
```

A GPU is recommended for `collect_demos` and `eval_policy` (they step the mjlab env;
collection also renders the demo videos). Training is tiny and runs fine on CPU.
Every script accepts `--device {cuda,cpu}`.

## TL;DR — run the whole comparison

```bash
for c in joint osc; do
  uv run python scripts/collect_demos.py --control $c --num-demos 20 --device cuda
  uv run python scripts/train_bc.py      --demos demos/$c --epochs 300 --device cuda
done
uv run python scripts/compare.py --episodes 50 --device cuda
```

That prints a table like:

```
 control |  success | mean reward
---------------------------------
   joint |     ~55% |     ~-120
     osc |     ~25% |     ~-140
```

(Exact numbers vary by seed and demo count — see [Reproducibility](#reproducibility)
and [Interpreting the result](#interpreting-the-result).)

## Step by step

### 1. Collect demonstrations

```bash
uv run python scripts/collect_demos.py --control joint --num-demos 20 --device cuda
uv run python scripts/collect_demos.py --control osc   --num-demos 20 --device cuda
```

Runs the scripted expert across `--num-demos` parallel envs (one demo per env, each
with a different random ball spawn) and saves every **successful** episode. Useful
flags:

- `--control {joint,osc}` — the action space demos are recorded in.
- `--num-demos N` — number of parallel envs (= max demos kept). Yield is ~80–95%,
  so ~20 envs gives ~16–19 demos; raise it for more.
- `--max-steps 450` — per-episode step budget for the expert.
- `--out demos/<control>` — output dir (cleared first so counts are exact).
- `--seed 0`, `--device cuda`.

### 2. Train the BC policy

```bash
uv run python scripts/train_bc.py --demos demos/joint --epochs 300 --device cuda
uv run python scripts/train_bc.py --demos demos/osc   --epochs 300 --device cuda
```

Loads the demos, trains the small MLP with **action chunking** (predicts the next
`--chunk` actions, default 16), and saves the policy. Useful flags:

- `--demos demos/<control>` — demo directory to train on.
- `--out policies/<control>.pt` — output checkpoint (defaults from the demo's mode).
- `--epochs 300` (default 200), `--hidden 256`, `--batch 256`, `--lr 1e-3`.
- `--chunk 16` — actions predicted per inference, executed open-loop at rollout.
  Chunking is what makes the grasp work; `--chunk 1` (single-step) tends to fail.
- `--seed 0`, `--device cpu` (training default; CPU is fine).

### 3. Evaluate

```bash
uv run python scripts/eval_policy.py --control joint --episodes 50 --device cuda
uv run python scripts/eval_policy.py --control osc   --episodes 50 --device cuda
```

Rolls the policy out over `--episodes` parallel envs and reports success rate
(`placed_in_bowl`) and mean reward. Flags: `--policy` (default
`policies/<control>.pt`), `--episodes 50`, `--max-steps 300`, `--seed 0`,
`--device cuda`.

### 4. Compare both modes

```bash
uv run python scripts/compare.py --episodes 50 --device cuda
```

Evaluates `policies/joint.pt` and `policies/osc.pt` on the **same** ball spawns
(shared seed) and prints the side-by-side table.

## Image-based BC (and the dataloading benchmark)

The same pipeline can train a policy **from camera images** instead of low-dim state.
Collection already renders scene + wrist RGB to mp4; image training reads those videos.

```bash
# Collect at a chosen resolution (must be a multiple of 16). Bigger = costlier to decode.
uv run python scripts/collect_demos.py --control joint --num-demos 20 --res 128 --device cuda

# Train the CNN policy from the mp4s (--obs image). Eval auto-detects the mode + res.
uv run python scripts/train_bc.py  --demos demos/joint --obs image --epochs 300 --device cuda
uv run python scripts/eval_policy.py --control joint --episodes 50 --device cuda
```

- `--res N` (collect) — square camera resolution, recorded in metadata so train/eval
  match. This is the knob the dataloading benchmark scales.
- `--obs {state,image}` (train) — `state` is the default low-dim MLP; `image` is the
  scene+wrist CNN (`ImagePolicy`). The policy file records its mode, so `eval_policy.py`
  / `compare.py` turn the cameras on automatically — no extra flag at eval.
- `--num-workers N` (train) — DataLoader workers; `0` (default) keeps the loader
  single-process.

> **Naive on purpose.** The image dataset (`src/pick_place_challenge/image_dataset.py`)
> **re-decodes an episode's entire mp4 on every sample fetch** — wildly redundant and
> slow. That is the baseline for the actual challenge: *make dataloading fast* (cache
> decoded clips, seek single frames, batch by episode, prefetch, a better on-disk
> format, …) without changing what the policy sees. Don't "fix" the redundant decode in
> the dataset itself — optimizing around it is the exercise.

### Dataloading benchmark

`scripts/benchmark_dataloading.py` feeds each loader backend through a `DataLoader`
exactly as image training would and reports throughput (img/s), per-batch latency, and
on-disk size — so the storage/speed tradeoff is concrete. Backends live in
`src/pick_place_challenge/loaders.py`; `jpeg`/`memmap` transcode once into a hidden
`demos/<c>/.loader_cache/<backend>/` store (self-contained, regenerable).

```bash
uv run python scripts/benchmark_dataloading.py --demos demos/joint                  # CPU, single-process
uv run python scripts/benchmark_dataloading.py --demos demos/joint --device cuda --num-workers 4
```

| backend | mechanism | disk |
|---|---|---|
| `naive` | re-decode the whole mp4 per sample (the baseline) | 1.0 MB |
| `seek` | cached ffmpeg reader, seek to one frame (no transcode) | 1.0 MB |
| `jpeg` | transcode → per-frame JPEG, one small CPU decode | 13.4 MB |
| `memmap` | transcode → raw uint8 `.npy`, `mmap` + slice (no decode) | 61.9 MB |
| `dali` | reads the JPEG store, decodes on **GPU** (nvJPEG via NVIDIA DALI) | 13.4 MB |
| `dali_video` | NVDEC-decodes one random frame straight from the mp4 on **GPU**, no transcode | 1.0 MB |

**CPU, single worker** (frames to host — the `dali*` rows are GPU-only, so N/A here):

| | naive | seek | jpeg | memmap |
|---|---|---|---|---|
| img/s | 13 | 23 | 3,162 | 50,431 |
| vs naive | 1× | 1.7× | 236× | 3762× |

Smarter *decoding* of the same mp4 (`seek`) barely helps (~1.7×) — you still pay a GOP
decode per random access. The big wins come from a frame-addressable **on-disk format**,
trading disk for speed: `jpeg` ≈236× at ~13× the bytes; `memmap` is decode-free, ~3762×
at ~62× the bytes (page-cache-bounded — no whole videos in RAM).

**`--device cuda`, 4 workers** (frames landed on the GPU — the training-relevant view):

| | naive | seek | jpeg | memmap | dali | dali_video |
|---|---|---|---|---|---|---|
| img/s @128² | 51 | 106 | 5,959 | 7,532 | **21,979** | 335 |
| vs naive | 1× | 2.1× | 116× | 147× | **428×** | 6.6× |
| img/s @512² | 10 | 58 | 331 | 356 | **4,152** | 170 |
| vs naive | 1× | 6.0× | 34× | 37× | **415×** | 17× |

Once every backend must get frames *onto* the GPU, the CPU loaders bottleneck on copying
raw uint8 over PCIe — `memmap`'s no-decode edge evaporates because it ships the raw frames
(62 MB at 128², ~1 GB at 512²). `dali` sends tiny *compressed* JPEG bytes across and
decodes on-GPU (nvJPEG), so it wins ~3× over the next best at 128² and **~12×** at 512²:
its lead widens with resolution because 16× the pixels punishes raw-byte transfer but
barely grows the compressed payload. `--num-workers N` scales the CPU rows ~linearly.

**`dali_video` (NVDEC straight from the mp4) is the cautionary row.** It needs no
transcode and keeps the tiny 1–6 MB h264 footprint, but it is *slow* (6–17× naive, far
below `dali`/`jpeg`) because the demos are encoded with a single keyframe per ~90-frame
clip — so NVDEC must decode ~45 frames on average to reach one random frame. GPU video
decode shines for *sequential/clip* reads, not the shuffled per-frame access BC wants. To
make this path fast you'd re-encode all-intra (`keyint=1`): cheap random seek at some disk
cost — the obvious next experiment.

DALI is GPU-only and optional — `uv sync --extra dali` to enable it (the `dali*` rows skip
automatically otherwise).

¹ 7 demos (n=629 transitions), batch 64, RTX 5090. Exact numbers vary by machine. Further
rung not benchmarked: seek-capable CPU decoders (PyAV/decord/torchcodec).

## Where the outputs go

```
demos/<control>/
  meta.json
  episode_000/
    metadata.json
    observations.parquet  actions.parquet
    eef_states.parquet    gripper_states.parquet
    scene_camera.mp4      wrist_camera.mp4
  episode_001/ ...

policies/<control>.pt          # "latest" checkpoint eval/compare default to

exp_local/<YYYY.MM.DD>/         # per-run archive (mechacarpal layout)
  <HHMMSS>_train_<control>/  policy.pt + config.json
  <HHMMSS>_eval_<control>/   config.json + metrics.json
```

Demos are step-aligned parquet streams plus mp4 videos (see
`src/pick_place_challenge/episode_io.py`); state training reads `observations` +
`actions`, while image training (`--obs image`) reads the mp4 videos + `actions`.
`demos/`, `policies/`, and `exp_local/` are git-ignored.

## Reproducibility

All scripts take `--seed` (default `0`), which pins the ball spawns, weight init,
and minibatch shuffling, so a run repeats. `compare.py` evaluates both modes on the
same spawns. Caveat: mujoco_warp GPU physics isn't fully deterministic yet, so the
spawns and success *rate* repeat but the exact reward can drift slightly (CPU is
tighter). For a trustworthy comparison, sweep a few seeds and average:

```bash
for s in 0 1 2; do uv run python scripts/compare.py --episodes 50 --seed $s; done
```

## Interpreting the result

`joint` often edges `osc` on raw success, but `osc` has a consistently less-negative
reward: open-loop joint chunks drift into joint limits (the `-10`-weighted
`joint_pos_limits` penalty), while the OSC controller keeps joint trajectories
feasible. Comparing and explaining that trade-off — and pushing either mode higher
(more demos, receding-horizon execution, observation history) — is the exercise.
