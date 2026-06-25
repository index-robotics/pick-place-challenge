# The environment

A **Franka Panda + Robotiq 2F-85** that must pick up a ball and place it in a bowl
on a table, inside a real modeled room (an Objaverse parking garage), built on
[mjlab](https://github.com/mujocolab/mjlab) (GPU-accelerated MuJoCo, Isaac-Lab-style
manager API). The ball (a Poly Haven mesh), bowl (a scanned object whose cavity
actually holds the ball), and room are real assets, not primitives.

Assets — the [Poly Haven](https://polyhaven.com) ball mesh, the scanned bowl from
[mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects), and
the [Objaverse](https://objaverse.allenai.org) room mesh — are fetched on first use
into `~/.cache` (the room is ~tens of MB, so first launch takes a moment). Pre-fetch
everything with `uv run pick-place-fetch-assets`.

## The two environments

Same task (place the ball in the bowl), same robot, same reward — they differ
**only in the observation**:

| Task ID | Observation |
|---|---|
| `Mjlab-PlaceBall-Franka-State-v0` | Low-dim **state**: joint pos/vel, end-effector→ball vector, ball→bowl vector, bowl position, last action. |
| `Mjlab-PlaceBall-Franka-Pixels-v0` | **Pixels**: an `84×84` RGB **wrist** camera + `84×84` RGB **scene** camera, plus proprioception and the (fixed) bowl position. The ball is *not* given as state — you have to see it. |

**Action** (both): 8-D — 7 arm joint-position targets + 1 gripper command
(`-1` open … `+1` closed; the Robotiq's tendon does the rest).

**Reward** (both): a simple reach-then-place shaping (reach the ball × bring it to
the bowl). It's deliberately basic — **rip it out and write your own** if doing RL.
Success = ball resting inside the bowl rim.

## Driving the env

List the tasks this repo registers:

```bash
uv run pick-place-envs
```

Drive it with a random policy and watch it in an interactive viewer:

```bash
uv run play Mjlab-PlaceBall-Franka-State-v0 --agent random --viewer native
# headless / over SSH? use:  --viewer viser   (opens a browser viewer)
```

Switch tasks by swapping the ID:

```bash
uv run play Mjlab-PlaceBall-Franka-Pixels-v0 --agent random --viewer native
```

A minimal, readable env-loop example to copy from:

```bash
uv run python scripts/random_rollout.py --task state --steps 200
```

## How it's built

Just two modules — `scene.py` (the physical world) and `task.py` (the task):

- **`src/pick_place_challenge/scene.py`** — everything physical, in one file:
  - *Asset fetching* — the Franka + Robotiq MJCFs (MuJoCo Menagerie via
    `robot_descriptions`), the ball mesh + wood texture (Poly Haven, CC0), the
    bowl (a Google Scanned Object), and the room (Objaverse, CC-BY). Nothing
    vendored; all fetched to `~/.cache`. Swap the room via `ROOM_UID`, the wood
    via `WOOD_ID`, the bowl via `BOWL_NAME`.
  - *Robot* — `build_franka_robotiq_spec` / `franka_robotiq_cfg` (the 2F-85's
    full mimic linkage is kept; it simulates correctly in MuJoCo-Warp).
  - *Objects* — `ball_spec` (real mesh + sphere collider) and `bowl_spec` (its
    convex decomposition keeps the cavity, so the ball nests inside).
  - *Room + table* — `add_studio`: a real textured room mesh + a wood table.
    All visual-only meshes (so they render in Viser too); the table top at
    `z=0` is the only collider.
- **`src/pick_place_challenge/task.py`** — where the env and its single MDP are
  wired together: the reach-and-place reward / observations / success check, the
  env configs (`state_env_cfg`, `pixels_env_cfg`, the BC-wired `build_bc_env_cfg`),
  and task registration.
- **`src/pick_place_challenge/expert.py`** — the scripted waypoint pick-and-place
  expert that generates demos, and **`kinematics.py`** — the resolved-rate Jacobian
  IK it uses to solve Cartesian waypoints to joint targets.
- **`src/pick_place_challenge/bc.py`** — the tiny behavior-cloning MLP (+ train,
  normalization, save/load); **`episode_io.py`** (parquet/mp4 demo storage) and
  **`run_dir.py`** (per-run archiving) support it.
- `scripts/` — `view_scene.py` (pure-MuJoCo CPU viewer, no GPU needed),
  `random_rollout.py` (a readable env-loop example to copy from),
  `showcase.py` (renders the README GIF), and the imitation-learning trio
  `collect_demos.py` / `train_bc.py` / `eval_policy.py`.
- `tests/` — `test_scene_smoke.py` and `test_pipeline_smoke.py`; `uv run pytest`.

You're free to change anything in here.

## Hardware

- **CPU / macOS is fine** for viewing the scene, the random/scripted paths, BC
  inference, and small-scale env stepping — MuJoCo-Warp has a CPU backend.
- **An NVIDIA GPU** makes large-scale **RL training** practical (thousands of
  parallel envs). The `--agent trained` and big `--env.scene.num-envs` paths
  effectively want one. You can do a lot without it.

Scripts and tasks default to `cuda`; pass `--device cpu` to `random_rollout.py`,
or `--device cpu` style flags where mjlab exposes them, to stay on CPU.

## Development

Quality gates run via [prek](https://github.com/j178/prek) (a fast pre-commit):

```bash
uv run prek install          # install the git hook (once)
uv run prek run --all-files  # or run them all now
```

The hooks are `ruff format`, `ruff check`, `ty check` (type-checking), and
`deptry` (dependency hygiene) — or run any individually, e.g. `uv run ty check`,
`uv run pytest`. Zed users get this in-editor too: `.zed/settings.json` selects
`ty` + `ruff` (and disables Zed's default basedpyright) so the editor matches.

`ty` type-checks against MuJoCo via committed stubs in `.typings/` (MuJoCo ships
none). They're pinned to the locked MuJoCo; regenerate only if you bump it:

```bash
uv run pybind11-stubgen --ignore-all-errors mujoco -o .typings
```
