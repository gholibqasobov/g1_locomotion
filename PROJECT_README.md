# G1 Locomotion (Isaac Lab)

Velocity-tracking locomotion training for the **Unitree G1** humanoid, built as a standalone
Isaac Lab project (manager-based RL environment + RSL-RL PPO).

> This is the project-specific guide. The repo's original `README.md` is the generic Isaac Lab
> extension-template doc and is left unchanged.

- **Task ID:** `Template-G1-Locomotion-v0`
- **Robot:** Unitree G1 (full `g1.usd`, 37 DoF including arms + fingers)
- **Algorithm:** PPO (RSL-RL)
- **Terrain:** flat ground plane

---

## Table of contents

1. [Installation](#installation)
2. [How to run](#how-to-run)
3. [Project layout](#project-layout)
4. [Environment design](#environment-design)
5. [Changelog — fixes & tuning applied](#changelog--fixes--tuning-applied)
6. [Things you should know](#things-you-should-know)
7. [Troubleshooting](#troubleshooting)
8. [Tuning levers / next steps](#tuning-levers--next-steps)

---

## Installation

You need a working [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
install (conda/uv recommended). Then install this project in editable mode with the same Python
interpreter that has Isaac Lab:

```bash
# from the repo root
python -m pip install -e source/g1_locomotion
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not in your conda/venv
```

Verify the task is registered:

```bash
python scripts/list_envs.py   # should list Template-G1-Locomotion-v0
```

---

## How to run

### Train

```bash
python scripts/rsl_rl/train.py --task=Template-G1-Locomotion-v0 --headless
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--headless` | No GUI (fastest, use for real training runs) |
| `--num_envs 4096` | Override parallel env count (default 4096) |
| `--video --video_length 200` | Record rollout clips during training |
| `--max_iterations 3000` | Override training length |

Logs and checkpoints are written to `logs/rsl_rl/g1_locomotion_ppo/<timestamp>/`.

### Play / evaluate a checkpoint

```bash
python scripts/rsl_rl/play.py --task=Template-G1-Locomotion-v0 --num_envs 32
```

By default it loads the most recent checkpoint from the experiment directory.

> **Important:** after changing physics rate / rewards / reset logic (as we did — see changelog),
> **retrain from scratch.** Do not resume or play an old checkpoint: its weights were adapted to
> the previous (broken) dynamics and will look unstable under the new config.

---

## Project layout

```
source/g1_locomotion/g1_locomotion/
├── robots/
│   └── unitree_g1.py                 # G1_CFG: USD, init pose, actuators / PD gains
└── tasks/manager_based/g1_locomotion/
    ├── __init__.py                   # gym.register("Template-G1-Locomotion-v0")
    ├── g1_locomotion_env_cfg.py      # the environment config (scene/obs/rewards/events/terminations)
    ├── agents/
    │   └── rsl_rl_ppo_cfg.py         # PPO hyperparameters (PPORunnerCfg)
    └── mdp/
        ├── __init__.py               # re-exports isaaclab.envs.mdp + local rewards
        └── rewards.py                # task-specific reward functions
scripts/rsl_rl/{train.py, play.py}
```

---

## Environment design

**Scene (`G1LocomotionSceneCfg`)**
- Ground: `GroundPlaneCfg` (flat). No procedural terrain / height scanner.
- Robot: `G1_CFG` spawned at `{ENV_REGEX_NS}/Robot`.
- Sensor: `ContactSensorCfg` on all robot bodies (`history_length=3`, air-time tracking).

**Observations (policy group, 123-dim)** — base lin/ang velocity, projected gravity, velocity
command, joint pos/vel (relative), last action. All with observation noise (corruption enabled
for training).

**Actions** — `JointPositionActionCfg` on all joints, `scale=0.5`, `use_default_offset=True`
(actions are deltas around the default stance).

**Commands** — `UniformVelocityCommandCfg`: `lin_vel_x ∈ [0,1]`, `lin_vel_y ∈ [-0.5,0.5]`,
`ang_vel_z ∈ [-1,1]`, with heading control.

**Rewards** — velocity tracking (lin + ang, exp), feet air-time (biped), and penalties for
vertical velocity, angular velocity, torques, joint accel, action rate, foot slide, joint-limit
violation, flat-orientation, and joint deviations (hips/arms/fingers/torso).

**Sim rate** — `dt = 0.005` (200 Hz physics), `decimation = 4` → **50 Hz control**.

---

## Changelog — fixes & tuning applied

This section documents every change made while getting the environment to load **and** to train a
stable walking policy. The robot was previously collapsing on the first step. The root causes
fell into two buckets: **config-loading errors** (env wouldn't start) and **dynamics/learning
issues** (env started but the humanoid fell instantly).

### A. Config-loading fixes (env wouldn't even start)

| # | File / location | Problem | Fix |
|---|-----------------|---------|-----|
| A1 | `g1_locomotion_env_cfg.py` `__post_init__` | `self.sim.physical_material = self.scene.terrain.physics_material` referenced a `terrain` attribute that doesn't exist (scene uses a `ground` plane, not a `TerrainImporterCfg`). Also `physical_material` was a typo for `physics_material`. | Removed the line — `GroundPlaneCfg` already carries its own default physics material. |
| A2 | `EventCfg` startup terms | `physics_material`, `add_base_mass`, `base_com` had **no `func=`**, so config validation failed with *"Missing values … func"*. | Wired each to its standard MDP function: `randomize_rigid_body_material`, `randomize_rigid_body_mass`, `randomize_rigid_body_com`. |
| A3 | Body-name regex `"base"` | G1's root body is **`pelvis`**, not `base`. Every term using `body_names="base"` raised *"Not all regular expressions are matched"* (affected `add_base_mass`, `base_com`, `base_external_force_torque`, and the `base_contact` termination). | Replaced `"base"` → `"pelvis"`. |
| A4 | `base_contact` termination | `mdp.illegal_contact` requires **both** `sensor_cfg` and `threshold`; only `sensor_cfg` was supplied. | Added `"threshold": 1.0`. |
| A5 | `RewardsCfg.joint_torques_l2` | The term named `joint_torques_l2` was wired to `mdp.joint_pos_target_l2`, which requires mandatory `target` + `asset_cfg` params that weren't provided. | Pointed it at the correct `mdp.joint_torques_l2`. |
| A6 | `RewardsCfg` duplicates | Two near-duplicate reward pairs (`joint_torques_l2`/`dof_torques_l2`, `joint_acc_l2`/`dof_acc_l2`) double-counted the same penalty. | Commented out one of each pair (kept `dof_torques_l2` and `joint_acc_l2`). |

### B. Stability / learning fixes (env started, but robot fell immediately)

Benchmarked against Isaac Lab's official `G1FlatEnvCfg` / `G1RoughEnvCfg`.

| # | What | Before | After | Why |
|---|------|--------|-------|-----|
| B1 | **Reset base velocity** (`events.reset_base`) | random ±0.5 on x/y/z **and roll/pitch/yaw** | **all zeros** | Spawning a humanoid with random body (esp. angular) velocity means it starts mid-tip and falls before it can act. Reference spawns at rest. **Biggest single cause.** |
| B2 | **Reset joint scaling** (`events.reset_robot_joints`) | `position_range=(0.5, 1.5)` | `(1.0, 1.0)` | `(0.5,1.5)` randomly scales the default stance → legs spawn half-collapsed. `(1.0,1.0)` = exact default crouch, matching the reference. |
| B3 | **Fall termination** (`terminations`) | only `pelvis` contact | `pelvis` **or** `torso_link` contact **+ new `bad_orientation` term** | Previously, falling onto **arms/head/knees/torso did not terminate** — the robot "lived" through the whole episode in a fallen state, poisoning the learning signal. `bad_orientation` (>~57° tilt) catches every fall mode regardless of contact. |
| B4 | **Upright reward** (`rewards`) | *missing* | added `flat_orientation_l2` weight `-1.0` | There was no reward keeping the torso level. |
| B5 | **Torque penalty** (`rewards.dof_torques_l2`) | `-2.0e-5` | `-2.0e-6` | 10× too harsh — discouraged the robot from exerting enough torque to hold itself up. Matches flat reference. |
| B6 | **Sim rate** (`__post_init__`) | `dt=1/120`, `decimation=2` (60 Hz control) | `dt=0.005`, `decimation=4` (50 Hz control, 200 Hz physics) | G1's stiff PD gains (stiffness 150–200) and all reference reward weights are tuned for 200 Hz physics / 50 Hz control. |
| B7 | **`undesired_contacts` reward** | enabled, `-1.0` on hip/knee links | disabled (commented) | Official G1 config disables this. G1's hip/knee collision meshes register spurious contacts during normal gait, so the penalty fights against walking. |
| B8 | **PPO entropy** (`rsl_rl_ppo_cfg.py`) | `entropy_coef=0.005` | `0.008` | Slightly more exploration for from-scratch locomotion (reference value). |

---

## Things you should know

- **Retrain from scratch** after the changes above (see the note under [How to run](#how-to-run)).
- **Robot is the full 37-DoF G1** (`g1.usd`) — it includes 2 arms and 14 finger joints, all in the
  action space. The `joint_deviation_*` rewards pin arms/fingers near their default pose so they
  don't interfere with locomotion. This works but is heavier than necessary. The official Isaac Lab
  G1 task uses `g1_minimal.usd` (same joints, lighter collision meshes → faster sim). If you want
  faster training, switch `usd_path` in `robots/unitree_g1.py` to `.../G1/g1_minimal.usd`.
- **`push_robot` is still enabled** (interval 10–15 s). The official G1 config disables it. It only
  fires after several seconds so it won't break early learning, but if you still see instability,
  set `self.events.push_robot = None` for the first run, then re-enable it later for robustness.
- **Expected training signal:** episode length should climb steadily within the first few hundred
  PPO iterations. If it flatlines near the minimum, the task is still too hard — see tuning levers.
- **The duplicate / disabled reward terms are left in the file as comments**, so you can re-enable
  or re-tune them deliberately rather than rediscovering them.

---

## Troubleshooting

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| `AttributeError: 'G1LocomotionSceneCfg' object has no attribute 'terrain'` | A `terrain`-based line copied from the rough-terrain template, but this scene uses a flat `ground` plane | `__post_init__` (fix A1) |
| `TypeError: Missing values detected … events.*.func` | An `EventTerm` without a `func=` | `EventCfg` (fix A2) |
| `ValueError: Not all regular expressions are matched … base: []` | A `body_names` regex doesn't match the G1 USD (e.g. `base` vs `pelvis`, or `torso_joint`, elbow/finger joint names) | Check the printed *"Available strings"* list against your `SceneEntityCfg` patterns |
| `ValueError: term '…' expects mandatory parameters […]` | A reward/termination `func` is missing required params, or is the wrong function for the term name | `RewardsCfg` / `TerminationsCfg` |
| Robot falls instantly when playing | Wrong reset randomization, missing orientation reward/termination, or mismatched sim rate | Changelog section B |

**Tip:** the *"Available strings"* dump in a regex error lists every valid body/joint name for the
loaded USD — always match your patterns against that, not against assumptions.

---

## Tuning levers / next steps

If the policy still struggles after the fixes:

1. **Make the task easier first.** Narrow the command range, e.g.
   `commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)` and `lin_vel_y = (0.0, 0.0)`, train to a
   stable walk, then widen the ranges.
2. **Add a command curriculum** so velocity ranges grow with competence.
3. **Disable pushing** for the first run (`self.events.push_robot = None`), then re-enable.
4. **Switch to `g1_minimal.usd`** for faster iteration.
5. **Scale up the network** if needed: the rough reference uses `[512, 256, 128]` for actor/critic;
   this project uses `[256, 128, 128]` (fine for flat ground).
6. **Watch these metrics** in the logs: mean episode length (should rise), `track_lin_vel_xy_exp` /
   `track_ang_vel_z_exp` rewards (should rise), and the termination rate (should fall).

### Reference

Compare against the upstream configs that this project mirrors:
`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/`
(`flat_env_cfg.py`, `rough_env_cfg.py`) and the shared base `velocity_env_cfg.py`.

---

# Stair climbing (teacher–student distillation)

A second, separate project that extends the flat policy to **stairs / rough terrain** using a
head-mounted **Intel RealSense D435** depth camera as the deployable sensor. The flat task above is
**untouched** — all of this lives in new files and new task IDs.

## Approach (why it's built this way)

Research consensus (2024–2026) for perceptive humanoid locomotion: a privileged **height-scanner
teacher** is fast and stable to train, but a height map is not something the real robot has. An
**egocentric depth camera** *is* deployable and has the smaller sim-to-real modality gap — so we train
the teacher on the height scanner, then **distill** it into a student that sees only the depth camera.
This is the standard teacher–student recipe and is supported natively by RSL-RL.

- **Teacher** — `Template-G1-Stairs-Teacher-v0`: observes a top-down **height scanner** (privileged).
  Trained with PPO (`OnPolicyRunner`).
- **Student** — `Template-G1-Stairs-Student-v0`: the policy observes only the **RealSense D435 depth
  image**; a second `teacher` observation group reproduces the height-scan input so the pretrained
  teacher network loads 1:1. Trained with `DistillationRunner`.

> Teacher–student is inherently **two runs** (the student imitates a teacher that must already exist).
> It is not one script — that's a property of distillation, not a limitation here.

## Sensors

| | Height scanner (teacher) | RealSense D435 depth (student) |
|---|---|---|
| Type | `RayCasterCfg` + `GridPatternCfg` | `RayCasterCameraCfg` + `PinholeCameraPatternCfg` |
| Mount | `torso_link`, top-down | `head_link`, pitched **42.4° down** |
| Output | 17×11 = **187** height samples | **64×36** depth image → flattened to **2304** |
| FOV | grid 1.6×1.0 m | ~87°×58° (matches D435 depth), range clipped to 5 m |
| Realism | privileged (not a real sensor) | egocentric depth + light noise/clip randomization |

The depth image is **downsampled and flattened into the MLP** (no CNN) — required because RSL-RL's
`StudentTeacher` module is MLP-only (1-D observations). The camera offset/intrinsics are sensible
defaults; **verify and tune them with the debug visualizer** (below).

## How to run

```bash
# 1) Train the teacher (height scanner + PPO)
python scripts/rsl_rl/train.py --task=Template-G1-Stairs-Teacher-v0 --headless
#    -> logs/rsl_rl/g1_stairs_teacher/<timestamp>/

# 2) Distill the student (depth camera), loading the teacher checkpoint
python scripts/rsl_rl/train.py --task=Template-G1-Stairs-Student-v0 \
       --load_run <teacher_timestamp> --headless
#    train.py auto-loads the teacher weights when the runner is a Distillation runner

# Evaluate
python scripts/rsl_rl/play.py --task=Template-G1-Stairs-Teacher-Play-v0 --num_envs 32
python scripts/rsl_rl/play.py --task=Template-G1-Stairs-Student-Play-v0 --num_envs 32
```

## Debug visualizer (check sensor placement)

```bash
# interactive viewer — shows the height-scan hit points + the camera frustum on the head
python scripts/debug/visualize_sensors.py --num_envs 4

# also dump a depth snapshot from env 0 (writes logs/debug/depth_env0.npy + .png)
python scripts/debug/visualize_sensors.py --num_envs 4 --save_depth --enable_cameras

# inspect the student task instead
python scripts/debug/visualize_sensors.py --task Template-G1-Stairs-Student-v0
```

It enables `debug_vis` on the terrain, height scanner, and camera, runs the robot idle, and prints
live sensor shapes. **Tip:** run with `PYTHONUNBUFFERED=1` (or watch the viewer) — when stdout is
redirected to a file, the Python prints are block-buffered and can be lost on shutdown while the
C++ Kit logs still appear.

To re-aim the camera, edit the constants at the top of
`tasks/manager_based/g1_locomotion/g1_stairs_env_cfg.py`:
`_CAM_DOWN_PITCH_QUAT` (down-pitch quaternion), the `camera.offset.pos` (mount point on the head),
and `_CAM_WIDTH/_CAM_HEIGHT` + aperture (resolution / FOV).

## New files (flat task untouched)

| File | Purpose |
|---|---|
| `tasks/manager_based/g1_locomotion/stairs_terrains.py` | `G1_STAIRS_TERRAINS_CFG` — stairs-weighted, curriculum-enabled terrain (level 0 ≈ flat → tall steps) |
| `tasks/manager_based/g1_locomotion/g1_stairs_env_cfg.py` | Scene (terrain + height scanner + D435 camera), obs groups, curriculum, teacher/student env cfgs (+ PLAY) |
| `tasks/manager_based/g1_locomotion/agents/rsl_rl_stairs_cfg.py` | `G1StairTeacherPPORunnerCfg` + `G1StairDistillRunnerCfg` |
| `scripts/debug/visualize_sensors.py` | Sensor-placement debug visualizer |
| `__init__.py` (edited) | Registers the 4 new stairs task IDs |

## Curriculum: flat → harder

`G1_STAIRS_TERRAINS_CFG` has `curriculum=True` with `num_rows` difficulty levels; step heights ramp
from ~2 cm (row 0 ≈ flat) to ~18 cm (last row). The scene sets `max_init_terrain_level=0` so every
robot **starts on the flat level**, and `mdp.terrain_levels_vel` promotes a robot to harder stairs
only once it travels far enough (and demotes it if it fails). So training literally starts flat and
gets harder as competence grows.

## Validated

Both envs were smoke-tested (`scripts/debug/visualize_sensors.py`, headless). Confirmed:
- Teacher `policy` obs = **310** (proprio 123 + height_scan 187).
- Student `policy` obs = **2427** (proprio 123 + depth 2304); student `teacher` obs = **310**
  (identical to the teacher's policy → distillation weight-load matches).
- Depth camera raycasts the terrain (range ~0.7–4.5 m); height scanner returns 187 points.
- No observation-concatenation errors; reset + stepping clean.

## Tuning / gotchas

- **Run the teacher first.** The student run **requires** `--load_run <teacher>`; `DistillationRunner`
  raises "Teacher model parameters not loaded" otherwise.
- **Teacher/student net sizes are linked.** `teacher_hidden_dims` in the distillation cfg **must equal**
  the teacher PPO `actor_hidden_dims` (`[512, 256, 128]`) — the loader maps the PPO `actor.*` weights
  into the teacher network. Change both together.
- **Warp note:** the height scanner and depth camera are Warp ray-casters (unlike the flat task, which
  used only PhysX contact sensors). The `cuDeviceGetUuid` lines Isaac Sim prints are benign warnings —
  Warp ray-casting works (verified on the RTX 5080 / CUDA 12.8 setup here).
- **Depth realism is minimal for now** (range clip + additive noise). To close the sim-to-real gap
  further, add pixel dropout / edge holes / latency to the depth term — the "realistic depth synthesis"
  the literature emphasizes.
- **Reference:** mirrors Isaac Lab's `…/locomotion/velocity/config/g1/rough_env_cfg.py` (height scanner,
  terrain curriculum) plus the native RSL-RL distillation API in
  `isaaclab_rl/rsl_rl/distillation_cfg.py`.
