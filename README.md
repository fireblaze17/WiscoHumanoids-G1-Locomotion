# G1 Locomotion

A simulation-only reinforcement-learning project for Unitree G1 locomotion on WSL2. The repository contains a working MuJoCo MJX/Warp GPU environment, explicit torque-limited PD control, RSL-RL PPO training, TensorBoard logging, deterministic checkpoint evaluation, MP4 rendering, and model/reward diagnostics.

## Project Scope

- Target robot: Unitree `g1_29dof_rev_1_0`, as used in the `Lavansurendra/berkeley_humanoid` repository.
- Simulator: MuJoCo MJX with the Warp implementation on an NVIDIA GPU inside WSL2.
- Training: RSL-RL PPO through MuJoCo Playground's supported Torch/JAX wrapper path.
- Control contract: v2, with bounded actions, full episode resets, timestep-integrated rewards, adaptive-KL PPO, and actuator-aware action scales.
- Canonical project location: `/home/ankus/projects/g1_locomotion`.

This project does not reuse, import, inspect, or copy configuration/code from any previous Go1/quadruped project.

The current environment is stochastic during training and deterministic during evaluation by default. The official Unitree model is preserved for provenance/viewing, while a derived feet-only collision model is used for stable Warp training. Detailed interfaces are in [docs/g1_env_contract.md](docs/g1_env_contract.md), training operations are in [docs/training.md](docs/training.md), and robot provenance is in [docs/g1_model_provenance.md](docs/g1_model_provenance.md).

## Quick Start

All commands run from `/home/ankus/projects/g1_locomotion`.

```bash
# Create/update the conda-owned environment and verify the GPU stack.
./scripts/setup.sh
conda run -n g1-locomotion python scripts/verify_stack.py

# Check the official 29-DOF model and the Warp environment contract.
conda run -n g1-locomotion python scripts/check_model.py
conda run -n g1-locomotion python scripts/test_env.py --num-envs 8 --steps 300

# Run a two-iteration end-to-end PPO smoke test.
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 4 --max-iterations 2 --checkpoint-frequency 1 \
  --experiment smoke_stability_v2_docs

# Watch only that experiment in TensorBoard.
./scripts/tensorboard.sh --logdir runs/smoke_stability_v2_docs

# Auto-discover the newest v2 checkpoint and render a short MP4.
conda run -n g1-locomotion python scripts/test_play.py
```

Given a checkpoint path, inspect rewards or view the deterministic actor interactively:

```bash
conda run -n g1-locomotion python scripts/reward_breakdown.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt \
  --command 0.5 0.0 0.0

conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt \
  --render viewer --command 0.5 0.0 0.0
```

## Conda Environment Setup

Set up the pinned conda-owned environment:

```bash
./scripts/setup.sh
```

Manual verification:

```bash
conda run -n g1-locomotion python scripts/verify_stack.py
```

Interactive use:

```bash
conda activate g1-locomotion
python scripts/verify_stack.py
```

Conda owns the environment and Python interpreter. The setup script uses `uv pip` inside the conda env only to synchronize `requirements.lock.txt`; it does not run `uv sync`, create `.venv`, or modify system Python. Registry artifacts are hash-locked. Playground is a VCS dependency, so it is pinned by full Git SHA and `verify_stack.py` checks the installed `direct_url.json` commit.

Regenerate the lock only after intentionally changing `requirements.in`:

```bash
./scripts/lock_requirements.sh
```

The lock command targets CPython 3.12 on Linux x86-64 and selects PyTorch's CUDA 13.0 backend without exposing the PyTorch wheel index as a general package source.

## Pinned Stack

- Conda env: `g1-locomotion`
- Python: `3.12.14`
- conda bootstrap tools: `pip==26.2.1`, `uv==0.12.17`
- MuJoCo Playground source: `google-deepmind/mujoco_playground@4057c147714b6ac09b395377f1a1724bbeacc4d3`
- MuJoCo: `3.13.0`
- MuJoCo MJX: `3.13.0`
- MuJoCo Warp: `3.13.0`
- NVIDIA Warp: `warp-lang==1.17.0`
- JAX CUDA package: `jax[cuda12]==0.6.2`
- PyTorch: `torch==2.9.1+cu130` from the official CUDA 13.0 wheel index
- RSL-RL: `rsl-rl-lib==3.2.0`
- TensorBoard: `tensorboard==2.20.0`
- Setuptools: `setuptools==80.9.0` for TensorBoard CLI `pkg_resources` compatibility

Version rationale:

- Python `3.12` matches MuJoCo Playground's current source install path.
- MuJoCo, MuJoCo MJX, and MuJoCo Warp are aligned at `3.13.0` to avoid API drift.
- `warp-lang==1.17.0` satisfies MuJoCo Playground and is pinned by `mujoco-mjx[warp]==3.13.0`.
- `jax==0.6.2` and `rsl-rl-lib==3.2.0` come from MuJoCo Playground's current lock resolution.
- PyTorch `2.9.1+cu130` is installed from PyTorch's CUDA 13.0 wheel index for the host NVIDIA R596/CUDA 13.2 driver path.

Observed upstream SHAs during dependency selection:

- MuJoCo Playground: `4057c147714b6ac09b395377f1a1724bbeacc4d3`
- MuJoCo Warp main: `87e742d31c96f69a70741c51b9ade43bd8d1b60b`
- NVIDIA Warp main: `061b4d01292b0d493f77a82c0767ef8e5d75e9a6`

## Current Limits

- No claim of late-run stability until a fresh v2 baseline reaches 10,000 iterations.
- No real-robot deployment or hardware interface; this repository remains simulation-only.
- Friction, mass, inertia, motor-strength, PD-gain, latency, or terrain randomization
- No custom Warp-to-Torch conversion layer; the pinned Playground bridge is used.
- Configuration or code copied from previous Go1/quadruped projects

## Unitree G1 Model Assets

The official Unitree `g1_29dof_rev_1_0` MuJoCo XML and its 35 referenced meshes are vendored under `assets/g1/`. Provenance, source SHAs, Wisconsin-vs-Unitree comparison notes, joint order, joint ranges, effort limits, and mesh list are documented in `docs/g1_model_provenance.md`.

Inspect the model:

```bash
conda run -n g1-locomotion python scripts/check_model.py
```

Run a headless render smoke test:

```bash
conda run -n g1-locomotion python scripts/view_model.py --headless
```

For an interactive MuJoCo viewer:

```bash
conda activate g1-locomotion
python scripts/view_model.py
```

These scripts load and render the provenance model independently of the RL environment.

## G1 Environment

The environment lives under `src/g1_locomotion/envs`. It uses the official vendored G1 model, converts the compiled model to direct torque actuators in memory, and runs MuJoCo MJX with the Warp backend. Policy actions are 29 bounded joint-position offsets around the TWIST2 G1 nominal pose; an explicit torque-limited PD controller applies those targets at 500 Hz physics / 50 Hz policy-control timing. The current control contract is version 2: policy outputs are clipped to `[-1, 1]` before control, observation history, and action-rate cost; per-joint target scales request 25% of each actuator's static effort at the nominal pose. Commands are sampled from narrowed Unitree-style ranges and every reward contribution is integrated over the `0.02 s` control timestep.

The training environment now includes MuJoCo Playground G1-style stochastic training aids: actor observation noise, randomized reset xy/yaw/joint-pose scale/root velocity, randomized gait frequency, random planar pushes, and command-axis sparsity. It deliberately does not add Playground friction, mass, inertia, motor-strength, PD-gain, latency, or terrain randomization because those were not part of the inspected Playground G1 joystick environment. Deterministic play/viewer mode disables the stochastic training aids by default unless `--enable-randomization` is passed.

Training uses `assets/g1/g1_scene_mjx_feetonly.xml`, a MuJoCo Playground-style MJX/Warp feet-only collision scene for the same 29-DOF G1. The official full-mesh Unitree scene remains available as `assets/g1/g1_scene.xml` for provenance checks and viewing. MuJoCo Warp contact buffers use MuJoCo Playground's G1 aggregate contact budget with bounded CCD memory for this laptop GPU: `naconmax=65536`, `naccdmax=512`, and `njmax=512`.

## Resolved Issues

- The old requirements file claimed it came from a deleted `uv.lock` export and had no artifact hashes. It is now generated by `scripts/lock_requirements.sh` with dependency annotations and registry hashes; the Playground VCS dependency is separately pinned and verified by full Git SHA.
- Setup previously accepted any Python 3.12/pip/uv patch versions and only printed most stack versions. The conda bootstrap and direct stack are now asserted exactly, dependency consistency is checked, and verification includes Playground provenance, bundled FFmpeg execution, and a nonblank EGL frame.
- TensorBoard CLI initially failed because `tensorboard==2.20.0` imports `pkg_resources`; fixed by pinning `setuptools==80.9.0` and verifying TensorBoard in `scripts/verify_stack.py`.
- MP4 play smoke initially reached video encoding and failed because Python `mediapy` does not provide the `ffmpeg` executable it invokes. Installing conda-forge's full `ffmpeg` package then shadowed WSL/NVIDIA EGL with conda GL libraries and broke headless MuJoCo rendering. The reproducible stack instead pins `imageio-ffmpeg==0.6.0` and passes its isolated bundled executable to `mediapy`; stack verification checks that binary without altering the simulator's GL stack.
- Early 2048-env rollouts produced MuJoCo Warp broadphase overflows with `naconmax=512`; fixed by increasing the aggregate contact budget to `naconmax=65536`.
- Matching `naccdmax=65536` caused CUDA OOM on the RTX 5070 Ti Laptop GPU; fixed by keeping `naccdmax=512` while preserving the larger aggregate contact budget.
- `njmax=90` produced `nefc` constraint overflows in this explicit-PD setup; fixed by using `njmax=512`.
- Long training later hit a CUDA launch failure while using the official full-collision mesh XML. The training config now uses a MuJoCo Playground-style feet-only MJX collision XML to avoid unstable mesh/cylinder collision paths during Warp training, while preserving the official model separately.
- The upstream MuJoCo Playground feet-only XML used a mirrored `right_hip_roll_joint` range that did not match the official Unitree/Wisconsin model. The local training XML overrides that joint range back to `[-2.9671, 0.5236]` so the MJX collision model keeps the official 29-DOF robot limits.
- Added a repo-style feet slide penalty with weight `-0.2` to discourage scuttling/sliding while a foot is in floor contact. The weight matches Unitree RL Lab and Unitree RL Gym's closest contact-velocity term; MuJoCo Playground G1 uses `-0.25`, and Berkeley-style biped configs commonly use `-0.1`.
- Set `feet_air_time=0.0` for now after adding explicit gait contact and swing-height shaping. The air-time metric is still logged for diagnostics, but it no longer contributes to reward.
- Added a no-phase `feet_swing_height=-20.0` penalty using the simple Unitree G1-style airborne-foot target-height term: airborne foot sites are shaped toward `0.08 m`, while planted feet and near-zero commands are ignored. Berkeley Humanoid Lite's public biped config uses air-time rather than a separate height target, so this is documented as Unitree-style swing-height shaping.
- Added gait timing/contact shaping using Unitree RL Lab 29DOF values: `period=0.8`, offsets `[0.0, 0.5]`, stance threshold `0.55`, and `gait_contact=0.5`. The actor observation now appends `[sin(2*pi*phase), cos(2*pi*phase)]`, zeroed for near-zero commands.
- Added an asymmetric RSL-RL critic observation: the actor/deployed policy now receives the 98-float `state`, while the critic receives `privileged_state` with the same 98 floats plus body-frame base linear velocity, for 101 floats total.
- Added an orientation penalty. The current config uses `-5.0 * (projected_gravity_x^2 + projected_gravity_y^2)` to discourage roll/pitch tilt while keeping yaw free.
- Added Unitree-style smoothness penalties: `action_rate=-0.01`, `dof_vel=-1e-3`, and `dof_acc=-2.5e-7`. Under control contract v2, the action-rate term uses consecutive clipped/applied actions, matching the values used for PD targets and previous-action observations.
- A previous raw-action bridge let unbounded Gaussian outputs enter action history and the action-rate reward. In late-run failures, action-rate cost and value loss jumped abruptly together. Control contract v2 retires that bridge: the environment receives only clipped actions, while the Torch wrapper logs pre-clip policy statistics without using them for dynamics or reward. The bounded action-rate cost cannot exceed `29 * (2^2) = 116`; at weight `-0.01` and `ctrl_dt=0.02`, its worst contribution is `-0.0232` per step.
- Auto-reset previously did not guarantee that every stateful environment field was replaced at an episode boundary. The v2 wrapper uses MuJoCo Playground's `wrap_for_brax_training(..., full_reset=True)`, resetting action history, gait phase, foot timers, command/push schedules, velocities, RNG state, and observations.
- Reward weights previously acted as per-step amounts rather than rates, making returns depend directly on the 50 Hz policy rate. Every v2 reward contribution is `raw_term * configured_weight * ctrl_dt`; raw costs and weighted rates remain separately logged. New reward/return curves are therefore roughly 50 times smaller than comparable legacy curves.
- Height-only termination allowed severely tilted and self-colliding states to remain in rollouts. V2 retains Wisconsin's `0.5-1.0 m` height bounds and adds a 70-degree tilt limit plus the already-active foot-to-foot and same-side hand-to-thigh sensors. It deliberately does not expand Warp collision pairs.
- Fixed-learning-rate PPO could keep applying the same step size during a sharp policy-distribution change. V2 uses RSL-RL adaptive KL scheduling with initial LR `3e-4`, desired KL `0.01`, and RSL-RL's built-in `[1e-5, 1e-2]` bounds.
- The former uniform `0.5 rad` action scale could ask low-effort joints for disproportionately aggressive targets. V2 derives each scale as `0.25 * effort_limit / Kp`, preserving TWIST2 gains while bounding the nominal static request to 25% of actuator effort.
- Changed the training command ranges to the narrower G1 range `vx=[-0.5, 1.0]`, `vy=[-0.3, 0.3]`, and `yaw_rate=[-0.2, 0.2]`.
- Added MuJoCo Playground G1-style stochastic training aids: actor observation noise, reset randomization, per-episode gait-frequency randomization, random planar pushes, and command-axis sparsity. A JAX/RSL-RL smoke run initially failed because push metric keys were only created after stepping; fixed by seeding `randomization/push_magnitude`, `randomization/push_x`, and `randomization/push_y` at reset so the `lax.scan` metrics tree is stable.

Run the environment smoke test:

```bash
conda run -n g1-locomotion python scripts/test_env.py
```

The action and observation contracts are documented in `docs/g1_env_contract.md`.

## Training

The training entrypoint uses a small project subclass of MuJoCo Playground's RSL-RL wrapper with RSL-RL's `OnPolicyRunner`/PPO implementation. It retains Playground's Torch/JAX DLPack bridge and bounded `[-1, 1]` step path, enables complete episode resets, and logs raw pre-clip policy statistics only as diagnostics. The actor uses the deployable 98-float `state`; the critic uses 101-float `privileged_state` with body-frame base linear velocity appended. PPO, GAE, critic loss, clipping, gradient clipping, and adaptive-KL learning-rate updates remain entirely RSL-RL-owned.

Run a short end-to-end PPO smoke test:

```bash
conda run -n g1-locomotion python scripts/train.py --num-envs 4 --max-iterations 2 --checkpoint-frequency 1 --experiment smoke
```

Runs are written under `runs/<experiment>/<timestamp>/` with the effective environment config, PPO config, run metadata, TensorBoard logs, finite-loss history, and RSL-RL `model_*.pt` checkpoints.

Recommended contract-v2 validation baseline:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 2048 \
  --max-iterations 10000 \
  --checkpoint-frequency 500 \
  --experiment g1_stability_v2_10k
```

This collects `2048 * 24 * 10000 = 491,520,000` environment transitions. Each PPO iteration advances every environment by `24 * 0.02 = 0.48` simulated seconds. The 1,000-step episode limit is 20 simulated seconds. This is a stability-validation baseline, not proof of a solved gait. See [docs/training.md](docs/training.md) for PPO defaults, metric interpretation, resume rules, and evaluation commands.

Do not resume a control-contract-v1 checkpoint under the v2 defaults. `scripts/train.py` rejects that mismatch explicitly. Legacy checkpoints remain loadable in `scripts/play.py` for visual comparison, with a warning; their rewards and returns are not numerically comparable to v2 because rewards are now timestep-scaled.

## Local Checkpoint References

`runs/` is intentionally ignored by Git. The following paths describe artifacts on the current workstation and are not present in a fresh clone.

### Legacy V1 Visual Baseline

This gait + swing-height checkpoint predates control contract v2. It is the best checkpoint from the local `g1_10k_gait_no_airtime` run by its legacy TensorBoard mean reward and episode length:

```text
runs/g1_10k_gait_no_airtime/20260922_182105/model_6500.pt
```

Selection notes from TensorBoard:

```text
checkpoint: model_6500.pt
Train/mean_reward: 1904.30
Train/mean_episode_length: 1000.00
actor observation: 98
critic observation: 101
```

Fixed straight-forward deterministic eval at `[0.5, 0.0, 0.0]` completed a full 1000-step episode without falling:

```text
total episode reward: 1937.176
episode length: 1000
termination reason: timeout
average commanded velocity tracking error: 0.385491
average actuator effort: 8.010837
summary: runs/play/20260922_205959/evaluation.json
```

This checkpoint is useful as a legacy visual/evaluation baseline, but it is not a solved locomotion policy. It predates control contract v2 and must not be resumed under the current environment defaults. Its reward values are not comparable to new timestep-scaled runs.

Fixed straight-forward viewer command for this legacy checkpoint:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/g1_10k_gait_no_airtime/20260922_182105/model_6500.pt \
  --env-config runs/g1_10k_gait_no_airtime/20260922_182105/env_config.yaml \
  --render viewer \
  --steps 1000 \
  --render-every 1 \
  --command 0.5 0.0 0.0
```

An older feet-airtime v1 reference used for visual inspection is also local:

```text
runs/g1_10k_feet_airtime/20260921_220546/model_5500.pt
```

By default, `scripts/play.py` disables command sampling, reset randomization, observation noise, gait-frequency randomization, and pushes so viewer/eval runs are repeatable. When `--command` is provided, that command is fixed exactly. Pass `--enable-randomization` only when you intentionally want training-style stochastic evaluation.

### V2 Smoke Checkpoint

The local validation-only v2 smoke checkpoint is:

```text
runs/smoke_stability_v2_docs/20261014_180907/model_1.pt
```

This proves the v2 rollout/update/checkpoint path, not locomotion quality. `scripts/test_play.py` now discovers the newest contract-v2 checkpoint from run metadata instead of relying on a particular experiment name.

## TensorBoard Training Dashboard

TensorBoard is installed into the `g1-locomotion` conda environment by `./scripts/setup.sh`. Point it at one experiment directory so historical runs do not clutter the dashboard.

Start the dashboard in one terminal:

```bash
cd ~/projects/g1_locomotion
./scripts/tensorboard.sh --logdir runs/g1_stability_v2_10k
```

Then start training in another terminal:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 2048 --max-iterations 10000 --checkpoint-frequency 500 \
  --experiment g1_stability_v2_10k
```

The launcher defaults to `http://127.0.0.1:6006` and tries to open Chrome from WSL first, then falls back to the default browser. To run without opening a browser:

```bash
./scripts/tensorboard.sh --logdir runs/g1_stability_v2_10k --browser none
```

## Deterministic Policy Evaluation

Evaluate a trained RSL-RL checkpoint with deterministic actor inference and MP4 rendering:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt --render video
```

Run the evaluation smoke test after a smoke training checkpoint exists:

```bash
conda run -n g1-locomotion python scripts/test_play.py
```

`scripts/play.py` uses the same environment, observation construction, action scaling, MuJoCo Playground wrapper, and RSL-RL actor architecture as training. It does not add reward terms. It disables training-time stochasticity by default for deterministic evaluation, while `--enable-randomization` is available when you want to inspect the policy under the same noise/reset/push path used during training.

## WSL And VS Code Access

Open the repository from inside WSL:

```bash
cd ~/projects/g1_locomotion
code .
```

From Windows Explorer or VS Code, use the WSL network path:

```text
\\wsl$\Ubuntu\home\ankus\projects\g1_locomotion
```

If Windows lists the distro under a different name, replace `Ubuntu` with the name shown under `\\wsl$`.

Keep this Linux filesystem path as the canonical project copy. Do not move the repository to `/mnt/c`, `/mnt/d`, or another Windows-mounted filesystem.

## Diagnostics

Run:

```bash
./scripts/system_info.sh
```

The script reports Ubuntu, WSL, kernel, CPU, memory, Python, Git, and NVIDIA GPU visibility details. GPU passthrough is considered working only when `/dev/dxg` exists and `nvidia-smi` completes successfully.
