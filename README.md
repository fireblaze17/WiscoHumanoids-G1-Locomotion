# WiscoHumanoids G1 Locomotion

Simulation-only reinforcement-learning environment for the Unitree `g1_29dof_rev_1_0`. The project runs MuJoCo MJX with the Warp GPU backend and trains policies with RSL-RL PPO.

## What Is Included

- Official Unitree G1 29-DOF revision 1.0 model and required meshes
- MuJoCo Warp batched simulation on NVIDIA GPUs
- Explicit torque-limited PD joint control
- 29-dimensional bounded policy action space
- Asymmetric 98-dimensional actor and 101-dimensional critic observations
- Command-conditioned locomotion rewards and training randomization
- RSL-RL PPO training, TensorBoard logging, checkpoint evaluation, and MP4 rendering
- Reproducible conda environment and hashed Python dependency lock

This repository does not include a real-robot interface or a pretrained production policy. Training outputs under `runs/` are local and intentionally excluded from Git.

## Requirements

- Windows with WSL2 and an Ubuntu distribution
- NVIDIA GPU with WSL GPU passthrough enabled
- Miniforge, Conda, or Mamba available inside WSL
- Git

Confirm GPU access before setup:

```bash
ls -l /dev/dxg
nvidia-smi
```

Keep the repository in the WSL Linux filesystem, such as `~/projects`, rather than under `/mnt/c` or another Windows-mounted path. A separate Linux CUDA toolkit is not required; the pinned JAX, PyTorch, and Warp packages provide the required runtime components.

## Setup

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/fireblaze17/WiscoHumanoids-G1-Locomotion.git
cd WiscoHumanoids-G1-Locomotion
./scripts/setup.sh
```

The setup script creates or updates the conda environment `g1-locomotion`, synchronizes the exact packages in `requirements.lock.txt`, and verifies CUDA, Warp, JAX, PyTorch, MuJoCo, EGL rendering, FFmpeg, MuJoCo Playground, and RSL-RL. It does not create `.venv` or modify system Python.

Activate the environment for interactive work:

```bash
conda activate g1-locomotion
```

## Verify The Project

```bash
# Host, WSL, and GPU diagnostics
./scripts/system_info.sh

# Pinned software stack and GPU execution
conda run -n g1-locomotion python scripts/verify_stack.py

# Official 29-DOF model structure and assets
conda run -n g1-locomotion python scripts/check_model.py

# Batched Warp environment and control/reward contracts
conda run -n g1-locomotion python scripts/test_env.py --num-envs 8 --steps 300
```

Successful validation should report:

- MuJoCo implementation: Warp
- JAX backend: GPU
- PyTorch CUDA: available
- Robot actuators: 29
- Actor observation size: 98
- Critic observation size: 101
- Finite batched rollouts with no contract-test failures

## Environment Summary

The main configuration is [`configs/g1_flat.yaml`](configs/g1_flat.yaml).

| Setting | Value |
| --- | --- |
| Robot | Unitree `g1_29dof_rev_1_0` |
| Physics rate | 500 Hz (`sim_dt=0.002`) |
| Policy/control rate | 50 Hz (`ctrl_dt=0.02`) |
| Physics substeps | 10 |
| Episode limit | 1,000 policy steps / 20 simulated seconds |
| Actions | 29 clipped joint-position offsets in `[-1, 1]` |
| Controller | Explicit torque-limited PD |
| Actor observation | 98 values |
| Critic observation | 101 values |
| Simulation backend | MuJoCo MJX Warp |

Control contract v2 uses actuator-aware action scales, complete episode resets, and reward contributions scaled by `ctrl_dt`. Training enables configured command sampling, reset randomization, actor observation noise, gait-frequency variation, and planar pushes. Evaluation disables these by default.

See [`docs/g1_env_contract.md`](docs/g1_env_contract.md) for the complete action, observation, reward, termination, control, and randomization contracts.

## Training

Run a short end-to-end smoke test first:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 4 \
  --max-iterations 2 \
  --checkpoint-frequency 1 \
  --experiment smoke
```

Runs are saved under:

```text
runs/<experiment>/<timestamp>/
```

Each run contains checkpoints, TensorBoard events, effective environment and PPO configurations, metadata, and finite-loss history.

A full contract-v2 stability baseline is:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 2048 \
  --max-iterations 10000 \
  --checkpoint-frequency 500 \
  --experiment g1_stability_v2_10k
```

This configuration collects 491,520,000 transitions. Treat it as a long-run stability validation, not a guarantee of a solved locomotion policy. Hardware capacity may require adjusting `--num-envs`; keep the environment and PPO contracts recorded with each run.

PPO defaults, rollout math, resume rules, and metric interpretation are documented in [`docs/training.md`](docs/training.md).

## TensorBoard

Start TensorBoard in a separate terminal and scope it to one experiment:

```bash
./scripts/tensorboard.sh --logdir runs/g1_stability_v2_10k
```

The dashboard opens at `http://127.0.0.1:6006` and updates while training runs. Use `--browser none` when browser auto-launch is not desired.

## Evaluate A Checkpoint

Render a deterministic MP4:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt \
  --render video \
  --command 0.5 0.0 0.0
```

Open the interactive MuJoCo viewer through WSLg:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt \
  --render viewer \
  --command 0.5 0.0 0.0
```

Inspect reward components and action/torque diagnostics:

```bash
conda run -n g1-locomotion python scripts/reward_breakdown.py \
  --checkpoint runs/<experiment>/<timestamp>/model_<iteration>.pt \
  --command 0.5 0.0 0.0
```

Omit `--command` and add `--enable-randomization` to evaluate with training-style command sampling and randomization. Legacy control-contract-v1 checkpoints are visual references only and cannot be resumed for v2 training.

## Model Viewing

View the nominal model without a policy:

```bash
conda run -n g1-locomotion python scripts/view_model.py
```

For a headless rendering check:

```bash
conda run -n g1-locomotion python scripts/view_model.py --headless
```

The immutable official model is used for provenance and viewing. Training uses the separately named feet-only collision model optimized for MJX/Warp. Source attribution, licenses, hashes, joint ordering, limits, and model differences are documented in [`docs/g1_model_provenance.md`](docs/g1_model_provenance.md).

## Repository Layout

```text
assets/g1/                 Vendored G1 XML, meshes, and licenses
configs/g1_flat.yaml       Environment, controller, reward, and randomization config
docs/                      Environment, training, and model documentation
scripts/setup.sh           Idempotent conda environment setup
scripts/verify_stack.py    Dependency, CUDA, EGL, and backend verification
scripts/test_env.py        Environment contract and rollout tests
scripts/train.py           RSL-RL PPO training entrypoint
scripts/play.py            Checkpoint evaluation and rendering
src/g1_locomotion/         Environment and training integration code
```

## Dependency Updates

Direct dependencies are pinned in `requirements.in`; the complete resolution is committed in `requirements.lock.txt`. After intentionally changing a direct dependency, regenerate and verify the lock with:

```bash
./scripts/lock_requirements.sh
./scripts/setup.sh
```

Do not independently upgrade CUDA, JAX, PyTorch, Warp, MuJoCo, Playground, or RSL-RL packages. They are pinned as a tested compatibility set.

## VS Code And Windows Access

From the repository inside WSL:

```bash
code .
```

The same files are accessible from Windows at:

```text
\\wsl$\<DistroName>\home\<LinuxUser>\projects\WiscoHumanoids-G1-Locomotion
```

Use VS Code's WSL extension so terminals, Python, Git, and GPU tools run inside the Linux environment.

## Common Problems

- Missing `/dev/dxg` or failing `nvidia-smi`: update the Windows NVIDIA driver, run `wsl --shutdown` from PowerShell, and reopen Ubuntu.
- JAX or PyTorch reports CPU: rerun `./scripts/setup.sh`; do not add a separate CUDA toolkit as a workaround.
- Interactive viewer fails: verify WSLg and use `--render video` for reliable offscreen evaluation.
- TensorBoard is cluttered: pass an experiment-specific `--logdir` instead of the entire `runs/` directory.
- Training runs out of GPU memory: reduce `--num-envs`; do not arbitrarily increase MuJoCo contact or CCD buffers.
