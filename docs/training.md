# G1 Training Operations

This document describes control-contract-v2 training. Environment semantics are defined in `docs/g1_env_contract.md`; robot provenance is defined in `docs/g1_model_provenance.md`.

## Runtime Contract

- Simulation: MuJoCo MJX with `impl=warp` on the JAX GPU device.
- Optimizer: RSL-RL `OnPolicyRunner` and PPO on PyTorch CUDA.
- Parallelism: one batched JAX/Warp environment with `--num-envs` actors.
- Actor observation: 98-float `state`.
- Critic observation: 101-float `privileged_state`.
- Action: 29 policy outputs clipped to `[-1, 1]` before PD targets, action history, and action-rate cost.
- Physics/control timing: 500 Hz physics, 50 Hz policy control, 10 physics substeps.
- Episode limit: 1,000 policy steps, or 20 simulated seconds.
- Episode resets: full reset of physics data, observations, actions, commands, gait/foot timers, push schedule, RNG, and velocity state.

Training only accepts control contract v2. A v1 checkpoint cannot be resumed, even when paired with an archived v1 YAML. Legacy playback is visual-only and does not recreate the old reward/control implementation.

## PPO Defaults

| Setting | Value |
| --- | ---: |
| Runner | `OnPolicyRunner` |
| Policy | `ActorCritic` |
| Actor hidden layers | `[512, 256, 128]` |
| Critic hidden layers | `[512, 256, 128]` |
| Activation | `elu` |
| Initial policy std | `1.0` |
| Steps per environment | `24` |
| Learning epochs | `5` |
| Minibatches | `4` |
| PPO clip | `0.2` |
| Value-loss coefficient | `1.0` |
| Clipped value loss | enabled |
| Entropy coefficient | `0.001` |
| Initial learning rate | `3e-4` |
| LR schedule | adaptive KL |
| Desired KL | `0.01` |
| Maximum gradient norm | `1.0` |
| Discount `gamma` | `0.99` |
| GAE `lambda` | `0.95` |
| Empirical observation normalization | enabled |
| RND / symmetry augmentation | disabled |

RSL-RL owns PPO, GAE, clipping, value loss, gradient clipping, and adaptive-KL updates. Its adaptive schedule divides or multiplies the learning rate by `1.5` when KL is above `2x` or below `0.5x` the target, bounded to `[1e-5, 1e-2]`.

For `N` environments, each iteration collects `N * 24` transitions. With 2,048 environments, that is 49,152 transitions per iteration and a nominal PPO minibatch size of 12,288 transitions before epoch reshuffling.

## Validation Baseline

The recommended fresh v2 stability run is:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 2048 \
  --max-iterations 10000 \
  --checkpoint-frequency 500 \
  --experiment g1_stability_v2_10k
```

It collects 491,520,000 transitions. Each iteration advances each environment by 0.48 simulated seconds. A complete timeout episode is 20 simulated seconds. Hardware throughput and wall-clock duration are not implied by simulated time.

Do not claim that the late-run collapse is fixed until a fresh v2 run completes all 10,000 iterations. The two-iteration smoke run only proves rollout collection, PPO update, logging, and checkpoint writing.

## TensorBoard

Scope TensorBoard to the active experiment so old runs do not overlap:

```bash
./scripts/tensorboard.sh --logdir runs/g1_stability_v2_10k
```

Important groups:

- `Train/*`: mean reward, episode length, policy/value losses, entropy, policy standard deviation, learning rate, and KL data exposed by RSL-RL.
- `reward_rate/*`: weighted rates before multiplying by `ctrl_dt`.
- `reward/*`: actual timestep-scaled PPO reward contributions; `reward/total` equals their sum.
- `action/raw_*`: unbounded actor-output diagnostics recorded before clipping.
- `action/applied_*`: bounded actions used by control, observation history, and action-rate cost.
- `action/rate_cost`: bounded applied-action change cost; it cannot exceed 116.
- `torque/raw_sum_sq`: unnormalized squared-torque cost averaged across physics substeps.
- `termination/*`: low/high height, tilt, illegal contact, and nonfinite causes.

After timestep scaling, v2 reward and return curves are expected to be about 50 times smaller than comparable v1 curves. Compare rates to rates and contributions to contributions; do not compare v1 return magnitude directly with v2.

## Resume Rules

Resume only a checkpoint whose adjacent `env_config.yaml` declares contract v2:

```bash
conda run -n g1-locomotion python scripts/train.py \
  --num-envs 2048 \
  --max-iterations 1000 \
  --checkpoint-frequency 500 \
  --experiment g1_stability_v2_resume \
  --resume runs/g1_stability_v2_10k/<timestamp>/model_<iteration>.pt
```

The new run receives its own timestamped directory and saves the effective environment config, PPO config, CLI metadata, package/device facts, losses, event files, and checkpoints. Existing run directories are never overwritten.

## Evaluation

Deterministic fixed-command MP4:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/g1_stability_v2_10k/<timestamp>/model_<iteration>.pt \
  --render video --command 0.5 0.0 0.0
```

Interactive WSLg viewer:

```bash
conda run -n g1-locomotion python scripts/play.py \
  --checkpoint runs/g1_stability_v2_10k/<timestamp>/model_<iteration>.pt \
  --render viewer --command 0.5 0.0 0.0
```

Reward and diagnostic breakdown:

```bash
conda run -n g1-locomotion python scripts/reward_breakdown.py \
  --checkpoint runs/g1_stability_v2_10k/<timestamp>/model_<iteration>.pt \
  --command 0.5 0.0 0.0
```

Play disables reset randomization, observation noise, randomized gait frequency, pushes, and command sampling by default. `--command` sets the exact command. Use `--enable-randomization` only for intentional stochastic evaluation.
