# G1 Flat Environment Contract V2

This environment is a working Unitree `g1_29dof_rev_1_0` flat-ground locomotion training environment. It is simulation-only and uses MuJoCo MJX with the Warp backend. It includes command sampling, MuJoCo Playground G1-style training randomization/noise, and a repo-style locomotion reward. It does not implement friction, mass, inertia, motor-strength, PD-gain, latency, or terrain randomization.

## Model And Control

- Training model: `assets/g1/g1_scene_mjx_feetonly.xml`, a MuJoCo Playground-style MJX/Warp feet-only collision scene for the same Unitree G1 29-DOF rev 1.0 body, joint, mass, inertia, range, and effort structure.
- Official provenance/viewing model: `assets/g1/g1_scene.xml`, which includes the vendored official Unitree `g1_29dof_rev_1_0.xml`.
- The official XML remains unchanged; the MJX training XML makes visual meshes non-colliding and uses simplified feet/thigh/shin/hand collision geoms to avoid unstable full-mesh collision paths in Warp training.
- At environment construction time, the compiled MuJoCo model is converted in memory to direct torque actuators.
- Policy actions still represent joint-position offsets, but those targets are applied by an explicit torque-limited PD controller.
- Control timestep: `0.02`.
- Simulation timestep: `0.002`.
- Physics rate: `500 Hz`.
- Policy/control rate: `50 Hz`.
- Physics substeps per policy step: `10`.
- MuJoCo Warp contact buffers use MuJoCo Playground's G1 aggregate contact budget with bounded CCD memory for this laptop GPU: `naconmax=65536`, `naccdmax=512`, `njmax=512`.
- Action dimension: `29`.
- Action range: policy actions are clipped to `[-1, 1]`.
- Control contract version: `2`.
- Action meaning:

```text
action_scale[j] = 0.25 * effort_limit[j] / kp[j]
target_qpos[j] = nominal_joint_pose[j] + action_scale[j] * clipped_action[j]
target_qpos = clip(target_qpos, joint_lower_limits, joint_upper_limits)
```

Actuator effort limits come from the official joint `actuatorfrcrange` values. At the nominal pose with zero joint velocity, an action of `+1` therefore requests 25% of the actuator's positive static effort before joint-limit clipping. Numeric scalar action scales remain readable only for archived configuration compatibility.

| # | Joint | Action scale (rad) |
| ---: | --- | ---: |
| 1 | `left_hip_pitch_joint` | `0.220000` |
| 2 | `left_hip_roll_joint` | `0.347500` |
| 3 | `left_hip_yaw_joint` | `0.220000` |
| 4 | `left_knee_joint` | `0.231667` |
| 5 | `left_ankle_pitch_joint` | `0.312500` |
| 6 | `left_ankle_roll_joint` | `0.312500` |
| 7 | `right_hip_pitch_joint` | `0.220000` |
| 8 | `right_hip_roll_joint` | `0.347500` |
| 9 | `right_hip_yaw_joint` | `0.220000` |
| 10 | `right_knee_joint` | `0.231667` |
| 11 | `right_ankle_pitch_joint` | `0.312500` |
| 12 | `right_ankle_roll_joint` | `0.312500` |
| 13 | `waist_yaw_joint` | `0.146667` |
| 14 | `waist_roll_joint` | `0.083333` |
| 15 | `waist_pitch_joint` | `0.083333` |
| 16 | `left_shoulder_pitch_joint` | `0.156250` |
| 17 | `left_shoulder_roll_joint` | `0.156250` |
| 18 | `left_shoulder_yaw_joint` | `0.156250` |
| 19 | `left_elbow_joint` | `0.156250` |
| 20 | `left_wrist_roll_joint` | `0.156250` |
| 21 | `left_wrist_pitch_joint` | `0.031250` |
| 22 | `left_wrist_yaw_joint` | `0.031250` |
| 23 | `right_shoulder_pitch_joint` | `0.156250` |
| 24 | `right_shoulder_roll_joint` | `0.156250` |
| 25 | `right_shoulder_yaw_joint` | `0.156250` |
| 26 | `right_elbow_joint` | `0.156250` |
| 27 | `right_wrist_roll_joint` | `0.156250` |
| 28 | `right_wrist_pitch_joint` | `0.031250` |
| 29 | `right_wrist_yaw_joint` | `0.031250` |

At every physics substep, the controller applies:

```text
torque = kp * (target_qpos - qpos[7:]) - kd * qvel[6:]
torque = clip(torque, actuator_effort_lower, actuator_effort_upper)
```

The PD gains are copied from TWIST2's G1 29-action mimic config:

| Joint bucket | Kp | Kd |
| --- | ---: | ---: |
| `hip_yaw` | 100 | 2 |
| `hip_roll` | 100 | 2 |
| `hip_pitch` | 100 | 2 |
| `knee` | 150 | 4 |
| `ankle` | 40 | 2 |
| `waist` | 150 | 4 |
| `shoulder` | 40 | 5 |
| `elbow` | 40 | 5 |
| `wrist` | 40 | 5 |

## Commands And Rewards

Commands are sampled from narrowed Unitree-style command ranges:

| Command | Range |
| --- | ---: |
| `vx` | `[-0.5, 1.0]` m/s |
| `vy` | `[-0.3, 0.3]` m/s |
| `yaw_rate` | `[-0.2, 0.2]` rad/s |

Commands are resampled every `10.0` seconds, or `500` policy steps at `ctrl_dt=0.02`. As in Unitree's legged-gym base environment, small xy commands with norm below `0.2` are zeroed.

Training command sampling also uses MuJoCo Playground G1-style axis sparsity while preserving the narrower ranges:

```text
command = sampled_command * amplitude * active_mask
amplitude = [1.0, 0.8, 1.0]
active_probability = [0.9, 0.25, 0.5]
```

Configured weights are reward rates. The actual PPO reward on each 50 Hz control step is:

```text
reward_rate = 1.0 * exp(-lin_vel_error / 0.25)
            + 0.5 * exp(-yaw_rate_error / 0.25)
            - 5.0 * orientation_cost
            + 0.15 * alive
            - 1.0e-5 * mean_substep_sum_torque_squared
            - 0.01 * action_rate_cost
            - 1.0e-3 * dof_vel_cost
            - 2.5e-7 * dof_acc_cost
            - 0.2 * feet_slide_cost
            + 0.5 * gait_contact_reward_raw
            + 0.0 * feet_air_time_reward_raw
            - 20.0 * feet_swing_height_cost
reward = reward_rate * ctrl_dt
```

where `lin_vel_error` compares body-frame base `vx/vy` against the sampled command, `yaw_rate_error` compares body-frame yaw rate against the sampled command, `orientation_cost = projected_gravity_x^2 + projected_gravity_y^2`, and the torque penalty is the raw Unitree-style `sum(torque^2)` averaged across the 10 physics substeps. Torque is not normalized by actuator limits in the reward.

The orientation penalty is:

```text
orientation_cost = projected_gravity_x^2 + projected_gravity_y^2
reward_rate_orientation = -5.0 * orientation_cost
reward_orientation = reward_rate_orientation * 0.02
```

This penalizes roll/pitch tilt while leaving yaw free. The current weight is `-5.0`.

The smoothness penalties are Unitree-style regularizers:

```text
action_rate_cost = sum((applied_action_t - applied_action_{t-1})^2)
reward_rate_action_rate = -0.01 * action_rate_cost
reward_action_rate = reward_rate_action_rate * 0.02

dof_vel_cost = sum(qvel_joint^2)
reward_rate_dof_vel = -1.0e-3 * dof_vel_cost
reward_dof_vel = reward_rate_dof_vel * 0.02

dof_acc_cost = sum(((qvel_joint_t - qvel_joint_{t-1}) / ctrl_dt)^2)
reward_rate_dof_acc = -2.5e-7 * dof_acc_cost
reward_dof_acc = reward_rate_dof_acc * 0.02
```

`applied_action` is the clipped `[-1, 1]` action used by the PD target, previous-action observation, and rate cost. The maximum transition on each joint is `2`, so `action_rate_cost <= 29 * 4 = 116`; its maximum negative contribution is `-0.01 * 116 * 0.02 = -0.0232`. The Torch wrapper separately logs raw pre-clip action magnitude and clipping fraction without feeding raw values into environment state.

These terms penalize jerky applied policy target changes, high joint speeds, and high joint accelerations.

The planted-foot slide penalty is:

```text
feet_slide_cost = sum_i(contact_i * norm(foot_global_linear_velocity_i_xy))
reward_feet_slide = -0.2 * feet_slide_cost * 0.02
```

Only feet currently touching the floor contribute. A planted foot moving horizontally receives negative reward; an airborne foot does not. The default `-0.2` weight matches Unitree RL Lab's `feet_slide` weight and Unitree RL Gym's closest `contact_no_vel` scale. MuJoCo Playground G1 uses `feet_slip=-0.25`, while Berkeley-style biped configs commonly use `feet_slide=-0.1`.

The gait contact reward is:

```text
phase = (phase + ctrl_dt / 0.8) % 1.0
left_phase = phase
right_phase = (phase + 0.5) % 1.0
is_stance_i = leg_phase_i < 0.55
gait_contact_reward_raw = sum_i(is_contact_i == is_stance_i) * moving_command
reward_gait_contact = 0.5 * gait_contact_reward_raw * 0.02
```

This is a simple alternating gait timing term using the Unitree RL Lab 29DOF values: `period=0.8`, offsets `[0.0, 0.5]`, stance threshold `0.55`, and weight `0.5`. It is disabled when command norm is below `0.1`.

The foot air-time reward is:

```text
feet_air_time_reward_raw = sum_i(
    clip(feet_air_time_i - 0.2, min=0.0, max=0.3) * first_contact_i
) * moving_command
reward_feet_air_time = 0.0 * feet_air_time_reward_raw * 0.02
```

`feet_air_time_i` accumulates while foot `i` is off the floor and resets when that foot touches down. `first_contact_i` is true on touchdown after a swing. The `0.2` to `0.5` second window is still logged for diagnostics, but its current weight is `0.0`; it follows the MuJoCo Playground/Berkeley-style air-time idea: no credit for tiny toe taps, capped credit for longer swings. `moving_command` disables this reward when the command norm is below `0.1`, so the policy is not paid to step in place for a zero command.

The no-phase swing-height penalty is:

```text
feet_swing_height_cost = sum_i((foot_site_z_i - 0.08)^2 * airborne_i)
reward_feet_swing_height = -20.0 * feet_swing_height_cost * 0.02
```

This follows the simple Unitree G1-style swing-height term: only airborne feet are shaped toward a target height, and planted feet do not contribute. This term itself is phase-independent; gait phase and left/right timing come from the separate gait-contact term. The term is disabled when the command norm is below `0.1`.

Termination preserves the Wisconsin height thresholds and adds the Unitree MJLab 70-degree tilt threshold plus already-active self-contact sensors:

```text
done = base_z < 0.5
    or base_z > 1.0
    or torso_tilt > 70 degrees
    or foot_to_foot_contact
    or left_hand_to_left_thigh_contact
    or right_hand_to_right_thigh_contact
    or nonfinite_state
```

No negative termination reward is added, and no new collision pairs are enabled. Individual reasons are logged under `termination/*`.

## Training Randomization

Training uses the following MuJoCo Playground G1-style stochastic aids:

```yaml
observation_noise:
  level: 1.0
  joint_pos: 0.03
  joint_vel: 1.5
  gravity: 0.05
  linvel: 0.1
  gyro: 0.2

reset_randomization:
  base_xy: [-0.5, 0.5]
  base_yaw: [-3.14, 3.14]
  joint_pose_scale: [0.5, 1.5]
  root_velocity: [-0.5, 0.5]

gait_randomization:
  frequency: [1.25, 1.5]

push_randomization:
  interval_range: [5.0, 10.0]
  magnitude_range: [0.1, 2.0]
```

Observation noise is applied only to the actor `state` channels for gyro, projected gravity, joint-position error, and joint velocity. Command, previous action, and gait phase are not noised. The `linvel` scale is kept in config for parity with Playground G1, but the actor observation has no base-linear-velocity channel; the critic-only body-frame base linear velocity remains clean privileged information.

Pushes add a random planar root velocity at sampled 5-10 second intervals. Reset randomization changes base xy, base yaw, joint-pose scale around the nominal pose, and root velocity. Gait randomization samples a per-episode gait frequency instead of using only the fixed `period=0.8`.

## Reproducibility And Evaluation

- Reset uses the RNG argument for command sampling, reset randomization, gait frequency, push interval, and actor observation noise.
- Step consumes random keys for command resampling, observation noise, and randomized push direction/magnitude.
- Same seed produces the same randomized reset and rollout.
- Different seeds intentionally produce different commands/resets/noise/push schedules.
- Sensor noise fields in the MuJoCo model are set to zero; actor observation noise is added in the environment.
- Friction, mass, actuator gains, terrain, and all fixed physics parameters are unchanged.
- `scripts/play.py` disables command sampling, reset randomization, observation noise, gait-frequency randomization, and pushes by default for deterministic viewer/evaluation runs. Pass `--enable-randomization` to evaluate with training-style stochasticity.
- RSL-RL training uses full episode resets. Every terminated environment receives fresh data, observation, command, action history, gait/foot timer, push schedule, RNG, and velocity state before its next transition.
- `reward_rate/*` metrics are weighted rates before timestep integration; `reward/*` metrics are actual per-step contributions. `reward/total` is exactly the sum of the contribution metrics.
- Multiplying by `ctrl_dt` treats configured weights as rates per simulated second. It prevents reward magnitude and returns from changing merely because the policy/control frequency changes; at 50 Hz, each per-step contribution is its weighted rate times `0.02`.
- Contract-v1 checkpoints used raw action history and unscaled rewards. Training refuses to resume them under v2; play mode permits visual-only loading with a compatibility warning.

## Observation Layout

The policy observation is a flat vector of exactly 98 floats:

| Indices | Size | Name | Meaning |
| --- | ---: | --- | --- |
| `[0:3]` | 3 | `base_angular_velocity` | pelvis IMU gyro, rad/s |
| `[3:6]` | 3 | `projected_gravity` | gravity direction in pelvis frame |
| `[6:9]` | 3 | `command` | sampled or fixed `[vx, vy, yaw_rate]` |
| `[9:38]` | 29 | `joint_position_error` | `qpos[7:] - nominal_joint_pose` |
| `[38:67]` | 29 | `joint_velocity` | actuated joint velocities, `qvel[6:]` |
| `[67:96]` | 29 | `previous_action` | previous clipped policy action |
| `[96:98]` | 2 | `gait_phase` | `[sin(2*pi*phase), cos(2*pi*phase)]`, zeroed for near-zero commands |

The actor policy receives only `state`, so the deployed policy observation is 98 floats. RSL-RL training uses an asymmetric critic observation group named `privileged_state` with 101 floats:

```text
privileged_state[0:98] = state
privileged_state[98:101] = body-frame base linear velocity [vx, vy, vz]
```

The critic-only base linear velocity is used for value learning during training and is not passed to the actor/inference policy. This keeps the deployable policy observation free of direct base linear velocity while giving the critic a cleaner velocity signal.
