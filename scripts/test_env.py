#!/usr/bin/env python3
"""Smoke-test the stochastic G1 MJX/Warp training environment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from g1_locomotion.envs import g1_config
from g1_locomotion.envs.g1_env import (
    expand_action_scales,
    expand_twist2_pd_gains,
    make_env,
)
from g1_locomotion.training.rsl_rl_wrapper import G1RSLRLBraxWrapper
from g1_locomotion.training.rsl_rl_config import default_ppo_config


def _assert_allclose(a, b, label: str) -> None:
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            raise AssertionError(f"{label}: one value is a mapping and the other is not")
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"{label}: mapping keys differ: {set(a.keys())} != {set(b.keys())}")
        for key in a:
            _assert_allclose(a[key], b[key], f"{label}[{key!r}]")
        return
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0, err_msg=label)


def _assert_close(a, b, label: str, atol: float) -> None:
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=atol, err_msg=label)


def _assert_finite(x, label: str) -> None:
    if isinstance(x, Mapping):
        for key, value in x.items():
            _assert_finite(value, f"{label}[{key!r}]")
        return
    if not bool(np.asarray(jp.isfinite(x).all())):
        raise AssertionError(f"{label} contains NaN or Inf")


def _assert_command_in_range(command, env) -> None:
    command = np.asarray(command)
    ranges = env._config.command_sampling.ranges
    if not (ranges.vx[0] <= command[0] <= ranges.vx[1]):
        raise AssertionError(f"sampled vx out of range: {command[0]}")
    if not (ranges.vy[0] <= command[1] <= ranges.vy[1]):
        raise AssertionError(f"sampled vy out of range: {command[1]}")
    if not (ranges.yaw_rate[0] <= command[2] <= ranges.yaw_rate[1]):
        raise AssertionError(f"sampled yaw_rate out of range: {command[2]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--config", type=Path, default=g1_config.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    if args.num_envs < 2:
        raise ValueError("--num-envs must be at least 2 for the duplicate-rollout determinism check")

    if jax.default_backend() != "gpu":
        raise AssertionError(f"expected JAX GPU backend, got {jax.default_backend()}")

    env = make_env(args.config)
    if env.action_size != g1_config.ACTION_SIZE:
        raise AssertionError(f"expected action size {g1_config.ACTION_SIZE}, got {env.action_size}")
    expected_obs_size = {
        "state": g1_config.OBSERVATION_SIZE,
        "privileged_state": g1_config.PRIVILEGED_OBSERVATION_SIZE,
    }
    if env.observation_size != expected_obs_size:
        raise AssertionError(f"expected observation size {expected_obs_size}, got {env.observation_size}")
    if env.mjx_model.impl != mjx.Impl.WARP:
        raise AssertionError(f"expected Warp backend, got {env.mjx_model.impl}")
    if env.mj_model.nu != 29:
        raise AssertionError(f"expected 29 actuators, got {env.mj_model.nu}")

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"MJX impl: {env.mjx_model.impl}")
    print(f"action_size: {env.action_size}")
    print(f"policy observation_size: {env.observation_size['state']}")
    print(f"critic privileged observation_size: {env.observation_size['privileged_state']}")
    print(f"sim_dt: {env._config.sim_dt} ({1.0 / env._config.sim_dt:.0f} Hz)")
    print(f"ctrl_dt: {env._config.ctrl_dt} ({1.0 / env._config.ctrl_dt:.0f} Hz)")
    print(f"n_substeps: {env.n_substeps}")

    if not np.isclose(env._config.sim_dt, 0.002):
        raise AssertionError(f"expected 0.002 sim_dt, got {env._config.sim_dt}")
    if not np.isclose(env._config.ctrl_dt, 0.02):
        raise AssertionError(f"expected 0.02 ctrl_dt, got {env._config.ctrl_dt}")
    if env.n_substeps != 10:
        raise AssertionError(f"expected 10 physics substeps, got {env.n_substeps}")
    if env.command_resampling_steps != 500:
        raise AssertionError(f"expected 500 command resampling steps, got {env.command_resampling_steps}")
    if env._config.control_contract_version != g1_config.CONTROL_CONTRACT_VERSION:
        raise AssertionError("unexpected control contract version")
    if env._config.action_scale.mode != "effort_fraction":
        raise AssertionError("default action scale must use effort_fraction mode")
    ppo_cfg = default_ppo_config()
    if ppo_cfg.algorithm.schedule != "adaptive":
        raise AssertionError("PPO learning-rate schedule must be adaptive")
    if not np.isclose(ppo_cfg.algorithm.desired_kl, 0.01):
        raise AssertionError("PPO desired KL must be 0.01")
    if not np.isclose(ppo_cfg.algorithm.learning_rate, 3.0e-4):
        raise AssertionError("PPO initial learning rate must remain 3e-4")

    expected_kp = np.array(
        [
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 40, 40, 40,
            40, 40, 40, 40, 40, 40, 40,
        ],
        dtype=np.float64,
    )
    expected_kd = np.array(
        [
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5,
        ],
        dtype=np.float64,
    )
    pd_kp, pd_kd = expand_twist2_pd_gains(env.mj_model, env._config)
    _assert_allclose(pd_kp, expected_kp, "TWIST2 Kp expansion mismatch")
    _assert_allclose(pd_kd, expected_kd, "TWIST2 Kd expansion mismatch")
    expected_action_scales = np.array(
        [
            0.22, 0.3475, 0.22, 139.0 / 600.0, 0.3125, 0.3125,
            0.22, 0.3475, 0.22, 139.0 / 600.0, 0.3125, 0.3125,
            88.0 / 600.0, 1.0 / 12.0, 1.0 / 12.0,
            0.15625, 0.15625, 0.15625, 0.15625, 0.15625, 0.03125, 0.03125,
            0.15625, 0.15625, 0.15625, 0.15625, 0.15625, 0.03125, 0.03125,
        ],
        dtype=np.float64,
    )
    computed_action_scales = expand_action_scales(env.mj_model, env._config, pd_kp)
    _assert_close(computed_action_scales, expected_action_scales, "action scale expansion mismatch", 1e-7)
    _assert_close(env.action_scales, expected_action_scales, "environment action scales mismatch", 1e-7)

    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)
    state_a = env.reset(key0)
    state_b = env.reset(key0)
    state_c = env.reset(key1)
    if "rng" not in state_a.info:
        raise AssertionError("environment state must carry an RNG key for command sampling")
    _assert_allclose(state_a.data.qpos, state_b.data.qpos, "same-seed qpos reset mismatch")
    _assert_allclose(state_a.data.qvel, state_b.data.qvel, "same-seed qvel reset mismatch")
    _assert_allclose(state_a.obs, state_b.obs, "same-seed obs reset mismatch")
    _assert_allclose(state_a.info["command"], state_b.info["command"], "same-seed command mismatch")
    if np.allclose(np.asarray(state_a.data.qpos), np.asarray(state_c.data.qpos), rtol=0.0, atol=0.0):
        raise AssertionError("different seeds unexpectedly produced identical randomized qpos")
    if np.allclose(np.asarray(state_a.data.qvel), np.asarray(state_c.data.qvel), rtol=0.0, atol=0.0):
        raise AssertionError("different seeds unexpectedly produced identical randomized qvel")
    _assert_command_in_range(state_a.info["command"], env)
    _assert_command_in_range(state_c.info["command"], env)
    deadbanded = env.apply_command_deadband(jp.array([0.1, 0.0, 0.5]))
    _assert_allclose(deadbanded, np.array([0.0, 0.0, 0.5]), "small xy command deadband mismatch")
    base_xy = np.asarray(state_a.data.qpos[:2])
    if not ((base_xy >= env._config.reset_randomization.base_xy[0] - 1e-6).all() and (base_xy <= env._config.reset_randomization.base_xy[1] + 1e-6).all()):
        raise AssertionError(f"reset base xy out of range: {base_xy}")
    root_velocity = np.asarray(state_a.data.qvel[:6])
    if not ((root_velocity >= env._config.reset_randomization.root_velocity[0] - 1e-6).all() and (root_velocity <= env._config.reset_randomization.root_velocity[1] + 1e-6).all()):
        raise AssertionError(f"reset root velocity out of range: {root_velocity}")
    nominal_pose = np.asarray(env._config.nominal_joint_pose)
    reset_pose = np.asarray(state_a.data.qpos[7:])
    nonzero_nominal = np.abs(nominal_pose) > 1e-8
    pose_scale = reset_pose[nonzero_nominal] / nominal_pose[nonzero_nominal]
    if not ((pose_scale >= env._config.reset_randomization.joint_pose_scale[0] - 1e-6).all() and (pose_scale <= env._config.reset_randomization.joint_pose_scale[1] + 1e-6).all()):
        raise AssertionError("reset joint pose scale out of range")
    _assert_close(reset_pose[~nonzero_nominal], 0.0, "zero nominal joints should stay zero after pose scaling", 1e-6)
    if not (env._config.gait_randomization.frequency[0] <= float(state_a.info["gait_frequency"]) <= env._config.gait_randomization.frequency[1]):
        raise AssertionError(f"gait frequency out of range: {state_a.info['gait_frequency']}")
    push_interval_seconds = float(state_a.info["push_interval_steps"] * env.dt)
    if not (env._config.push_randomization.interval_range[0] - env.dt <= push_interval_seconds <= env._config.push_randomization.interval_range[1] + env.dt):
        raise AssertionError(f"push interval out of range: {push_interval_seconds}")
    _assert_finite(state_a.obs, "reset observation")
    clean_reset_obs = env._get_obs(state_a.data, dict(state_a.info), add_noise=False)
    obs_noise = np.asarray(state_a.obs["state"] - clean_reset_obs["state"])
    noise_level = env._config.observation_noise.level
    noise_scales = env._config.observation_noise.scales
    if not (np.abs(obs_noise[0:3]) <= noise_level * noise_scales.gyro + 1e-6).all():
        raise AssertionError("gyro observation noise exceeds configured bounds")
    if not (np.abs(obs_noise[3:6]) <= noise_level * noise_scales.gravity + 1e-6).all():
        raise AssertionError("gravity observation noise exceeds configured bounds")
    if not (np.abs(obs_noise[9:38]) <= noise_level * noise_scales.joint_pos + 1e-6).all():
        raise AssertionError("joint position observation noise exceeds configured bounds")
    if not (np.abs(obs_noise[38:67]) <= noise_level * noise_scales.joint_vel + 1e-6).all():
        raise AssertionError("joint velocity observation noise exceeds configured bounds")
    _assert_close(obs_noise[6:9], 0.0, "command observation should not be noise-corrupted", 1e-6)
    _assert_close(obs_noise[67:98], 0.0, "action/phase observation should not be noise-corrupted", 1e-6)
    _assert_allclose(
        state_a.obs["privileged_state"][: g1_config.OBSERVATION_SIZE],
        state_a.obs["state"],
        "privileged critic obs must prefix actor obs",
    )
    _assert_close(
        state_a.obs["privileged_state"][g1_config.OBSERVATION_SIZE :],
        env._get_base_linvel_body(state_a.data),
        "critic base linear velocity mismatch",
        1e-6,
    )
    _assert_finite(state_a.reward, "reset reward")
    if not bool(np.asarray(state_a.reward != 0.0)):
        raise AssertionError("reset reward should be nonzero with tracking/alive terms")
    reward_term_names = [
        "tracking_lin_vel",
        "tracking_yaw_rate",
        "orientation",
        "alive",
        "torques",
        "action_rate",
        "dof_vel",
        "dof_acc",
        "feet_slide",
        "gait_contact",
        "feet_air_time",
        "feet_swing_height",
    ]
    for metric in [
        *[f"reward_rate/{name}" for name in reward_term_names],
        *[f"reward/{name}" for name in reward_term_names],
        "reward/total",
        "randomization/gait_frequency",
        "randomization/push_interval_steps",
        "action/applied_abs_mean",
        "action/applied_abs_max",
        "termination/nonfinite",
        "termination/height_low",
        "termination/height_high",
        "termination/tilt",
        "termination/illegal_contact",
        "termination/upright_cosine",
        "orientation/cost",
        "torque/raw_sum_sq",
        "action/rate_cost",
        "dof/vel_cost",
        "dof/acc_cost",
        "feet/slide_cost",
        "feet/air_time_reward_raw",
        "feet/swing_height_cost",
        "gait/contact_reward_raw",
        "gait/phase",
        "gait/left_phase",
        "gait/right_phase",
        "feet/left_height",
        "feet/right_height",
        "feet/left_air_time",
        "feet/right_air_time",
        "feet/left_first_contact",
        "feet/right_first_contact",
        "feet/left_contact",
        "feet/right_contact",
    ]:
        if metric not in state_a.metrics:
            raise AssertionError(f"missing reward metric {metric}")
        _assert_finite(state_a.metrics[metric], metric)

    zero_action = jp.zeros(env.action_size)
    zero_target = env.action_to_target(zero_action)
    _assert_close(zero_target, np.asarray(env._config.nominal_joint_pose), "zero-action target mismatch", 1e-6)
    nominal_data = mjx.forward(
        env.mjx_model,
        state_a.data.replace(qpos=env._reset_qpos, qvel=jp.zeros(env.mjx_model.nv)),
    )
    nominal_static_request = env._pd_kp * env.action_scales
    expected_quarter_effort = 0.25 * env._effort_uppers
    _assert_close(
        nominal_static_request,
        expected_quarter_effort,
        "unit action scale must represent 25 percent static effort",
        1e-4,
    )
    positive_target = env.action_to_target(jp.ones(env.action_size))
    positive_torque = env.compute_pd_torque(nominal_data, positive_target)
    if not bool(np.asarray((positive_torque <= expected_quarter_effort + 1e-4).all())):
        raise AssertionError("joint-limit clipping increased the static effort request")
    reset_torque = env.compute_pd_torque(state_a.data, zero_target)
    _assert_finite(reset_torque, "reset PD torque")
    if not bool(np.asarray((reset_torque >= env._effort_lowers).all() & (reset_torque <= env._effort_uppers).all())):
        raise AssertionError("reset PD torque exceeds effort limits")

    state_step = env.step(state_a, zero_action)
    _assert_finite(state_step.obs, "single-step observation")
    _assert_finite(state_step.data.qpos, "single-step qpos")
    _assert_finite(state_step.data.qvel, "single-step qvel")
    _assert_finite(state_step.reward, "single-step reward")
    _assert_finite(state_step.info["pd_torque"], "single-step PD torque")
    _assert_finite(state_step.info["mean_torque_sq_sum"], "single-step raw torque square sum")
    scaled_reward_sum = sum(
        state_step.metrics[f"reward/{name}"] for name in reward_term_names
    )
    _assert_close(
        state_step.metrics["reward/total"],
        scaled_reward_sum,
        "scaled reward components must sum to total",
        1e-6,
    )
    for name in reward_term_names:
        _assert_close(
            state_step.metrics[f"reward/{name}"],
            state_step.metrics[f"reward_rate/{name}"] * env.dt,
            f"reward timestep scaling mismatch for {name}",
            1e-6,
        )
    _assert_close(
        state_step.metrics["torque/raw_sum_sq"],
        state_step.info["mean_torque_sq_sum"],
        "raw torque square metric mismatch",
        1e-5,
    )
    _assert_close(
        state_step.metrics["reward/torques"],
        env._config.reward.torques * state_step.info["mean_torque_sq_sum"] * env.dt,
        "raw torque reward mismatch",
        1e-5,
    )
    if not np.isclose(env._config.reward.orientation, -5.0):
        raise AssertionError(f"expected orientation weight -5.0, got {env._config.reward.orientation}")
    expected_orientation_cost = jp.sum(jp.square(env._get_projected_gravity(state_step.data)[..., :2]), axis=-1)
    _assert_close(
        state_step.metrics["orientation/cost"],
        expected_orientation_cost,
        "orientation cost mismatch",
        1e-6,
    )
    _assert_close(
        state_step.metrics["reward/orientation"],
        env._config.reward.orientation * state_step.metrics["orientation/cost"] * env.dt,
        "orientation reward mismatch",
        1e-6,
    )

    if not np.isclose(env._config.reward.action_rate, -0.01):
        raise AssertionError(f"expected action_rate weight -0.01, got {env._config.reward.action_rate}")
    if not np.isclose(env._config.reward.dof_vel, -1.0e-3):
        raise AssertionError(f"expected dof_vel weight -1e-3, got {env._config.reward.dof_vel}")
    if not np.isclose(env._config.reward.dof_acc, -2.5e-7):
        raise AssertionError(f"expected dof_acc weight -2.5e-7, got {env._config.reward.dof_acc}")
    if "previous_raw_action" in state_a.info:
        raise AssertionError("raw action history must not be stored in environment state")
    expected_action_rate_cost = jp.sum(
        jp.square(zero_action - state_a.info["previous_action"])
    )
    expected_dof_vel_cost = jp.sum(jp.square(state_step.data.qvel[6:]))
    expected_dof_acc_cost = jp.sum(jp.square((state_step.data.qvel[6:] - state_a.data.qvel[6:]) / env.dt))
    _assert_close(state_step.metrics["action/rate_cost"], expected_action_rate_cost, "action-rate cost mismatch", 1e-6)
    _assert_close(
        state_step.metrics["reward/action_rate"],
        env._config.reward.action_rate * state_step.metrics["action/rate_cost"] * env.dt,
        "action-rate reward mismatch",
        1e-6,
    )
    big_action = jp.ones(env.action_size) * 2.0
    big_action_step = env.step(state_a, big_action)
    expected_big_action_rate_cost = jp.sum(
        jp.square(jp.ones(env.action_size) - state_a.info["previous_action"])
    )
    _assert_close(
        big_action_step.metrics["action/rate_cost"],
        expected_big_action_rate_cost,
        "bounded action-rate cost mismatch",
        1e-6,
    )
    _assert_close(
        big_action_step.info["previous_action"],
        jp.ones(env.action_size),
        "previous action should store the clipped/applied policy action",
        1e-6,
    )
    _assert_close(
        big_action_step.metrics["action/applied_abs_mean"],
        1.0,
        "applied action mean metric mismatch",
        1e-6,
    )
    _assert_close(
        big_action_step.metrics["action/applied_abs_max"],
        1.0,
        "applied action max metric mismatch",
        1e-6,
    )
    max_rate_info = dict(state_a.info)
    max_rate_info["previous_action"] = -jp.ones(env.action_size)
    max_rate_state = state_a.replace(info=max_rate_info)
    max_rate_step = env.step(max_rate_state, big_action)
    _assert_close(max_rate_step.metrics["action/rate_cost"], 116.0, "maximum action-rate cost", 1e-6)
    if float(max_rate_step.metrics["action/rate_cost"]) > 116.0:
        raise AssertionError("bounded action-rate cost exceeded 116")
    _assert_close(state_step.metrics["dof/vel_cost"], expected_dof_vel_cost, "dof velocity cost mismatch", 1e-5)
    _assert_close(state_step.metrics["reward/dof_vel"], env._config.reward.dof_vel * state_step.metrics["dof/vel_cost"] * env.dt, "dof velocity reward mismatch", 1e-5)
    _assert_close(state_step.metrics["dof/acc_cost"], expected_dof_acc_cost, "dof acceleration cost mismatch", 1e-2)
    _assert_close(state_step.metrics["reward/dof_acc"], env._config.reward.dof_acc * state_step.metrics["dof/acc_cost"] * env.dt, "dof acceleration reward mismatch", 1e-5)

    if not np.isclose(env._config.reward.feet_slide, -0.2):
        raise AssertionError(f"expected feet_slide weight -0.2, got {env._config.reward.feet_slide}")
    expected_feet_slide_cost = jp.sum(
        jp.linalg.norm(env._get_foot_global_linvel(state_step.data)[..., :2], axis=-1)
        * env._get_foot_contacts(state_step.data),
        axis=-1,
    )
    _assert_close(
        state_step.metrics["feet/slide_cost"],
        expected_feet_slide_cost,
        "feet slide cost mismatch",
        1e-6,
    )
    _assert_close(
        state_step.metrics["reward/feet_slide"],
        env._config.reward.feet_slide * state_step.metrics["feet/slide_cost"] * env.dt,
        "feet slide reward mismatch",
        1e-6,
    )
    if not np.isclose(env._config.reward.gait_contact, 0.5):
        raise AssertionError(f"expected gait_contact weight 0.5, got {env._config.reward.gait_contact}")
    if not np.isclose(env._config.gait.period, 0.8):
        raise AssertionError(f"expected gait period 0.8, got {env._config.gait.period}")
    if not np.isclose(env._config.gait.stance_threshold, 0.55):
        raise AssertionError(f"expected gait stance threshold 0.55, got {env._config.gait.stance_threshold}")
    expected_phase = np.mod(0.0 + state_a.info["gait_phase_dt"], 1.0)
    _assert_close(state_step.info["gait_phase"], expected_phase, "gait phase step mismatch", 1e-6)
    _assert_close(state_step.metrics["gait/phase"], expected_phase, "gait phase metric mismatch", 1e-6)
    _assert_close(
        state_step.metrics["reward/gait_contact"],
        env._config.reward.gait_contact * state_step.metrics["gait/contact_reward_raw"] * env.dt,
        "gait contact reward mismatch",
        1e-6,
    )
    gait_match_example = env._reward_gait_contact(
        jp.array([1.0, 0.0]),
        jp.array(0.30),
        jp.array([0.2, 0.0, 0.0]),
    )
    _assert_close(gait_match_example, 2.0, "gait contact match formula mismatch", 1e-6)
    no_command_gait = env._reward_gait_contact(
        jp.array([1.0, 0.0]),
        jp.array(0.30),
        jp.array([0.0, 0.0, 0.0]),
    )
    _assert_close(no_command_gait, 0.0, "gait contact zero-command mask mismatch", 1e-6)
    expected_phase_obs = env._get_gait_phase_observation(state_step.info["gait_phase"], state_step.info["command"])
    _assert_close(state_step.obs["state"][-2:], expected_phase_obs, "gait phase observation mismatch", 1e-6)

    if not np.isclose(env._config.reward.feet_air_time, 0.0):
        raise AssertionError(f"expected feet_air_time weight 0.0, got {env._config.reward.feet_air_time}")
    _assert_close(
        state_step.metrics["reward/feet_air_time"],
        env._config.reward.feet_air_time * state_step.metrics["feet/air_time_reward_raw"] * env.dt,
        "feet air-time reward mismatch",
        1e-6,
    )
    air_time_example = env._reward_feet_air_time(
        jp.array([0.3, 0.6]),
        jp.array([1.0, 1.0]),
        jp.array([0.2, 0.0, 0.0]),
    )
    _assert_close(air_time_example, 0.4, "feet air-time threshold/cap mismatch", 1e-6)
    no_command_air_time = env._reward_feet_air_time(
        jp.array([0.3, 0.6]),
        jp.array([1.0, 1.0]),
        jp.array([0.0, 0.0, 0.0]),
    )
    _assert_close(no_command_air_time, 0.0, "feet air-time zero-command mask mismatch", 1e-6)
    if not np.isclose(env._config.reward.feet_swing_height, -20.0):
        raise AssertionError(f"expected feet_swing_height weight -20.0, got {env._config.reward.feet_swing_height}")
    expected_swing_height_cost = env._reward_feet_swing_height(
        state_step.data,
        env._get_foot_contacts(state_step.data),
        state_step.info["command"],
    )
    _assert_close(
        state_step.metrics["feet/swing_height_cost"],
        expected_swing_height_cost,
        "feet swing-height cost mismatch",
        1e-6,
    )
    _assert_close(
        state_step.metrics["reward/feet_swing_height"],
        env._config.reward.feet_swing_height * state_step.metrics["feet/swing_height_cost"] * env.dt,
        "feet swing-height reward mismatch",
        1e-6,
    )
    no_command_swing_height = env._reward_feet_swing_height(
        state_step.data,
        env._get_foot_contacts(state_step.data),
        jp.array([0.0, 0.0, 0.0]),
    )
    _assert_close(no_command_swing_height, 0.0, "feet swing-height zero-command mask mismatch", 1e-6)
    if not bool(
        np.asarray(
            (state_step.info["pd_torque"] >= env._effort_lowers).all()
            & (state_step.info["pd_torque"] <= env._effort_uppers).all()
        )
    ):
        raise AssertionError("single-step PD torque exceeds effort limits")

    forced_push_info = dict(state_a.info)
    forced_push_info["push_step"] = state_a.info["push_interval_steps"] - 1
    forced_push_state = state_a.replace(info=forced_push_info)
    forced_push_step = env.step(forced_push_state, zero_action)
    push_magnitude = float(forced_push_step.info["push_magnitude"])
    if not (env._config.push_randomization.magnitude_range[0] <= push_magnitude <= env._config.push_randomization.magnitude_range[1]):
        raise AssertionError(f"forced push magnitude out of range: {push_magnitude}")
    _assert_finite(forced_push_step.info["push"], "forced push vector")

    info_no_resample = dict(state_a.info)
    info_no_resample["step"] = jp.array(env.command_resampling_steps - 2, dtype=jp.int32)
    state_no_resample_seed = state_a.replace(info=info_no_resample)
    state_no_resample = env.step(state_no_resample_seed, zero_action)
    _assert_allclose(
        state_no_resample.info["command"],
        state_a.info["command"],
        "command resampled one step too early",
    )
    info_resample = dict(state_a.info)
    info_resample["step"] = jp.array(env.command_resampling_steps - 1, dtype=jp.int32)
    state_resample_seed = state_a.replace(info=info_resample)
    state_resample = env.step(state_resample_seed, zero_action)
    if np.allclose(
        np.asarray(state_resample.info["command"]),
        np.asarray(state_a.info["command"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("command did not resample at the configured interval")
    _assert_command_in_range(state_resample.info["command"], env)

    clean_sensordata = state_a.data.sensordata
    for sensor_name in env._illegal_contact_sensor_names:
        sensor_id = env.mj_model.sensor(sensor_name).id
        sensor_adr = int(env.mj_model.sensor_adr[sensor_id])
        sensor_dim = int(env.mj_model.sensor_dim[sensor_id])
        clean_sensordata = clean_sensordata.at[sensor_adr : sensor_adr + sensor_dim].set(0.0)
    clean_data = state_a.data.replace(sensordata=clean_sensordata)
    clean_obs = env._get_obs(clean_data, dict(state_a.info), add_noise=False)
    clean_done, clean_terms = env._get_done(clean_data, clean_obs)
    if bool(np.asarray(clean_done)):
        raise AssertionError(f"valid upright state unexpectedly terminated: {clean_terms}")

    low_qpos = clean_data.qpos.at[2].set(env._config.termination.min_base_height - 0.01)
    high_qpos = clean_data.qpos.at[2].set(env._config.termination.max_base_height + 0.01)
    low_data = clean_data.replace(qpos=low_qpos)
    high_data = clean_data.replace(qpos=high_qpos)
    low_done, low_terms = env._get_done(low_data, clean_obs)
    high_done, high_terms = env._get_done(high_data, clean_obs)
    if not bool(np.asarray(low_done == 1.0)):
        raise AssertionError("Wisconsin low-height termination did not trigger")
    if not bool(np.asarray(high_done == 1.0)):
        raise AssertionError("Wisconsin high-height termination did not trigger")
    if not bool(np.asarray(low_terms["termination/height_low"] == 1.0)):
        raise AssertionError("low-height termination reason was not recorded")
    if not bool(np.asarray(high_terms["termination/height_high"] == 1.0)):
        raise AssertionError("high-height termination reason was not recorded")

    tilt_radians = np.deg2rad(env._config.termination.max_tilt_degrees + 1.0)
    tilt_quat = jp.array([
        np.cos(tilt_radians / 2.0),
        np.sin(tilt_radians / 2.0),
        0.0,
        0.0,
    ])
    tilt_qpos = clean_data.qpos.at[3:7].set(tilt_quat)
    tilt_data = mjx.forward(env.mjx_model, clean_data.replace(qpos=tilt_qpos))
    tilt_obs = env._get_obs(tilt_data, dict(state_a.info), add_noise=False)
    tilt_done, tilt_terms = env._get_done(tilt_data, tilt_obs)
    if not bool(np.asarray(tilt_done == 1.0)) or not bool(
        np.asarray(tilt_terms["termination/tilt"] == 1.0)
    ):
        raise AssertionError("70-degree tilt termination did not trigger")

    for sensor_name in env._illegal_contact_sensor_names:
        sensor_id = env.mj_model.sensor(sensor_name).id
        sensor_adr = int(env.mj_model.sensor_adr[sensor_id])
        contact_data = clean_data.replace(
            sensordata=clean_sensordata.at[sensor_adr].set(1.0)
        )
        contact_done, contact_terms = env._get_done(contact_data, clean_obs)
        if not bool(np.asarray(contact_done == 1.0)) or not bool(
            np.asarray(contact_terms["termination/illegal_contact"] == 1.0)
        ):
            raise AssertionError(f"illegal contact termination failed for {sensor_name}")

    nonfinite_data = clean_data.replace(qpos=clean_data.qpos.at[0].set(jp.nan))
    nonfinite_done, nonfinite_terms = env._get_done(nonfinite_data, clean_obs)
    if not bool(np.asarray(nonfinite_done == 1.0)) or not bool(
        np.asarray(nonfinite_terms["termination/nonfinite"] == 1.0)
    ):
        raise AssertionError("nonfinite termination did not trigger")

    reset_jit = jax.jit(jax.vmap(env.reset))
    step_jit = jax.jit(jax.vmap(env.step))
    keys = jp.repeat(jax.random.PRNGKey(42)[None, :], args.num_envs, axis=0)
    batched_state = reset_jit(keys)
    batched_actions = jp.zeros((args.num_envs, env.action_size))
    _assert_finite(batched_state.obs, "batched reset observation")

    batched_one_step = step_jit(batched_state, batched_actions)
    _assert_finite(batched_one_step.obs, "batched one-step observation")
    _assert_finite(batched_one_step.data.qpos, "batched one-step qpos")
    _assert_finite(batched_one_step.data.qvel, "batched one-step qvel")
    _assert_finite(batched_one_step.reward, "batched one-step reward")

    rollout_a = batched_one_step
    for _ in range(args.steps):
        rollout_a = step_jit(rollout_a, batched_actions)
        _assert_finite(rollout_a.obs, "batched rollout observation")
        _assert_finite(rollout_a.data.qpos, "batched rollout qpos")
        _assert_finite(rollout_a.data.qvel, "batched rollout qvel")

    import torch

    full_reset_wrapper = G1RSLRLBraxWrapper(
        env,
        num_actors=2,
        seed=123,
        episode_length=env._config.episode_length,
        action_repeat=1,
        randomization_fn=None,
        device_rank=0,
    )
    full_reset_wrapper.reset()
    forced_state = full_reset_wrapper.env_state
    forced_info = dict(forced_state.info)
    forced_info["previous_action"] = jp.ones_like(forced_info["previous_action"])
    forced_info["step"] = jp.full_like(forced_info["step"], 123)
    forced_info["feet_air_time"] = jp.ones_like(forced_info["feet_air_time"])
    forced_info["gait_phase"] = jp.ones_like(forced_info["gait_phase"]) * 0.75
    forced_info["gait_frequency"] = jp.ones_like(forced_info["gait_frequency"]) * 9.0
    forced_info["gait_phase_dt"] = jp.ones_like(forced_info["gait_phase_dt"]) * 9.0
    forced_info["command"] = jp.ones_like(forced_info["command"]) * 9.0
    forced_info["push"] = jp.ones_like(forced_info["push"]) * 9.0
    forced_info["push_magnitude"] = jp.ones_like(forced_info["push_magnitude"]) * 9.0
    forced_info["push_step"] = jp.full_like(forced_info["push_step"], 99)
    forced_info["push_interval_steps"] = jp.ones_like(
        forced_info["push_interval_steps"]
    )
    forced_info["target_qpos"] = jp.ones_like(forced_info["target_qpos"]) * 9.0
    forced_info["pd_torque"] = jp.ones_like(forced_info["pd_torque"]) * 9.0
    forced_info["mean_torque_sq_sum"] = jp.ones_like(
        forced_info["mean_torque_sq_sum"]
    ) * 9.0
    forced_qpos = forced_state.data.qpos.at[:, 2].set(
        env._config.termination.min_base_height - 0.4
    )
    forced_qvel = jp.ones_like(forced_state.data.qvel) * 20.0
    forced_qvel = forced_qvel.at[:, 2].set(-20.0)
    full_reset_wrapper.env_state = forced_state.replace(
        data=forced_state.data.replace(qpos=forced_qpos, qvel=forced_qvel),
        info=forced_info,
    )
    raw_torch_action = torch.full(
        (2, env.action_size), 2.0, device="cuda:0", dtype=torch.float32
    )
    _obs, _reward, reset_done, reset_info = full_reset_wrapper.step(raw_torch_action)
    if not bool(torch.all(reset_done > 0)):
        raise AssertionError("forced terminal states did not report done")
    reset_state_info = full_reset_wrapper.env_state.info
    _assert_close(reset_state_info["previous_action"], 0.0, "full reset previous action", 1e-6)
    _assert_close(reset_state_info["step"], 0.0, "full reset environment step", 1e-6)
    _assert_close(reset_state_info["feet_air_time"], 0.0, "full reset feet air time", 1e-6)
    _assert_close(reset_state_info["gait_phase"], 0.0, "full reset gait phase", 1e-6)
    _assert_close(reset_state_info["push_step"], 0.0, "full reset push step", 1e-6)
    _assert_close(reset_state_info["push"], 0.0, "full reset push vector", 1e-6)
    _assert_close(
        reset_state_info["push_magnitude"], 0.0, "full reset push magnitude", 1e-6
    )
    _assert_close(reset_state_info["pd_torque"], 0.0, "full reset PD torque", 1e-6)
    _assert_close(
        reset_state_info["mean_torque_sq_sum"],
        0.0,
        "full reset torque accumulator",
        1e-6,
    )
    _assert_close(
        reset_state_info["target_qpos"],
        jp.broadcast_to(
            env._nominal_joint_pose, reset_state_info["target_qpos"].shape
        ),
        "full reset target pose",
        1e-6,
    )
    reset_commands = np.asarray(reset_state_info["command"])
    if np.any(np.isclose(reset_commands, 9.0)):
        raise AssertionError("full reset retained the pre-termination command")
    reset_gait_frequency = np.asarray(reset_state_info["gait_frequency"])
    if not np.all(
        (reset_gait_frequency >= env._config.gait_randomization.frequency[0])
        & (reset_gait_frequency <= env._config.gait_randomization.frequency[1])
    ):
        raise AssertionError("full reset gait frequency is outside its configured range")
    reset_push_steps = np.asarray(reset_state_info["push_interval_steps"])
    push_step_bounds = np.asarray(env._config.push_randomization.interval_range) / env.dt
    if not np.all(
        (reset_push_steps >= np.floor(push_step_bounds[0]))
        & (reset_push_steps <= np.ceil(push_step_bounds[1]))
    ):
        raise AssertionError("full reset push interval is outside its configured range")
    _assert_close(
        _obs["state"].detach().cpu().numpy(),
        full_reset_wrapper.env_state.obs["state"],
        "full reset returned observation mismatch",
        1e-6,
    )
    if float(jp.max(jp.abs(full_reset_wrapper.env_state.data.qvel[:, :6]))) > (
        env._config.reset_randomization.root_velocity[1] + 1e-5
    ):
        raise AssertionError("full reset retained pre-termination root velocity")
    _assert_close(
        full_reset_wrapper.env_state.data.qvel[:, 6:],
        0.0,
        "full reset retained pre-termination joint velocity",
        1e-6,
    )
    _assert_close(reset_info["log"]["action/raw_abs_mean"], 2.0, "wrapper raw action mean", 1e-6)
    _assert_close(reset_info["log"]["action/raw_abs_max"], 2.0, "wrapper raw action max", 1e-6)
    _assert_close(reset_info["log"]["action/clip_fraction"], 1.0, "wrapper clip fraction", 1e-6)

    print(f"single env step: OK")
    print(f"batched envs: OK ({args.num_envs})")
    print(f"smoke steps: OK ({args.steps})")
    print("TWIST2 PD gains: OK")
    print("actuator-aware action scales: OK")
    print("command sampling: OK")
    print("Playground-style reset randomization: OK")
    print("Playground-style observation noise: OK")
    print("Playground-style gait frequency randomization: OK")
    print("Playground-style pushes: OK")
    print("repo-style rewards: OK")
    print("asymmetric critic observation: OK")
    print("orientation penalty: OK")
    print("action-rate penalty: OK")
    print("full episode-state reset: OK")
    print("dof velocity penalty: OK")
    print("dof acceleration penalty: OK")
    print("feet slide penalty: OK")
    print("gait contact reward: OK")
    print("gait phase observation: OK")
    print("feet air-time reward: OK")
    print("feet swing-height reward: OK")
    print("height/tilt/contact termination: OK")
    print("torque limits: OK")
    print("determinism: OK")
    print("finite observations/state: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
