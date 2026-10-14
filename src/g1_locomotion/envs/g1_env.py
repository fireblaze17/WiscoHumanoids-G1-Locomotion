"""Unitree G1 flat-ground MJX/Warp training environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from g1_locomotion.envs import g1_config
from g1_locomotion.envs import observations


def _twist2_gain_key(joint_name: str) -> str:
    if "hip_yaw" in joint_name:
        return "hip_yaw"
    if "hip_roll" in joint_name:
        return "hip_roll"
    if "hip_pitch" in joint_name:
        return "hip_pitch"
    if "knee" in joint_name:
        return "knee"
    if "ankle" in joint_name:
        return "ankle"
    if "waist" in joint_name:
        return "waist"
    if "shoulder" in joint_name:
        return "shoulder"
    if "elbow" in joint_name:
        return "elbow"
    if "wrist" in joint_name:
        return "wrist"
    raise ValueError(f"no TWIST2 PD gain bucket matches joint {joint_name!r}")


def _actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"actuator {actuator_id} is attached to an unnamed joint")
        names.append(joint_name)
    return names


def expand_twist2_pd_gains(
    model: mujoco.MjModel,
    cfg: config_dict.ConfigDict,
) -> tuple[np.ndarray, np.ndarray]:
    stiffness = []
    damping = []
    for joint_name in _actuated_joint_names(model):
        key = _twist2_gain_key(joint_name)
        stiffness.append(float(cfg.pd_stiffness[key]))
        damping.append(float(cfg.pd_damping[key]))
    return np.array(stiffness, dtype=np.float64), np.array(damping, dtype=np.float64)


def expand_action_scales(
    model: mujoco.MjModel,
    cfg: config_dict.ConfigDict,
    pd_kp: np.ndarray,
) -> np.ndarray:
    """Expands the configured policy action scale in actuator order."""
    if cfg.action_scale.mode == "uniform":
        return np.full(model.nu, float(cfg.action_scale.uniform), dtype=np.float64)

    effort_limits = np.minimum(
        np.abs(model.actuator_forcerange[:, 0]),
        np.abs(model.actuator_forcerange[:, 1]),
    )
    if np.any(effort_limits <= 0.0) or np.any(pd_kp <= 0.0):
        raise ValueError("effort-fraction action scaling requires positive effort limits and Kp")
    return float(cfg.action_scale.effort_fraction) * effort_limits / pd_kp


def _prepare_model(model: mujoco.MjModel, cfg: config_dict.ConfigDict) -> None:
    """Applies fixed environment model settings to a compiled MjModel."""
    model.opt.timestep = float(cfg.sim_dt)

    floor_id = model.geom("floor").id
    model.geom_friction[floor_id] = np.array(cfg.floor_friction, dtype=np.float64)

    if model.nsensor:
        model.sensor_noise[:] = 0.0

    if model.nu != g1_config.ACTION_SIZE:
        raise ValueError(f"expected {g1_config.ACTION_SIZE} actuators, got {model.nu}")

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = 1.0

        model.actuator_ctrlrange[actuator_id] = model.jnt_actfrcrange[joint_id]
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = model.jnt_actfrcrange[joint_id]
        model.actuator_forcelimited[actuator_id] = 1


class G1FlatEnv(mjx_env.MjxEnv):
    """A flat-ground G1 training environment with torque-limited PD actions."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[dict[str, object]] = None,
    ) -> None:
        cfg = config or g1_config.load_config()
        super().__init__(cfg, config_overrides)

        self._xml_path = str(Path(self._config.xml_path).resolve())
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        _prepare_model(self._mj_model, self._config)

        self._mjx_model = mjx_env.put_model(
            self._mj_model,
            impl=self._config.impl,
        )
        if self._mjx_model.impl != mjx.Impl.WARP:
            raise RuntimeError(f"expected MJX Warp backend, got {self._mjx_model.impl}")

        self._nominal_joint_pose = jp.array(self._config.nominal_joint_pose)
        self._command = jp.array(self._config.command)
        self._reset_qpos = jp.concatenate(
            [
                jp.array(self._config.reset_base_position),
                jp.array(self._config.reset_base_quaternion),
                self._nominal_joint_pose,
            ]
        )
        self._reset_qvel = jp.zeros(self._mjx_model.nv)
        self._joint_lowers = jp.array(self._mj_model.jnt_range[1:, 0])
        self._joint_uppers = jp.array(self._mj_model.jnt_range[1:, 1])
        pd_kp, pd_kd = expand_twist2_pd_gains(self._mj_model, self._config)
        self._pd_kp = jp.array(pd_kp)
        self._pd_kd = jp.array(pd_kd)
        self._effort_lowers = jp.array(self._mj_model.actuator_forcerange[:, 0])
        self._effort_uppers = jp.array(self._mj_model.actuator_forcerange[:, 1])
        self._action_scales = jp.array(
            expand_action_scales(self._mj_model, self._config, pd_kp)
        )
        self._obs_ids = observations.ObservationIds(
            pelvis_imu_site_id=self._mj_model.site("imu_in_pelvis").id
        )
        self._foot_floor_sensor_names = ("left_foot_floor_found", "right_foot_floor_found")
        self._foot_linvel_sensor_names = ("left_foot_global_linvel", "right_foot_global_linvel")
        self._foot_site_ids = jp.array([
            self._mj_model.site("left_foot").id,
            self._mj_model.site("right_foot").id,
        ])
        self._illegal_contact_sensor_names = tuple(
            self._config.termination.illegal_contact_sensors
        )
        for sensor_name in self._illegal_contact_sensor_names:
            self._mj_model.sensor(sensor_name)

    def action_to_target(self, action: jax.Array) -> jax.Array:
        clipped_action = jp.clip(action, -1.0, 1.0)
        targets = self._nominal_joint_pose + clipped_action * self._action_scales
        return jp.clip(targets, self._joint_lowers, self._joint_uppers)

    def compute_pd_torque(self, data: mjx.Data, target_qpos: jax.Array) -> jax.Array:
        torque = self._pd_kp * (target_qpos - data.qpos[7:]) - self._pd_kd * data.qvel[6:]
        return jp.clip(torque, self._effort_lowers, self._effort_uppers)

    def _step_pd(self, data: mjx.Data, target_qpos: jax.Array) -> tuple[mjx.Data, jax.Array]:
        def single_step(data: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
            torque = self.compute_pd_torque(data, target_qpos)
            data = data.replace(ctrl=torque)
            data = mjx.step(self.mjx_model, data)
            return data, jp.sum(jp.square(torque))

        data, torque_sq_sums = jax.lax.scan(single_step, data, (), self.n_substeps)
        return data, jp.mean(torque_sq_sums)

    def _yaw_quat(self, yaw: jax.Array) -> jax.Array:
        half_yaw = 0.5 * yaw
        return jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)]).reshape((4,))

    def _random_uniform(self, rng: jax.Array, shape: tuple[int, ...], bounds: list[float]) -> jax.Array:
        return jax.random.uniform(rng, shape, minval=bounds[0], maxval=bounds[1])

    def sample_command(self, rng: jax.Array) -> jax.Array:
        if not self._config.command_sampling.enabled:
            return self._command
        rng_x, rng_y, rng_yaw, rng_active = jax.random.split(rng, 4)
        ranges = self._config.command_sampling.ranges
        command = jp.array([
            jax.random.uniform(rng_x, (), minval=ranges.vx[0], maxval=ranges.vx[1]),
            jax.random.uniform(rng_y, (), minval=ranges.vy[0], maxval=ranges.vy[1]),
            jax.random.uniform(rng_yaw, (), minval=ranges.yaw_rate[0], maxval=ranges.yaw_rate[1]),
        ])
        if self._config.command_sampling.axis_sparsity.enabled:
            active_probability = jp.array(self._config.command_sampling.axis_sparsity.active_probability)
            amplitude = jp.array(self._config.command_sampling.axis_sparsity.amplitude)
            active = jax.random.bernoulli(rng_active, active_probability).astype(jp.float32)
            command = command * amplitude * active
        return self.apply_command_deadband(command)

    def apply_command_deadband(self, command: jax.Array) -> jax.Array:
        xy = command[:2]
        xy = jp.where(
            jp.linalg.norm(xy) > self._config.command_sampling.zero_small_xy_threshold,
            xy,
            jp.zeros(2),
        )
        return jp.array([xy[0], xy[1], command[2]])

    def _get_base_linvel_body(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._obs_ids.pelvis_imu_site_id].T @ data.qvel[:3]

    def _get_projected_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._obs_ids.pelvis_imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def _get_base_angvel_body(self, data: mjx.Data) -> jax.Array:
        return observations.get_sensor_data(self.mj_model, data, "imu-pelvis-angular-velocity")

    def _get_foot_contacts(self, data: mjx.Data) -> jax.Array:
        contacts = []
        for name in self._foot_floor_sensor_names:
            contact = observations.get_sensor_data(self.mj_model, data, name) > 0.0
            if contact.shape and contact.shape[-1] == 1:
                contact = jp.squeeze(contact, axis=-1)
            contacts.append(contact)
        return jp.stack(contacts, axis=-1).astype(jp.float32)

    def _get_foot_global_linvel(self, data: mjx.Data) -> jax.Array:
        velocities = [
            observations.get_sensor_data(self.mj_model, data, name)
            for name in self._foot_linvel_sensor_names
        ]
        return jp.stack(velocities, axis=-2)

    def _reward_feet_air_time(
        self,
        feet_air_time: jax.Array,
        first_contact: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        air_time = jp.clip(
            feet_air_time - self._config.reward.feet_air_time_min,
            min=0.0,
            max=self._config.reward.feet_air_time_max - self._config.reward.feet_air_time_min,
        )
        moving_command = (
            jp.linalg.norm(command) > self._config.reward.feet_air_time_command_threshold
        ).astype(jp.float32)
        return jp.sum(air_time * first_contact) * moving_command

    def _reward_feet_swing_height(
        self,
        data: mjx.Data,
        foot_contacts: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        foot_heights = data.site_xpos[self._foot_site_ids, 2]
        airborne = 1.0 - foot_contacts
        height_error = jp.square(foot_heights - self._config.reward.feet_swing_height_target)
        moving_command = (
            jp.linalg.norm(command) > self._config.reward.feet_swing_height_command_threshold
        ).astype(jp.float32)
        return jp.sum(height_error * airborne) * moving_command

    def _get_leg_phase(self, gait_phase: jax.Array) -> jax.Array:
        return jp.mod(gait_phase + jp.array(self._config.gait.offsets), 1.0)

    def _get_gait_phase_observation(self, gait_phase: jax.Array, command: jax.Array) -> jax.Array:
        active = (jp.linalg.norm(command) > self._config.gait.command_threshold).astype(jp.float32)
        angle = 2.0 * jp.pi * gait_phase
        return jp.array([jp.sin(angle), jp.cos(angle)]) * active

    def _reward_gait_contact(
        self,
        foot_contacts: jax.Array,
        gait_phase: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        if not self._config.gait.enabled:
            return jp.array(0.0)
        leg_phase = self._get_leg_phase(gait_phase)
        is_stance = leg_phase < self._config.gait.stance_threshold
        is_contact = foot_contacts > 0.0
        matches = jp.logical_not(jp.logical_xor(is_stance, is_contact)).astype(jp.float32)
        moving_command = (jp.linalg.norm(command) > self._config.gait.command_threshold).astype(jp.float32)
        return jp.sum(matches) * moving_command

    def _tree_all_finite(self, value: Any) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(value)
        finite = jp.array(True)
        for leaf in leaves:
            finite = finite & jp.isfinite(leaf).all()
        return finite

    def _get_illegal_contact(self, data: mjx.Data) -> jax.Array:
        illegal_contact = jp.array(False)
        for sensor_name in self._illegal_contact_sensor_names:
            illegal_contact = illegal_contact | jp.any(
                observations.get_sensor_data(self.mj_model, data, sensor_name) > 0.0
            )
        return illegal_contact

    def _get_done(
        self, data: mjx.Data, obs: Any
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all() & self._tree_all_finite(obs)
        too_low = data.qpos[2] < self._config.termination.min_base_height
        too_high = data.qpos[2] > self._config.termination.max_base_height
        upright_cosine = -self._get_projected_gravity(data)[2]
        minimum_upright_cosine = jp.cos(
            jp.deg2rad(self._config.termination.max_tilt_degrees)
        )
        excessive_tilt = upright_cosine < minimum_upright_cosine
        illegal_contact = self._get_illegal_contact(data)
        nonfinite = ~finite
        done = nonfinite | too_low | too_high | excessive_tilt | illegal_contact
        terms = {
            "termination/nonfinite": nonfinite.astype(jp.float32),
            "termination/height_low": too_low.astype(jp.float32),
            "termination/height_high": too_high.astype(jp.float32),
            "termination/tilt": excessive_tilt.astype(jp.float32),
            "termination/illegal_contact": illegal_contact.astype(jp.float32),
            "termination/upright_cosine": upright_cosine,
        }
        return done.astype(jp.float32), terms

    def _get_reward(
        self,
        data: mjx.Data,
        command: jax.Array,
        mean_torque_sq_sum: jax.Array,
        done: jax.Array,
        feet_air_time: jax.Array,
        first_contact: jax.Array,
        action_rate_cost: jax.Array,
        dof_acc_cost: jax.Array,
        gait_phase: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        linvel = self._get_base_linvel_body(data)
        angvel = self._get_base_angvel_body(data)
        lin_vel_error = jp.sum(jp.square(command[:2] - linvel[:2]))
        yaw_rate_error = jp.square(command[2] - angvel[2])
        tracking_lin_vel = jp.exp(-lin_vel_error / self._config.reward.tracking_sigma)
        tracking_yaw_rate = jp.exp(-yaw_rate_error / self._config.reward.tracking_sigma)
        projected_gravity = self._get_projected_gravity(data)
        orientation_cost = jp.sum(jp.square(projected_gravity[..., :2]), axis=-1)
        alive = 1.0 - done
        foot_contacts = self._get_foot_contacts(data)
        foot_linvel = self._get_foot_global_linvel(data)
        feet_slide_cost = jp.sum(jp.linalg.norm(foot_linvel[..., :2], axis=-1) * foot_contacts, axis=-1)
        feet_air_time_reward_raw = self._reward_feet_air_time(feet_air_time, first_contact, command)
        feet_swing_height_cost = self._reward_feet_swing_height(data, foot_contacts, command)
        gait_contact_reward_raw = self._reward_gait_contact(foot_contacts, gait_phase, command)
        dof_vel_cost = jp.sum(jp.square(data.qvel[6:]))
        reward_rates = {
            "tracking_lin_vel": self._config.reward.tracking_lin_vel * tracking_lin_vel,
            "tracking_yaw_rate": self._config.reward.tracking_yaw_rate * tracking_yaw_rate,
            "orientation": self._config.reward.orientation * orientation_cost,
            "alive": self._config.reward.alive * alive,
            "torques": self._config.reward.torques * mean_torque_sq_sum,
            "action_rate": self._config.reward.action_rate * action_rate_cost,
            "dof_vel": self._config.reward.dof_vel * dof_vel_cost,
            "dof_acc": self._config.reward.dof_acc * dof_acc_cost,
            "feet_slide": self._config.reward.feet_slide * feet_slide_cost,
            "gait_contact": self._config.reward.gait_contact * gait_contact_reward_raw,
            "feet_air_time": self._config.reward.feet_air_time * feet_air_time_reward_raw,
            "feet_swing_height": self._config.reward.feet_swing_height * feet_swing_height_cost,
        }
        reward_contributions = {
            name: value * self.dt for name, value in reward_rates.items()
        }
        reward = sum(reward_contributions.values())
        metrics = {
            **{f"reward_rate/{name}": value for name, value in reward_rates.items()},
            **{f"reward/{name}": value for name, value in reward_contributions.items()},
            "reward/total": reward,
            "tracking/lin_vel_error": lin_vel_error,
            "tracking/yaw_rate_error": yaw_rate_error,
            "orientation/cost": orientation_cost,
            "torque/raw_sum_sq": mean_torque_sq_sum,
            "action/rate_cost": action_rate_cost,
            "dof/vel_cost": dof_vel_cost,
            "dof/acc_cost": dof_acc_cost,
            "feet/slide_cost": feet_slide_cost,
            "feet/air_time_reward_raw": feet_air_time_reward_raw,
            "feet/swing_height_cost": feet_swing_height_cost,
            "gait/contact_reward_raw": gait_contact_reward_raw,
            "gait/phase": gait_phase,
            "gait/left_phase": self._get_leg_phase(gait_phase)[0],
            "gait/right_phase": self._get_leg_phase(gait_phase)[1],
            "feet/left_height": data.site_xpos[self._foot_site_ids[0], 2],
            "feet/right_height": data.site_xpos[self._foot_site_ids[1], 2],
            "feet/left_air_time": feet_air_time[..., 0],
            "feet/right_air_time": feet_air_time[..., 1],
            "feet/left_first_contact": first_contact[..., 0],
            "feet/right_first_contact": first_contact[..., 1],
            "feet/left_contact": foot_contacts[..., 0],
            "feet/right_contact": foot_contacts[..., 1],
        }
        return reward, metrics

    @property
    def command_resampling_steps(self) -> int:
        return int(round(self._config.command_sampling.resampling_time / self._config.ctrl_dt))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        (
            rng,
            command_rng,
            base_xy_rng,
            base_yaw_rng,
            joint_scale_rng,
            root_velocity_rng,
            gait_freq_rng,
            push_interval_rng,
        ) = jax.random.split(rng, 8)
        command = self.sample_command(command_rng)
        qpos = self._reset_qpos
        qvel = self._reset_qvel
        if self._config.reset_randomization.enabled:
            dxy = self._random_uniform(base_xy_rng, (2,), self._config.reset_randomization.base_xy)
            yaw = self._random_uniform(base_yaw_rng, (), self._config.reset_randomization.base_yaw)
            joint_scale = self._random_uniform(
                joint_scale_rng,
                (self.action_size,),
                self._config.reset_randomization.joint_pose_scale,
            )
            root_velocity = self._random_uniform(
                root_velocity_rng,
                (6,),
                self._config.reset_randomization.root_velocity,
            )
            qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
            qpos = qpos.at[3:7].set(self._yaw_quat(yaw))
            qpos = qpos.at[7:].set(qpos[7:] * joint_scale)
            qvel = qvel.at[:6].set(root_velocity)
        gait_frequency = jp.where(
            self._config.gait_randomization.enabled,
            self._random_uniform(gait_freq_rng, (), self._config.gait_randomization.frequency),
            1.0 / self._config.gait.period,
        )
        gait_phase_dt = self.dt * gait_frequency
        push_interval = self._random_uniform(
            push_interval_rng, (), self._config.push_randomization.interval_range
        )
        push_interval_steps = jp.maximum(
            jp.round(push_interval / self.dt).astype(jp.int32), jp.array(1, dtype=jp.int32)
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.action_size),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        previous_action = jp.zeros(self.action_size)
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "command": command,
            "previous_action": previous_action,
            "target_qpos": self._nominal_joint_pose,
            "pd_torque": jp.zeros(self.action_size),
            "mean_torque_sq_sum": jp.array(0.0),
            "feet_air_time": jp.zeros(2),
            "last_foot_contact": self._get_foot_contacts(data),
            "gait_phase": jp.array(0.0),
            "gait_frequency": gait_frequency,
            "gait_phase_dt": gait_phase_dt,
            "push": jp.zeros(2),
            "push_magnitude": jp.array(0.0),
            "push_step": jp.array(0, dtype=jp.int32),
            "push_interval_steps": push_interval_steps,
        }
        obs = self._get_obs(data, info)
        done, termination_metrics = self._get_done(data, obs)
        reward, reward_metrics = self._get_reward(
            data,
            command,
            jp.array(0.0),
            done,
            jp.zeros(2),
            jp.zeros(2),
            jp.array(0.0),
            jp.array(0.0),
            jp.array(0.0),
        )
        metrics = {
            "command/vx": command[0],
            "command/vy": command[1],
            "command/yaw_rate": command[2],
            "randomization/gait_frequency": gait_frequency,
            "randomization/push_interval_steps": push_interval_steps,
            "randomization/push_magnitude": jp.array(0.0),
            "randomization/push_x": jp.array(0.0),
            "randomization/push_y": jp.array(0.0),
            "action/applied_abs_mean": jp.array(0.0),
            "action/applied_abs_max": jp.array(0.0),
            **termination_metrics,
            **reward_metrics,
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        clipped_action = jp.clip(action, -1.0, 1.0)
        targets = self.action_to_target(clipped_action)

        previous_joint_vel = state.data.qvel[6:]
        info = dict(state.info)
        rng, command_rng, push_theta_rng, push_magnitude_rng = jax.random.split(info["rng"], 4)
        next_push_step = info["push_step"] + 1
        should_push = (
            self._config.push_randomization.enabled
            & (jp.mod(next_push_step, info["push_interval_steps"]) == 0)
        )
        push_theta = jax.random.uniform(push_theta_rng, (), minval=0.0, maxval=2.0 * jp.pi)
        sampled_push_magnitude = self._random_uniform(
            push_magnitude_rng, (), self._config.push_randomization.magnitude_range
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)]) * sampled_push_magnitude * should_push.astype(jp.float32)
        start_qvel = state.data.qvel.at[:2].set(state.data.qvel[:2] + push)
        start_data = state.data.replace(qvel=start_qvel)
        data, mean_torque_sq_sum = self._step_pd(start_data, targets)
        pd_torque = self.compute_pd_torque(data, targets)
        action_rate_cost = jp.sum(
            jp.square(clipped_action - state.info["previous_action"])
        )
        dof_acc = (data.qvel[6:] - previous_joint_vel) / self.dt
        dof_acc_cost = jp.sum(jp.square(dof_acc))
        next_step = info["step"] + 1
        should_resample = (
            self._config.command_sampling.enabled
            & (jp.mod(next_step, self.command_resampling_steps) == 0)
        )
        sampled_command = self.sample_command(command_rng)
        command = jp.where(should_resample, sampled_command, info["command"])
        gait_phase = jp.mod(info["gait_phase"] + info["gait_phase_dt"], 1.0)
        info["rng"] = rng
        info["step"] = next_step
        info["command"] = command
        info["gait_phase"] = gait_phase
        info["previous_action"] = clipped_action
        info["target_qpos"] = targets
        info["push"] = push
        info["push_magnitude"] = sampled_push_magnitude * should_push.astype(jp.float32)
        info["push_step"] = next_push_step
        foot_contacts = self._get_foot_contacts(data)
        feet_air_time = info["feet_air_time"] + self.dt
        first_contact = (
            (info["feet_air_time"] > 0.0)
            & ((foot_contacts + info["last_foot_contact"]) > 0.0)
        ).astype(jp.float32)
        info["pd_torque"] = pd_torque
        info["mean_torque_sq_sum"] = mean_torque_sq_sum
        info["feet_air_time"] = feet_air_time * (1.0 - foot_contacts)
        info["last_foot_contact"] = foot_contacts
        obs = self._get_obs(data, info)

        done, termination_metrics = self._get_done(data, obs)
        reward, reward_metrics = self._get_reward(
            data,
            command,
            mean_torque_sq_sum,
            done,
            feet_air_time,
            first_contact,
            action_rate_cost,
            dof_acc_cost,
            info["gait_phase"],
        )
        metrics = dict(state.metrics)
        metrics["command/vx"] = command[0]
        metrics["command/vy"] = command[1]
        metrics["command/yaw_rate"] = command[2]
        metrics["randomization/gait_frequency"] = info["gait_frequency"]
        metrics["randomization/push_magnitude"] = info["push_magnitude"]
        metrics["randomization/push_x"] = info["push"][0]
        metrics["randomization/push_y"] = info["push"][1]
        metrics["action/applied_abs_mean"] = jp.mean(jp.abs(clipped_action))
        metrics["action/applied_abs_max"] = jp.max(jp.abs(clipped_action))
        metrics.update(termination_metrics)
        metrics.update(reward_metrics)
        return state.replace(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def _sample_observation_noise(self, info: dict[str, jax.Array]) -> dict[str, jax.Array]:
        if not self._config.observation_noise.enabled or self._config.observation_noise.level == 0.0:
            return {}
        scales = self._config.observation_noise.scales
        level = self._config.observation_noise.level
        rng, gyro_rng, gravity_rng, joint_pos_rng, joint_vel_rng = jax.random.split(info["rng"], 5)
        info["rng"] = rng

        def uniform_noise(rng: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
            return (2.0 * jax.random.uniform(rng, shape=shape) - 1.0) * level * scale

        return {
            "gyro": uniform_noise(gyro_rng, (3,), scales.gyro),
            "gravity": uniform_noise(gravity_rng, (3,), scales.gravity),
            "joint_pos": uniform_noise(joint_pos_rng, (self.action_size,), scales.joint_pos),
            "joint_vel": uniform_noise(joint_vel_rng, (self.action_size,), scales.joint_vel),
        }

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, jax.Array],
        add_noise: bool = True,
    ) -> dict[str, jax.Array]:
        noise = self._sample_observation_noise(info) if add_noise else {}
        policy_obs = observations.build_policy_observation(
            model=self.mj_model,
            data=data,
            ids=self._obs_ids,
            command=info["command"],
            nominal_joint_pose=self._nominal_joint_pose,
            previous_action=info["previous_action"],
            gait_phase_observation=self._get_gait_phase_observation(
                info["gait_phase"],
                info["command"],
            ),
            noise=noise,
        )
        privileged_obs = observations.build_privileged_critic_observation(
            policy_observation=policy_obs,
            base_linear_velocity=self._get_base_linvel_body(data),
        )
        return {"state": policy_obs, "privileged_state": privileged_obs}

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return int(self._mjx_model.nu)

    @property
    def observation_size(self) -> dict[str, int]:
        return {
            "state": g1_config.OBSERVATION_SIZE,
            "privileged_state": g1_config.PRIVILEGED_OBSERVATION_SIZE,
        }

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def action_scales(self) -> jax.Array:
        return self._action_scales


def make_env(config_path: str | Path = g1_config.DEFAULT_CONFIG_PATH) -> G1FlatEnv:
    return G1FlatEnv(config=g1_config.load_config(config_path))
