"""Actor and critic observation contracts for the G1 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src import mjx_env


OBSERVATION_LAYOUT = (
    ("base_angular_velocity", 0, 3, "pelvis gyro, rad/s"),
    ("projected_gravity", 3, 6, "gravity direction in pelvis frame"),
    ("command", 6, 9, "[vx, vy, yaw_rate]"),
    ("joint_position_error", 9, 38, "qpos[7:] - nominal_joint_pose"),
    ("joint_velocity", 38, 67, "qvel[6:]"),
    ("previous_action", 67, 96, "previous clipped policy action"),
    ("gait_phase", 96, 98, "[sin(2*pi*phase), cos(2*pi*phase)]"),
)
OBSERVATION_SIZE = 98
PRIVILEGED_OBSERVATION_SIZE = 101


@dataclass(frozen=True)
class ObservationIds:
    pelvis_imu_site_id: int


def get_sensor_data(model: Any, data: mjx.Data, sensor_name: str) -> jax.Array:
    return mjx_env.get_sensor_data(model, data, sensor_name)


def build_policy_observation(
    *,
    model: Any,
    data: mjx.Data,
    ids: ObservationIds,
    command: jax.Array,
    nominal_joint_pose: jax.Array,
    previous_action: jax.Array,
    gait_phase_observation: jax.Array,
    noise: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Builds the actor observation with no history."""
    noise = noise or {}
    base_ang_vel = get_sensor_data(model, data, "imu-pelvis-angular-velocity") + noise.get("gyro", 0.0)
    projected_gravity = (
        data.site_xmat[ids.pelvis_imu_site_id].T @ jp.array([0.0, 0.0, -1.0])
    ) + noise.get("gravity", 0.0)
    joint_pos_error = data.qpos[7:] - nominal_joint_pose + noise.get("joint_pos", 0.0)
    joint_vel = data.qvel[6:] + noise.get("joint_vel", 0.0)
    obs = jp.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            command,
            joint_pos_error,
            joint_vel,
            previous_action,
            gait_phase_observation,
        ]
    )
    if obs.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(f"expected observation size {OBSERVATION_SIZE}, got {obs.shape[-1]}")
    return obs


def build_privileged_critic_observation(
    *,
    policy_observation: jax.Array,
    base_linear_velocity: jax.Array,
) -> jax.Array:
    """Builds critic-only privileged obs; actor policy never receives this."""
    obs = jp.concatenate([policy_observation, base_linear_velocity])
    if obs.shape[-1] != PRIVILEGED_OBSERVATION_SIZE:
        raise ValueError(
            f"expected privileged observation size {PRIVILEGED_OBSERVATION_SIZE}, got {obs.shape[-1]}"
        )
    return obs


def describe_observation_layout() -> str:
    lines = ["Observation layout:"]
    for name, start, end, description in OBSERVATION_LAYOUT:
        lines.append(f"  [{start}:{end}] {name}: {description}")
    return "\n".join(lines)
