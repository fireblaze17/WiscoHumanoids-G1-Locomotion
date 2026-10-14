"""Configuration loading for the G1 flat-ground training environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_collections import config_dict
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "g1_flat.yaml"
DEFAULT_XML_PATH = REPO_ROOT / "assets" / "g1" / "g1_scene.xml"

ACTION_SIZE = 29
OBSERVATION_SIZE = 98
PRIVILEGED_OBSERVATION_SIZE = 101
CONTROL_CONTRACT_VERSION = 2
TWIST2_GAIN_KEYS = (
    "hip_yaw",
    "hip_roll",
    "hip_pitch",
    "knee",
    "ankle",
    "waist",
    "shoulder",
    "elbow",
    "wrist",
)


def _resolve_path(value: str | Path, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _require_vector(raw: dict[str, Any], key: str, length: int) -> list[float]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be a list of {length} numbers")
    return [float(x) for x in value]


def _require_range(raw: dict[str, Any], key: str) -> list[float]:
    value = raw.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be a list of 2 numbers")
    lower, upper = [float(x) for x in value]
    if lower > upper:
        raise ValueError(f"{key} lower bound must be <= upper bound")
    return [lower, upper]


def _load_gain_group(raw: dict[str, Any], key: str, defaults: dict[str, float]) -> config_dict.ConfigDict:
    values = raw.get(key, defaults)
    if not isinstance(values, dict):
        raise ValueError(f"{key} must be a mapping")
    missing = [name for name in TWIST2_GAIN_KEYS if name not in values]
    if missing:
        raise ValueError(f"{key} is missing gains for: {', '.join(missing)}")

    cfg = config_dict.ConfigDict()
    for name in TWIST2_GAIN_KEYS:
        cfg[name] = float(values[name])
    return cfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> config_dict.ConfigDict:
    """Loads and validates the flat-ground G1 training config."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = config_dict.ConfigDict()
    cfg.xml_path = _resolve_path(raw.get("xml_path", DEFAULT_XML_PATH), config_path.parent)
    cfg.impl = str(raw.get("impl", "warp"))
    cfg.sim_dt = float(raw.get("sim_dt", 0.002))
    cfg.ctrl_dt = float(raw.get("ctrl_dt", 0.02))
    cfg.episode_length = int(raw.get("episode_length", 1000))
    cfg.control_contract_version = int(raw.get("control_contract_version", 1))
    action_scale = raw.get("action_scale", 0.5)
    cfg.action_scale = config_dict.ConfigDict()
    if isinstance(action_scale, (int, float)):
        cfg.action_scale.mode = "uniform"
        cfg.action_scale.uniform = float(action_scale)
        cfg.action_scale.effort_fraction = 0.25
    elif isinstance(action_scale, dict):
        cfg.action_scale.mode = str(action_scale.get("mode", "effort_fraction"))
        cfg.action_scale.uniform = float(action_scale.get("uniform", 0.5))
        cfg.action_scale.effort_fraction = float(action_scale.get("effort_fraction", 0.25))
    else:
        raise ValueError("action_scale must be a number or mapping")
    cfg.controller = str(raw.get("controller", "torque_pd"))
    cfg.command = _require_vector(raw, "command", 3)

    command_sampling = raw.get("command_sampling", {})
    if not isinstance(command_sampling, dict):
        raise ValueError("command_sampling must be a mapping")
    command_ranges = command_sampling.get("ranges", {})
    if not isinstance(command_ranges, dict):
        raise ValueError("command_sampling.ranges must be a mapping")
    cfg.command_sampling = config_dict.ConfigDict()
    cfg.command_sampling.enabled = bool(command_sampling.get("enabled", False))
    cfg.command_sampling.resampling_time = float(command_sampling.get("resampling_time", 10.0))
    cfg.command_sampling.zero_small_xy_threshold = float(
        command_sampling.get("zero_small_xy_threshold", 0.2)
    )
    cfg.command_sampling.ranges = config_dict.ConfigDict()
    cfg.command_sampling.ranges.vx = _require_range(command_ranges, "vx")
    cfg.command_sampling.ranges.vy = _require_range(command_ranges, "vy")
    cfg.command_sampling.ranges.yaw_rate = _require_range(command_ranges, "yaw_rate")
    axis_sparsity = command_sampling.get("axis_sparsity", {})
    if not isinstance(axis_sparsity, dict):
        raise ValueError("command_sampling.axis_sparsity must be a mapping")
    cfg.command_sampling.axis_sparsity = config_dict.ConfigDict()
    cfg.command_sampling.axis_sparsity.enabled = bool(axis_sparsity.get("enabled", False))
    cfg.command_sampling.axis_sparsity.amplitude = _require_vector(
        axis_sparsity or {"amplitude": [1.0, 1.0, 1.0]}, "amplitude", 3
    )
    cfg.command_sampling.axis_sparsity.active_probability = _require_vector(
        axis_sparsity or {"active_probability": [1.0, 1.0, 1.0]}, "active_probability", 3
    )

    observation_noise = raw.get("observation_noise", {})
    if not isinstance(observation_noise, dict):
        raise ValueError("observation_noise must be a mapping")
    noise_scales = observation_noise.get("scales", {})
    if not isinstance(noise_scales, dict):
        raise ValueError("observation_noise.scales must be a mapping")
    cfg.observation_noise = config_dict.ConfigDict()
    cfg.observation_noise.enabled = bool(observation_noise.get("enabled", False))
    cfg.observation_noise.level = float(observation_noise.get("level", 0.0))
    cfg.observation_noise.scales = config_dict.ConfigDict()
    cfg.observation_noise.scales.joint_pos = float(noise_scales.get("joint_pos", 0.03))
    cfg.observation_noise.scales.joint_vel = float(noise_scales.get("joint_vel", 1.5))
    cfg.observation_noise.scales.gravity = float(noise_scales.get("gravity", 0.05))
    cfg.observation_noise.scales.linvel = float(noise_scales.get("linvel", 0.1))
    cfg.observation_noise.scales.gyro = float(noise_scales.get("gyro", 0.2))

    reset_randomization = raw.get("reset_randomization", {})
    if not isinstance(reset_randomization, dict):
        raise ValueError("reset_randomization must be a mapping")
    cfg.reset_randomization = config_dict.ConfigDict()
    cfg.reset_randomization.enabled = bool(reset_randomization.get("enabled", False))
    cfg.reset_randomization.base_xy = _require_range(
        reset_randomization or {"base_xy": [0.0, 0.0]}, "base_xy"
    )
    cfg.reset_randomization.base_yaw = _require_range(
        reset_randomization or {"base_yaw": [0.0, 0.0]}, "base_yaw"
    )
    cfg.reset_randomization.joint_pose_scale = _require_range(
        reset_randomization or {"joint_pose_scale": [1.0, 1.0]}, "joint_pose_scale"
    )
    cfg.reset_randomization.root_velocity = _require_range(
        reset_randomization or {"root_velocity": [0.0, 0.0]}, "root_velocity"
    )

    gait_randomization = raw.get("gait_randomization", {})
    if not isinstance(gait_randomization, dict):
        raise ValueError("gait_randomization must be a mapping")
    cfg.gait_randomization = config_dict.ConfigDict()
    cfg.gait_randomization.enabled = bool(gait_randomization.get("enabled", False))
    cfg.gait_randomization.frequency = _require_range(
        gait_randomization or {"frequency": [1.25, 1.25]}, "frequency"
    )

    push_randomization = raw.get("push_randomization", {})
    if not isinstance(push_randomization, dict):
        raise ValueError("push_randomization must be a mapping")
    cfg.push_randomization = config_dict.ConfigDict()
    cfg.push_randomization.enabled = bool(push_randomization.get("enabled", False))
    cfg.push_randomization.interval_range = _require_range(
        push_randomization or {"interval_range": [5.0, 5.0]}, "interval_range"
    )
    cfg.push_randomization.magnitude_range = _require_range(
        push_randomization or {"magnitude_range": [0.0, 0.0]}, "magnitude_range"
    )

    gait = raw.get("gait", {})
    if not isinstance(gait, dict):
        raise ValueError("gait must be a mapping")
    cfg.gait = config_dict.ConfigDict()
    cfg.gait.enabled = bool(gait.get("enabled", True))
    cfg.gait.period = float(gait.get("period", 0.8))
    cfg.gait.offsets = _require_vector(gait, "offsets", 2)
    cfg.gait.stance_threshold = float(gait.get("stance_threshold", 0.55))
    cfg.gait.command_threshold = float(gait.get("command_threshold", 0.1))

    reward = raw.get("reward", {})
    if not isinstance(reward, dict):
        raise ValueError("reward must be a mapping")
    cfg.reward = config_dict.ConfigDict()
    cfg.reward.tracking_sigma = float(reward.get("tracking_sigma", 0.25))
    cfg.reward.tracking_lin_vel = float(reward.get("tracking_lin_vel", 1.0))
    cfg.reward.tracking_yaw_rate = float(reward.get("tracking_yaw_rate", 0.5))
    cfg.reward.orientation = float(reward.get("orientation", -1.0))
    cfg.reward.alive = float(reward.get("alive", 0.15))
    cfg.reward.torques = float(reward.get("torques", -1.0e-5))
    cfg.reward.action_rate = float(reward.get("action_rate", -0.01))
    cfg.reward.dof_vel = float(reward.get("dof_vel", -1.0e-3))
    cfg.reward.dof_acc = float(reward.get("dof_acc", -2.5e-7))
    cfg.reward.feet_slide = float(reward.get("feet_slide", -0.2))
    cfg.reward.gait_contact = float(reward.get("gait_contact", 0.5))
    cfg.reward.feet_air_time = float(reward.get("feet_air_time", 2.0))
    cfg.reward.feet_air_time_min = float(reward.get("feet_air_time_min", 0.2))
    cfg.reward.feet_air_time_max = float(reward.get("feet_air_time_max", 0.5))
    cfg.reward.feet_air_time_command_threshold = float(
        reward.get("feet_air_time_command_threshold", 0.1)
    )
    cfg.reward.feet_swing_height = float(reward.get("feet_swing_height", -20.0))
    cfg.reward.feet_swing_height_target = float(reward.get("feet_swing_height_target", 0.08))
    cfg.reward.feet_swing_height_command_threshold = float(
        reward.get("feet_swing_height_command_threshold", 0.1)
    )

    termination = raw.get("termination", {})
    if not isinstance(termination, dict):
        raise ValueError("termination must be a mapping")
    cfg.termination = config_dict.ConfigDict()
    cfg.termination.min_base_height = float(termination.get("min_base_height", 0.5))
    cfg.termination.max_base_height = float(termination.get("max_base_height", 1.0))
    cfg.termination.max_tilt_degrees = float(termination.get("max_tilt_degrees", 70.0))
    illegal_contact_sensors = termination.get("illegal_contact_sensors", [])
    if not isinstance(illegal_contact_sensors, list) or not all(
        isinstance(name, str) for name in illegal_contact_sensors
    ):
        raise ValueError("termination.illegal_contact_sensors must be a list of sensor names")
    cfg.termination.illegal_contact_sensors = illegal_contact_sensors

    cfg.reset_base_position = _require_vector(raw, "reset_base_position", 3)
    cfg.reset_base_quaternion = _require_vector(raw, "reset_base_quaternion", 4)
    cfg.nominal_joint_pose = _require_vector(raw, "nominal_joint_pose", ACTION_SIZE)
    cfg.floor_friction = _require_vector(raw, "floor_friction", 3)
    cfg.naconmax = int(raw.get("naconmax", 65536))
    cfg.naccdmax = int(raw.get("naccdmax", 512))
    cfg.njmax = int(raw.get("njmax", 512))

    cfg.pd_stiffness = _load_gain_group(
        raw,
        "pd_stiffness",
        {
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle": 40.0,
            "waist": 150.0,
            "shoulder": 40.0,
            "elbow": 40.0,
            "wrist": 40.0,
        },
    )
    cfg.pd_damping = _load_gain_group(
        raw,
        "pd_damping",
        {
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 4.0,
            "ankle": 2.0,
            "waist": 4.0,
            "shoulder": 5.0,
            "elbow": 5.0,
            "wrist": 5.0,
        },
    )

    if cfg.impl != "warp":
        raise ValueError(f"this environment expects impl='warp', got {cfg.impl!r}")
    if cfg.controller != "torque_pd":
        raise ValueError(f"this environment expects controller='torque_pd', got {cfg.controller!r}")
    if cfg.ctrl_dt <= 0 or cfg.sim_dt <= 0:
        raise ValueError("ctrl_dt and sim_dt must be positive")
    if cfg.command_sampling.resampling_time <= 0:
        raise ValueError("command_sampling.resampling_time must be positive")
    if cfg.command_sampling.zero_small_xy_threshold < 0:
        raise ValueError("command_sampling.zero_small_xy_threshold must be non-negative")
    for value in cfg.command_sampling.axis_sparsity.active_probability:
        if not 0.0 <= value <= 1.0:
            raise ValueError("command_sampling.axis_sparsity.active_probability values must be in [0, 1]")
    if cfg.observation_noise.level < 0:
        raise ValueError("observation_noise.level must be non-negative")
    for name in ["joint_pos", "joint_vel", "gravity", "linvel", "gyro"]:
        if cfg.observation_noise.scales[name] < 0:
            raise ValueError(f"observation_noise.scales.{name} must be non-negative")
    if cfg.gait_randomization.frequency[0] <= 0:
        raise ValueError("gait_randomization.frequency lower bound must be positive")
    if cfg.push_randomization.interval_range[0] <= 0:
        raise ValueError("push_randomization.interval_range lower bound must be positive")
    if cfg.push_randomization.magnitude_range[0] < 0:
        raise ValueError("push_randomization.magnitude_range lower bound must be non-negative")
    if cfg.gait.period <= 0:
        raise ValueError("gait.period must be positive")
    if not 0.0 <= cfg.gait.stance_threshold <= 1.0:
        raise ValueError("gait.stance_threshold must be in [0, 1]")
    if cfg.gait.command_threshold < 0:
        raise ValueError("gait.command_threshold must be non-negative")
    if cfg.reward.tracking_sigma <= 0:
        raise ValueError("reward.tracking_sigma must be positive")
    if cfg.reward.feet_air_time_min < 0:
        raise ValueError("reward.feet_air_time_min must be non-negative")
    if cfg.reward.feet_air_time_max < cfg.reward.feet_air_time_min:
        raise ValueError("reward.feet_air_time_max must be at least feet_air_time_min")
    if cfg.reward.feet_air_time_command_threshold < 0:
        raise ValueError("reward.feet_air_time_command_threshold must be non-negative")
    if cfg.reward.feet_swing_height_target < 0:
        raise ValueError("reward.feet_swing_height_target must be non-negative")
    if cfg.reward.feet_swing_height_command_threshold < 0:
        raise ValueError("reward.feet_swing_height_command_threshold must be non-negative")
    if cfg.termination.min_base_height >= cfg.termination.max_base_height:
        raise ValueError("termination.min_base_height must be less than max_base_height")
    if cfg.control_contract_version < 1:
        raise ValueError("control_contract_version must be positive")
    if cfg.action_scale.mode not in {"uniform", "effort_fraction"}:
        raise ValueError("action_scale.mode must be 'uniform' or 'effort_fraction'")
    if cfg.action_scale.uniform <= 0:
        raise ValueError("action_scale.uniform must be positive")
    if not 0.0 < cfg.action_scale.effort_fraction <= 1.0:
        raise ValueError("action_scale.effort_fraction must be in (0, 1]")
    if not 0.0 < cfg.termination.max_tilt_degrees <= 180.0:
        raise ValueError("termination.max_tilt_degrees must be in (0, 180]")
    n_substeps = cfg.ctrl_dt / cfg.sim_dt
    if round(n_substeps) < 1:
        raise ValueError("ctrl_dt must be at least sim_dt")
    if abs(n_substeps - round(n_substeps)) > 1e-9:
        raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
    if not Path(cfg.xml_path).exists():
        raise FileNotFoundError(cfg.xml_path)

    return cfg
