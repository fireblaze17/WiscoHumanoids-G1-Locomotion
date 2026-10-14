#!/usr/bin/env python3
"""Evaluate an RSL-RL checkpoint with deterministic actor inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CONTROL_CONTRACT_VERSION = 2
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs" / "g1_flat.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--render", choices=("video", "viewer", "none"), default="video")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument(
        "--enable-randomization",
        action="store_true",
        help="Use training-time randomization/noise during evaluation; disabled by default for deterministic play.",
    )
    return parser.parse_args()


def _cuda_rank(device: str) -> int:
    if device == "cuda":
        return 0
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError(f"only CUDA evaluation devices are supported, got {device!r}")
    return int(match.group(1))


def _resolve_project_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _default_video_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "runs" / "play" / stamp / "evaluation.mp4"


def _assert_cuda_ready(device: str, rank: int) -> tuple[Any, Any]:
    import jax
    import torch

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected JAX GPU backend, got {jax.default_backend()}")
    gpu_devices = jax.devices("gpu")
    if rank >= len(gpu_devices):
        raise RuntimeError(f"requested {device}, but JAX sees only {len(gpu_devices)} GPU device(s)")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested {device}, but PyTorch sees only {torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(rank)
    return jax, torch


def _load_env(config_path: Path, command: list[float] | None, enable_randomization: bool) -> Any:
    from g1_locomotion.envs import g1_config
    from g1_locomotion.envs.g1_env import G1FlatEnv

    cfg = g1_config.load_config(config_path)
    if not enable_randomization:
        cfg.command_sampling.enabled = False
        cfg.command_sampling.axis_sparsity.enabled = False
        cfg.observation_noise.enabled = False
        cfg.reset_randomization.enabled = False
        cfg.gait_randomization.enabled = False
        cfg.push_randomization.enabled = False
    if command is not None:
        cfg.command = [float(x) for x in command]
        cfg.command_sampling.enabled = False
    return G1FlatEnv(config=cfg)


def _plain_env_config(cfg: Any) -> dict[str, Any]:
    from g1_locomotion.training import rsl_rl_config

    return rsl_rl_config.to_plain_dict(cfg)


def _build_runner(raw_env: Any, checkpoint: Path, device: str, rank: int, seed: int) -> tuple[Any, Any, Any, str]:
    from mujoco import mjx
    from rsl_rl.runners import OnPolicyRunner

    from g1_locomotion.training import rsl_rl_config
    from g1_locomotion.training.rsl_rl_wrapper import G1RSLRLBraxWrapper

    if raw_env.mjx_model.impl != mjx.Impl.WARP:
        raise RuntimeError(f"expected MJX Warp backend, got {raw_env.mjx_model.impl}")

    wrapped_env = G1RSLRLBraxWrapper(
        raw_env,
        num_actors=1,
        seed=seed,
        episode_length=raw_env._config.episode_length,
        action_repeat=1,
        randomization_fn=None,
        device_rank=rank,
    )
    wrapped_env.cfg = _plain_env_config(raw_env._config)

    train_cfg = rsl_rl_config.default_ppo_config()
    train_cfg.seed = int(seed)

    def make_runner(cfg: Any) -> Any:
        return OnPolicyRunner(
            wrapped_env,
            rsl_rl_config.to_plain_dict(cfg),
            log_dir=None,
            device=device,
        )

    runner = make_runner(train_cfg)
    critic_group = "privileged_state"
    try:
        runner.load(str(checkpoint), load_optimizer=False, map_location=device)
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch" not in message and "shape" not in message:
            raise
        # Older checkpoints were trained before the critic-only base-linear-velocity
        # observation was added, so their critic input was the actor state. Keep
        # those checkpoints viewable without changing new training behavior.
        train_cfg.obs_groups.critic = ["state"]
        runner = make_runner(train_cfg)
        runner.load(str(checkpoint), load_optimizer=False, map_location=device)
        critic_group = "state"
    policy = runner.get_inference_policy(device=device)
    return wrapped_env, runner, policy, critic_group


def _warn_legacy_checkpoint(checkpoint: Path, current_contract: int) -> None:
    config_path = checkpoint.parent / "env_config.yaml"
    saved_version = 1
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            saved_config = yaml.safe_load(f) or {}
        saved_version = int(saved_config.get("control_contract_version", 1))
    if saved_version < CURRENT_CONTROL_CONTRACT_VERSION:
        print(
            "WARNING: checkpoint uses legacy control contract "
            f"v{saved_version}. Legacy playback is visual-only: supplying an archived "
            "v1 environment config does not recreate the old v1 reward/control behavior, "
            "and its metrics are not comparable with v2."
        )
    elif saved_version != current_contract:
        print(
            "WARNING: checkpoint contract "
            f"v{saved_version} differs from loaded environment contract v{current_contract}; "
            "results are visual-only and not directly comparable."
        )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _tracking_error(qpos: np.ndarray, qvel: np.ndarray, command: np.ndarray) -> float:
    rot = _quat_to_matrix(qpos[3:7])
    body_linear = rot.T @ qvel[:3]
    body_angular = rot.T @ qvel[3:6]
    actual = np.array([body_linear[0], body_linear[1], body_angular[2]], dtype=np.float64)
    return float(np.linalg.norm(actual - command))


def _first_env_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 1:
        return np.array(array[0])
    return np.array(array)



def _empty_eval_stats() -> dict[str, Any]:
    return {
        "total_episode_reward": 0.0,
        "episode_length": 0,
        "mean_commanded_velocity_tracking_error": math.nan,
        "termination_reason": "timeout",
        "mean_absolute_actuator_effort": None,
        "action_shape": None,
    }


def _update_eval_stats(
    *,
    stats_acc: dict[str, Any],
    raw_env: Any,
    wrapped_env: Any,
    reward: Any,
    done: Any,
    info: Any,
    obs: Any,
    torch: Any,
    action_shape: list[int],
    step: int,
) -> bool:
    stats_acc["episode_length"] = step + 1
    stats_acc["total_episode_reward"] += float(reward[0].item())
    stats_acc["action_shape"] = action_shape

    state = wrapped_env.env_state
    qpos = _first_env_array(state.data.qpos)
    qvel = _first_env_array(state.data.qvel)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        stats_acc["termination_reason"] = "nonfinite"
        return True
    if not bool(torch.isfinite(obs["state"]).all()):
        stats_acc["termination_reason"] = "nonfinite"
        return True

    command = np.asarray(raw_env._config.command, dtype=np.float64)
    stats_acc.setdefault("_tracking_errors", []).append(_tracking_error(qpos, qvel, command))
    actuator_force = getattr(state.data, "actuator_force", None)
    if actuator_force is not None:
        effort = np.abs(_first_env_array(actuator_force))
        if effort.size:
            stats_acc.setdefault("_efforts", []).append(float(np.mean(effort)))

    if bool(done[0].item()):
        time_out = bool(info.get("time_outs", torch.zeros_like(done))[0].item())
        stats_acc["termination_reason"] = "timeout" if time_out else "environment_done"
        return True
    return False


def _finalize_eval_stats(stats_acc: dict[str, Any]) -> dict[str, Any]:
    tracking_errors = stats_acc.pop("_tracking_errors", [])
    efforts = stats_acc.pop("_efforts", [])
    stats_acc["mean_commanded_velocity_tracking_error"] = (
        float(np.mean(tracking_errors)) if tracking_errors else math.nan
    )
    stats_acc["mean_absolute_actuator_effort"] = float(np.mean(efforts)) if efforts else None
    if stats_acc["action_shape"] is None:
        raise RuntimeError("policy did not produce an action")
    return stats_acc


def _rollout(
    *,
    raw_env: Any,
    wrapped_env: Any,
    policy: Any,
    torch: Any,
    steps: int,
    render_every: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    obs = wrapped_env.reset()
    qpos_frames = [_first_env_array(wrapped_env.env_state.data.qpos)]
    total_reward = 0.0
    tracking_errors: list[float] = []
    efforts: list[float] = []
    command = np.asarray(raw_env._config.command, dtype=np.float64)
    termination_reason = "timeout"
    action_shape: list[int] | None = None
    episode_length = 0

    with torch.inference_mode():
        for step in range(steps):
            action = policy(obs)
            if tuple(action.shape) != (1, raw_env.action_size):
                raise RuntimeError(f"expected action shape (1, {raw_env.action_size}), got {tuple(action.shape)}")
            if not bool(torch.isfinite(action).all()):
                termination_reason = "nonfinite"
                break
            action_shape = list(action.shape)

            obs, reward, done, info = wrapped_env.step(action)
            episode_length = step + 1
            total_reward += float(reward[0].item())

            state = wrapped_env.env_state
            qpos = _first_env_array(state.data.qpos)
            qvel = _first_env_array(state.data.qvel)
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                termination_reason = "nonfinite"
                break
            if not bool(torch.isfinite(obs["state"]).all()):
                termination_reason = "nonfinite"
                break

            tracking_errors.append(_tracking_error(qpos, qvel, command))
            actuator_force = getattr(state.data, "actuator_force", None)
            if actuator_force is not None:
                effort = np.abs(_first_env_array(actuator_force))
                if effort.size:
                    efforts.append(float(np.mean(effort)))

            if (step + 1) % render_every == 0:
                qpos_frames.append(qpos)

            if bool(done[0].item()):
                time_out = bool(info.get("time_outs", torch.zeros_like(done))[0].item())
                termination_reason = "timeout" if time_out else "environment_done"
                break

    if action_shape is None:
        raise RuntimeError("policy did not produce an action")

    stats = {
        "total_episode_reward": total_reward,
        "episode_length": episode_length,
        "mean_commanded_velocity_tracking_error": float(np.mean(tracking_errors)) if tracking_errors else math.nan,
        "termination_reason": termination_reason,
        "mean_absolute_actuator_effort": float(np.mean(efforts)) if efforts else None,
        "action_shape": action_shape,
    }
    return qpos_frames, stats



def _rollout_viewer_live(
    *,
    raw_env: Any,
    wrapped_env: Any,
    policy: Any,
    torch: Any,
    steps: int,
    render_every: int,
    fps: int,
) -> dict[str, Any]:
    import mujoco
    import mujoco.viewer

    obs = wrapped_env.reset()
    data = mujoco.MjData(raw_env.mj_model)
    qpos = _first_env_array(wrapped_env.env_state.data.qpos)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(raw_env.mj_model, data)

    stats_acc = _empty_eval_stats()
    dt = 1.0 / max(fps, 1)
    try:
        with mujoco.viewer.launch_passive(raw_env.mj_model, data) as viewer:
            with torch.inference_mode():
                for step in range(steps):
                    action = policy(obs)
                    if tuple(action.shape) != (1, raw_env.action_size):
                        raise RuntimeError(
                            f"expected action shape (1, {raw_env.action_size}), got {tuple(action.shape)}"
                        )
                    if not bool(torch.isfinite(action).all()):
                        stats_acc["termination_reason"] = "nonfinite"
                        break

                    obs, reward, done, info = wrapped_env.step(action)
                    should_stop = _update_eval_stats(
                        stats_acc=stats_acc,
                        raw_env=raw_env,
                        wrapped_env=wrapped_env,
                        reward=reward,
                        done=done,
                        info=info,
                        obs=obs,
                        torch=torch,
                        action_shape=list(action.shape),
                        step=step,
                    )

                    if (step + 1) % render_every == 0:
                        qpos = _first_env_array(wrapped_env.env_state.data.qpos)
                        data.qpos[:] = qpos
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(raw_env.mj_model, data)
                        viewer.sync()
                        time.sleep(dt)

                    if should_stop or not viewer.is_running():
                        break
    except Exception as exc:
        raise RuntimeError(
            "interactive MuJoCo viewer failed. Under WSL, use '--render video' "
            "unless WSLg/OpenGL GUI rendering is working."
        ) from exc

    return _finalize_eval_stats(stats_acc)


def _render_video(raw_env: Any, qpos_frames: list[np.ndarray], path: Path, width: int, height: int, fps: int) -> None:
    import imageio_ffmpeg
    import mediapy
    import mujoco

    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    data = mujoco.MjData(raw_env.mj_model)
    frames = []
    with mujoco.Renderer(raw_env.mj_model, height=height, width=width) as renderer:
        for qpos in qpos_frames:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(raw_env.mj_model, data)
            renderer.update_scene(data)
            frames.append(renderer.render())
    if not frames:
        raise RuntimeError("no frames were captured for video rendering")
    mediapy.write_video(path, frames, fps=fps)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"video writer did not create a non-empty file: {path}")


def _render_viewer(raw_env: Any, qpos_frames: list[np.ndarray], fps: int) -> None:
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(raw_env.mj_model)
    try:
        with mujoco.viewer.launch_passive(raw_env.mj_model, data) as viewer:
            dt = 1.0 / max(fps, 1)
            for qpos in qpos_frames:
                data.qpos[:] = qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(raw_env.mj_model, data)
                viewer.sync()
                time.sleep(dt)
    except Exception as exc:
        raise RuntimeError(
            "interactive MuJoCo viewer failed. Under WSL, use '--render video' "
            "unless WSLg/OpenGL GUI rendering is working."
        ) from exc


def main() -> int:
    args = _parse_args()
    if args.render == "video":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/g1_locomotion_matplotlib")

    checkpoint = _resolve_project_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.render_every <= 0:
        raise ValueError("--render-every must be positive")

    rank = _cuda_rank(args.device)
    jax, torch = _assert_cuda_ready(args.device, rank)
    raw_env = _load_env(_resolve_project_path(args.env_config), args.command, args.enable_randomization)
    _warn_legacy_checkpoint(
        checkpoint, int(raw_env._config.control_contract_version)
    )
    steps = int(args.steps or raw_env._config.episode_length)
    video_path = _resolve_project_path(args.video) if args.video else _default_video_path()
    summary_path = video_path.with_suffix(".json")

    wrapped_env, _runner, policy, critic_group = _build_runner(raw_env, checkpoint, args.device, rank, args.seed)
    if args.render == "viewer":
        stats = _rollout_viewer_live(
            raw_env=raw_env,
            wrapped_env=wrapped_env,
            policy=policy,
            torch=torch,
            steps=steps,
            render_every=args.render_every,
            fps=args.video_fps,
        )
    else:
        qpos_frames, stats = _rollout(
            raw_env=raw_env,
            wrapped_env=wrapped_env,
            policy=policy,
            torch=torch,
            steps=steps,
            render_every=args.render_every,
        )
        if args.render == "video":
            _render_video(raw_env, qpos_frames, video_path, args.width, args.height, args.video_fps)

    actor_obs_dim = (
        raw_env.observation_size["state"]
        if isinstance(raw_env.observation_size, dict)
        else raw_env.observation_size
    )
    critic_obs_dim = (
        raw_env.observation_size[critic_group]
        if isinstance(raw_env.observation_size, dict)
        else raw_env.observation_size
    )
    summary = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "jax_backend": jax.default_backend(),
        "simulation_backend": str(raw_env.mjx_model.impl),
        "action_dimension": raw_env.action_size,
        "observation_dimension": actor_obs_dim,
        "critic_observation_dimension": critic_obs_dim,
        "critic_observation_group": critic_group,
        "command": list(map(float, raw_env._config.command)),
        "command_sampling_enabled": bool(raw_env._config.command_sampling.enabled),
        "randomization_enabled": bool(args.enable_randomization),
        "observation_noise_enabled": bool(raw_env._config.observation_noise.enabled),
        "reset_randomization_enabled": bool(raw_env._config.reset_randomization.enabled),
        "gait_randomization_enabled": bool(raw_env._config.gait_randomization.enabled),
        "push_randomization_enabled": bool(raw_env._config.push_randomization.enabled),
        "render": args.render,
        "video": str(video_path) if args.render == "video" else None,
        **stats,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("===== G1 Policy Evaluation =====")
    print(f"checkpoint: {checkpoint}")
    print(f"simulation backend: {raw_env.mjx_model.impl}")
    print(f"device: {args.device}")
    print(f"action dimension: {raw_env.action_size}")
    print(f"actor observation dimension: {summary['observation_dimension']}")
    print(f"critic observation dimension: {summary['critic_observation_dimension']} ({summary['critic_observation_group']})")
    print(f"command: {summary['command']}")
    print(f"command sampling enabled: {summary['command_sampling_enabled']}")
    print(f"training randomization enabled: {summary['randomization_enabled']}")
    print(f"total episode reward: {summary['total_episode_reward']:.6f}")
    print(f"episode length: {summary['episode_length']}")
    print(
        "average commanded velocity tracking error: "
        f"{summary['mean_commanded_velocity_tracking_error']:.6f}"
    )
    print(f"termination reason: {summary['termination_reason']}")
    if summary["mean_absolute_actuator_effort"] is not None:
        print(f"average actuator effort: {summary['mean_absolute_actuator_effort']:.6f}")
    if args.render == "video":
        print(f"video: {video_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
