#!/usr/bin/env python3
"""Evaluate one checkpoint and print an episode reward breakdown."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import play


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs" / "g1_flat.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--enable-randomization", action="store_true")
    return parser.parse_args()


def _scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim > 0:
        array = array.reshape(-1)[0]
    return float(array)


def _add_metric(acc: dict[str, list[float]], key: str, value: Any) -> None:
    scalar = _scalar(value)
    if math.isfinite(scalar):
        acc.setdefault(key, []).append(scalar)


def _print_table(title: str, keys: list[str], values: dict[str, list[float]], steps: int) -> None:
    print(f"\n{title}")
    print(f"{'metric':42s} {'mean/step':>14s} {'episode total':>16s}")
    print("-" * 76)
    for key in keys:
        samples = values.get(key, [])
        if not samples:
            continue
        mean = float(np.mean(samples))
        total = mean * steps
        print(f"{key:42s} {mean:14.6f} {total:16.3f}")


def main() -> int:
    args = _parse_args()
    checkpoint = play._resolve_project_path(args.checkpoint)
    env_config = play._resolve_project_path(args.env_config)
    rank = play._cuda_rank(args.device)
    _jax, torch = play._assert_cuda_ready(args.device, rank)
    raw_env = play._load_env(env_config, args.command, args.enable_randomization)
    steps = int(args.steps or raw_env._config.episode_length)
    wrapped_env, _runner, policy, critic_group = play._build_runner(
        raw_env,
        checkpoint,
        args.device,
        rank,
        args.seed,
    )

    obs = wrapped_env.reset()
    metrics: dict[str, list[float]] = {}
    rewards: list[float] = []
    termination_reason = "timeout"

    with torch.inference_mode():
        for step in range(steps):
            action = policy(obs)
            if tuple(action.shape) != (1, raw_env.action_size):
                raise RuntimeError(f"expected action shape (1, {raw_env.action_size}), got {tuple(action.shape)}")
            obs, reward, done, info = wrapped_env.step(action)
            rewards.append(float(reward[0].item()))
            for key, value in wrapped_env.env_state.metrics.items():
                if (
                    key.startswith("reward/")
                    or key.startswith("reward_rate/")
                    or key.startswith("tracking/")
                    or key.startswith("orientation/")
                    or key.startswith("torque/")
                    or key.startswith("action/")
                    or key.startswith("dof/")
                    or key.startswith("feet/")
                    or key.startswith("gait/")
                    or key.startswith("command/")
                    or key.startswith("randomization/")
                    or key.startswith("termination/")
                ):
                    _add_metric(metrics, key, value)
            for key, value in info.get("log", {}).items():
                if key.startswith("action/raw_") or key == "action/clip_fraction":
                    _add_metric(metrics, key, value)
            if bool(done[0].item()):
                time_out = bool(info.get("time_outs", torch.zeros_like(done))[0].item())
                termination_reason = "timeout" if time_out else "environment_done"
                break

    episode_steps = len(rewards)
    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards)) if rewards else math.nan

    reward_keys = [
        "reward/total",
        "reward/tracking_lin_vel",
        "reward/tracking_yaw_rate",
        "reward/alive",
        "reward/gait_contact",
        "reward/feet_air_time",
        "reward/orientation",
        "reward/torques",
        "reward/action_rate",
        "reward/dof_vel",
        "reward/dof_acc",
        "reward/feet_slide",
        "reward/feet_swing_height",
    ]
    reward_rate_keys = [key.replace("reward/", "reward_rate/") for key in reward_keys]
    diagnostic_keys = [
        "tracking/lin_vel_error",
        "tracking/yaw_rate_error",
        "orientation/cost",
        "torque/raw_sum_sq",
        "action/rate_cost",
        "action/raw_abs_mean",
        "action/raw_abs_max",
        "action/clip_fraction",
        "action/applied_abs_mean",
        "action/applied_abs_max",
        "dof/vel_cost",
        "dof/acc_cost",
        "feet/slide_cost",
        "feet/swing_height_cost",
        "gait/contact_reward_raw",
        "feet/air_time_reward_raw",
        "command/vx",
        "command/vy",
        "command/yaw_rate",
        "termination/height_low",
        "termination/height_high",
        "termination/tilt",
        "termination/illegal_contact",
        "termination/nonfinite",
    ]

    print("===== G1 Reward Breakdown =====")
    print(f"checkpoint: {checkpoint}")
    print(f"env config: {env_config}")
    print(f"critic observation group: {critic_group}")
    print(f"command: {list(map(float, raw_env._config.command))}")
    print(f"command sampling enabled: {bool(raw_env._config.command_sampling.enabled)}")
    print(f"training randomization enabled: {bool(args.enable_randomization)}")
    print(f"steps: {episode_steps}")
    print(f"termination reason: {termination_reason}")
    print(f"total reward from wrapper: {total_reward:.6f}")
    print(f"mean reward per step: {mean_reward:.6f}")
    _print_table("Weighted reward rates (before ctrl_dt)", reward_rate_keys, metrics, episode_steps)
    _print_table("Reward terms", reward_keys, metrics, episode_steps)
    _print_table("Diagnostics / raw costs", diagnostic_keys, metrics, episode_steps)

    penalty_keys = [
        key for key in reward_keys
        if key in metrics and float(np.mean(metrics[key])) < 0.0
    ]
    penalty_keys.sort(key=lambda key: float(np.mean(metrics[key])))
    if penalty_keys:
        print("\nWorst penalties by mean per step")
        print(f"{'metric':42s} {'mean/step':>14s} {'episode total':>16s}")
        print("-" * 76)
        for key in penalty_keys:
            mean = float(np.mean(metrics[key]))
            print(f"{key:42s} {mean:14.6f} {mean * episode_steps:16.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
