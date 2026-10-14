#!/usr/bin/env python3
"""Train the stochastic G1 Warp environment with RSL-RL PPO."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
).strip()

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_CONTROL_CONTRACT_VERSION = 2
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from ml_collections import config_dict
from mujoco import mjx
import mujoco
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

from g1_locomotion.envs import g1_config
from g1_locomotion.envs.g1_env import make_env
from g1_locomotion.training import rsl_rl_config
from g1_locomotion.training.rsl_rl_wrapper import G1RSLRLBraxWrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment", default="g1_flat")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint-frequency", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--env-config", type=Path, default=g1_config.DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _cuda_rank(device: str) -> int:
    if device == "cuda":
        return 0
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError(f"only CUDA training devices are supported, got {device!r}")
    return int(match.group(1))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "run"


def _resolve_project_log_root(path: Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"log directory must be inside the project, got {resolved}")
    return resolved


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _plain_env_config(cfg: config_dict.ConfigDict) -> dict[str, Any]:
    return rsl_rl_config.to_plain_dict(cfg)


def _assert_cuda_ready(device: str, rank: int) -> None:
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


def _assert_stack_ready(env: Any) -> None:
    if env.mjx_model.impl != mjx.Impl.WARP:
        raise RuntimeError(f"expected MJX Warp backend, got {env.mjx_model.impl}")


def _make_run_dir(args: argparse.Namespace) -> Path:
    log_root = _resolve_project_log_root(args.log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"-{_safe_name(args.run_name)}" if args.run_name else ""
    run_dir = log_root / _safe_name(args.experiment) / f"{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_files(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    env_cfg: config_dict.ConfigDict,
    train_cfg: config_dict.ConfigDict,
    device: str,
    rank: int,
) -> None:
    with (run_dir / "env_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_plain_env_config(env_cfg), f, sort_keys=False)

    with (run_dir / "ppo_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(rsl_rl_config.to_plain_dict(train_cfg), f, sort_keys=False)

    metadata = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": device,
        "torch_device_name": torch.cuda.get_device_name(rank),
        "mujoco_version": mujoco.__version__,
        "control_contract_version": int(env_cfg.control_contract_version),
        "git_head": _git_text(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_text(["status", "--short"]),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    if Path(args.env_config).resolve().exists():
        shutil.copy2(Path(args.env_config).resolve(), run_dir / "source_g1_flat.yaml")


def _print_startup(
    *,
    env: Any,
    train_cfg: config_dict.ConfigDict,
    run_dir: Path,
    num_envs: int,
    device: str,
    rank: int,
) -> None:
    rollout = int(train_cfg.num_steps_per_env)
    minibatches = int(train_cfg.algorithm.num_mini_batches)
    batch_size = num_envs * rollout
    minibatch_size = batch_size // minibatches

    print("===== G1 RSL-RL Training Startup =====")
    print(f"simulation implementation: {env.mjx_model.impl}")
    print(f"simulation device: {jax.devices('gpu')[rank]}")
    print(f"PyTorch/RSL-RL device: {device} ({torch.cuda.get_device_name(rank)})")
    print(f"number of environments: {num_envs}")
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        print(f"actor observation dimension: {obs_size['state']}")
        print(f"critic privileged observation dimension: {obs_size['privileged_state']}")
    else:
        print(f"observation dimension: {obs_size}")
    print(f"action dimension: {env.action_size}")
    print(f"control contract version: {env._config.control_contract_version}")
    print(f"action scale mode: {env._config.action_scale.mode}")
    print(
        "action scale range: "
        f"[{float(env.action_scales.min()):.6f}, {float(env.action_scales.max()):.6f}] rad"
    )
    print(f"PPO rollout length: {rollout}")
    print(f"PPO minibatches: {minibatches}")
    print(f"PPO minibatch size: {minibatch_size}")
    print(f"actor network: {list(train_cfg.policy.actor_hidden_dims)}")
    print(f"critic network: {list(train_cfg.policy.critic_hidden_dims)}")
    print(f"PPO learning-rate schedule: {train_cfg.algorithm.schedule}")
    print(f"PPO initial learning rate: {train_cfg.algorithm.learning_rate}")
    print(f"PPO desired KL: {train_cfg.algorithm.desired_kl}")
    print(f"run output directory: {run_dir}")
    print("======================================")


def _install_loss_checks(runner: OnPolicyRunner, loss_history: list[dict[str, float]]) -> None:
    update = runner.alg.update

    def checked_update() -> dict[str, float]:
        losses = update()
        checked: dict[str, float] = {}
        for name, value in losses.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise RuntimeError(f"non-finite PPO loss {name}: {scalar}")
            checked[name] = scalar
        loss_history.append(checked)
        return losses

    runner.alg.update = checked_update


def _assert_post_training(runner: OnPolicyRunner, run_dir: Path, loss_history: list[dict[str, float]]) -> None:
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"no RSL-RL checkpoints were written under {run_dir}")
    if not loss_history:
        raise RuntimeError("RSL-RL PPO did not report any loss values")

    rewards = runner.alg.storage.rewards
    if rewards.numel() and not bool(torch.isfinite(rewards).all()):
        raise RuntimeError("rollout rewards contain NaN or Inf")
    for name, param in runner.alg.policy.named_parameters():
        if not bool(torch.isfinite(param).all()):
            raise RuntimeError(f"policy parameter {name} contains NaN or Inf")

    with (run_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2, sort_keys=True)

    print(f"finite PPO updates: OK ({len(loss_history)})")
    print(f"finite rollout rewards: OK")
    print(f"finite policy parameters: OK")
    print(f"checkpoint written: {checkpoints[-1]}")


def _assert_resume_contract(resume_path: Path, expected_version: int) -> None:
    if expected_version != CURRENT_CONTROL_CONTRACT_VERSION:
        raise RuntimeError(
            "training only supports current control contract "
            f"v{CURRENT_CONTROL_CONTRACT_VERSION}, but the environment config declares "
            f"v{expected_version}; archived configs are playback-only"
        )
    config_path = resume_path.parent / "env_config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            "resume checkpoint has no env_config.yaml; legacy checkpoints cannot be "
            "resumed under the stabilized control contract"
        )
    with config_path.open("r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f) or {}
    saved_version = int(saved_config.get("control_contract_version", 1))
    if saved_version != CURRENT_CONTROL_CONTRACT_VERSION:
        raise RuntimeError(
            "resume checkpoint uses control contract "
            f"v{saved_version}, but training requires current contract "
            f"v{CURRENT_CONTROL_CONTRACT_VERSION}; "
            "start a fresh run instead"
        )


def main() -> int:
    args = _parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.checkpoint_frequency <= 0:
        raise ValueError("--checkpoint-frequency must be positive")

    rank = _cuda_rank(args.device)
    _assert_cuda_ready(args.device, rank)
    raw_env = make_env(args.env_config)
    _assert_stack_ready(raw_env)
    env_contract = int(raw_env._config.control_contract_version)
    if env_contract != CURRENT_CONTROL_CONTRACT_VERSION:
        raise RuntimeError(
            "training only supports current control contract "
            f"v{CURRENT_CONTROL_CONTRACT_VERSION}, but {args.env_config} declares "
            f"v{env_contract}; archived configs are playback-only"
        )
    resume_path = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        _assert_resume_contract(
            resume_path, env_contract
        )

    train_cfg = rsl_rl_config.default_ppo_config()
    train_cfg.seed = int(args.seed)
    train_cfg.max_iterations = int(args.max_iterations)
    train_cfg.save_interval = int(args.checkpoint_frequency)
    train_cfg.experiment_name = str(args.experiment)
    train_cfg.run_name = str(args.run_name)

    run_dir = _make_run_dir(args)
    _write_run_files(
        run_dir=run_dir,
        args=args,
        env_cfg=raw_env._config,
        train_cfg=train_cfg,
        device=args.device,
        rank=rank,
    )
    _print_startup(
        env=raw_env,
        train_cfg=train_cfg,
        run_dir=run_dir,
        num_envs=args.num_envs,
        device=args.device,
        rank=rank,
    )

    wrapped_env = G1RSLRLBraxWrapper(
        raw_env,
        num_actors=args.num_envs,
        seed=args.seed,
        episode_length=raw_env._config.episode_length,
        action_repeat=1,
        randomization_fn=None,
        device_rank=rank,
    )
    wrapped_env.cfg = _plain_env_config(raw_env._config)

    train_cfg_dict = rsl_rl_config.to_plain_dict(train_cfg)
    runner = OnPolicyRunner(
        wrapped_env,
        train_cfg_dict,
        log_dir=str(run_dir),
        device=args.device,
    )

    if resume_path is not None:
        runner.load(str(resume_path), map_location=args.device)
        print(f"resumed checkpoint: {resume_path}")

    loss_history: list[dict[str, float]] = []
    _install_loss_checks(runner, loss_history)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    _assert_post_training(runner, run_dir, loss_history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
