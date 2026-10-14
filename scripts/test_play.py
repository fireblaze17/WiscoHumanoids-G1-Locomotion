#!/usr/bin/env python3
"""Smoke-test deterministic policy evaluation from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


def _latest_v2_checkpoint() -> Path:
    candidates: list[Path] = []
    for metadata_path in (REPO_ROOT / "runs").glob("*/*/run_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(metadata.get("control_contract_version", 1)) != 2:
            continue
        candidates.extend(metadata_path.parent.glob("model_*.pt"))
    checkpoints = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            "no contract-v2 checkpoint found under runs/*/*; run scripts/train.py first"
        )
    return checkpoints[-1]


def main() -> int:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _latest_v2_checkpoint()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "runs" / "play_smoke" / stamp
    video = out_dir / "evaluation.mp4"

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "play.py"),
        "--checkpoint",
        str(checkpoint),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--render",
        "video",
        "--video",
        str(video),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--video-fps",
        str(args.video_fps),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    summary_path = video.with_suffix(".json")
    if not video.exists() or video.stat().st_size == 0:
        raise AssertionError(f"expected non-empty video at {video}")
    if not summary_path.exists():
        raise AssertionError(f"expected summary at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["action_dimension"] != 29:
        raise AssertionError(f"expected action dimension 29, got {summary['action_dimension']}")
    if summary["observation_dimension"] != 98:
        raise AssertionError(f"expected observation dimension 98, got {summary['observation_dimension']}")
    if summary["action_shape"] != [1, 29]:
        raise AssertionError(f"expected action shape [1, 29], got {summary['action_shape']}")
    if summary["episode_length"] <= 0:
        raise AssertionError("expected a non-empty rollout")
    if summary["termination_reason"] not in {"timeout", "environment_done", "nonfinite"}:
        raise AssertionError(f"unexpected termination reason {summary['termination_reason']}")
    if summary["termination_reason"] == "nonfinite":
        raise AssertionError("rollout terminated due to nonfinite state/action")

    print(f"play smoke checkpoint: {checkpoint}")
    print(f"play smoke video: {video}")
    print(f"play smoke summary: {summary_path}")
    print("play smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
