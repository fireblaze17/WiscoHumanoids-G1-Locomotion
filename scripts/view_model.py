#!/usr/bin/env python3
"""View or smoke-render the vendored Unitree G1 MuJoCo model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "assets" / "g1" / "g1_scene.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo XML scene/model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render one offscreen frame and exit.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model_path = args.model.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.headless:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()
        if pixels.size == 0:
            raise RuntimeError("renderer returned an empty frame")
        print(
            "headless render OK: "
            f"{pixels.shape[1]}x{pixels.shape[0]} frame from {model_path}"
        )
        return 0

    import mujoco.viewer

    print(f"Launching MuJoCo viewer for {model_path}")
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
