#!/usr/bin/env python3
"""Inspect the vendored Unitree G1 MuJoCo model."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "assets" / "g1" / "g1_scene.xml"


def name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or f"<unnamed:{index}>"


def model_xml_name(model_path: Path) -> str:
    scene_root = ET.parse(model_path).getroot()
    include = scene_root.find("include")
    target = model_path.parent / include.attrib["file"] if include is not None else model_path
    return ET.parse(target).getroot().attrib.get("model", "<unnamed>")


def actuated_joints(model: mujoco.MjModel) -> list[tuple[int, int, str, str]]:
    rows: list[tuple[int, int, str, str]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        rows.append(
            (
                actuator_id,
                joint_id,
                name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo XML scene/model path. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rows = actuated_joints(model)
    non_free_rows = [
        row
        for row in rows
        if model.jnt_type[row[1]] != mujoco.mjtJoint.mjJNT_FREE
    ]

    print(f"model path: {model_path}")
    print(f"model XML name: {model_xml_name(model_path)}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"timestep: {model.opt.timestep:g}")
    print(
        "initial base position: "
        f"[{model.qpos0[0]:.6g}, {model.qpos0[1]:.6g}, {model.qpos0[2]:.6g}]"
    )
    print(f"total body mass: {float(model.body_mass.sum()):.6f} kg")

    print("\nactuated joints in order:")
    for actuator_id, joint_id, joint_name, actuator_name in rows:
        print(f"{actuator_id + 1:02d}. {joint_name}  actuator={actuator_name}")

    print("\nactuator names in order:")
    for actuator_id in range(model.nu):
        print(f"{actuator_id + 1:02d}. {name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)}")

    print("\njoint position limits and actuator force limits:")
    for _, joint_id, joint_name, _ in rows:
        joint_range = model.jnt_range[joint_id]
        force_range = model.jnt_actfrcrange[joint_id]
        print(
            f"{joint_name}: "
            f"qpos=[{joint_range[0]:.6g}, {joint_range[1]:.6g}] "
            f"actfrc=[{force_range[0]:.6g}, {force_range[1]:.6g}]"
        )

    if model.nu != 29:
        raise AssertionError(f"expected 29 actuators, got {model.nu}")
    if len(non_free_rows) != 29:
        raise AssertionError(f"expected 29 actuated non-free joints, got {len(non_free_rows)}")

    print("\nOK: 29 controllable non-free joints confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
