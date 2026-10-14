#!/usr/bin/env python3
"""Verify the pinned conda-owned MuJoCo-Warp/RSL-RL GPU software stack."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPECTED_ENV = "g1-locomotion"
EXPECTED_PLAYGROUND_SHA = "4057c147714b6ac09b395377f1a1724bbeacc4d3"
EXPECTED_PYTHON = "3.12.14"
EXPECTED_VERSIONS = {
    "imageio-ffmpeg": "0.6.0",
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "mujoco": "3.13.0",
    "mujoco-mjx": "3.13.0",
    "mujoco-warp": "3.13.0",
    "playground": "0.2.0",
    "rsl-rl-lib": "3.2.0",
    "setuptools": "80.9.0",
    "tensorboard": "2.20.0",
    "torch": "2.9.1+cu130",
    "torchvision": "0.24.1+cu130",
    "warp-lang": "1.17.0",
}
ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "egl")


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def require_version(distribution: str) -> str:
    actual = version(distribution)
    expected = EXPECTED_VERSIONS[distribution]
    print(f"{distribution}: {actual} (expected {expected})")
    if actual != expected:
        fail(f"expected {distribution}=={expected}, found {actual}")
    return actual


def print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def require_import(module_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        fail(f"could not import {module_name}: {exc!r}")
    print(f"import {module_name}: ok")
    return module


def verify_python() -> None:
    print_section("Python")
    print(sys.version)
    print(f"executable: {sys.executable}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '')}")

    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != EXPECTED_PYTHON:
        fail(f"expected Python {EXPECTED_PYTHON}, found {actual_python}")

    project_venv = ROOT_DIR / ".venv"
    if Path(sys.prefix).resolve() == project_venv:
        fail("verification is running from the old project .venv")

    if ".venv" in Path(sys.executable).parts:
        fail("verification is running from a .venv Python")

    if os.environ.get("CONDA_DEFAULT_ENV") != EXPECTED_ENV:
        fail(f"expected conda env {EXPECTED_ENV!r}")


def verify_torch() -> None:
    print_section("PyTorch")
    torch = require_import("torch")
    print(f"torch package version: {torch.__version__}")
    require_version("torch")
    require_version("torchvision")
    print(f"torch.version.cuda: {torch.version.cuda}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        fail("torch reports no CUDA devices")
    print(f"torch CUDA device count: {device_count}")
    print(f"torch CUDA device 0: {torch.cuda.get_device_name(0)}")
    tensor = torch.ones(1, device="cuda")
    print(f"torch CUDA tensor device: {tensor.device}")


def verify_warp() -> None:
    print_section("Warp")
    wp = require_import("warp")
    require_version("warp-lang")
    print(f"warp __version__: {getattr(wp, '__version__', 'unknown')}")
    wp.init()

    cuda_devices: list[Any] = []
    if hasattr(wp, "get_cuda_devices"):
        cuda_devices = list(wp.get_cuda_devices())
    elif hasattr(wp, "get_devices"):
        cuda_devices = [
            device
            for device in wp.get_devices()
            if getattr(device, "is_cuda", False) or str(device).startswith("cuda")
        ]

    if not cuda_devices:
        try:
            if wp.is_cuda_available():
                cuda_devices = [wp.get_device("cuda:0")]
        except Exception:  # noqa: BLE001
            pass

    if not cuda_devices:
        fail("Warp reports no CUDA devices")

    for index, device in enumerate(cuda_devices):
        print(f"warp CUDA device {index}: {device}")

    array = wp.array([1.0], dtype=wp.float32, device="cuda:0")
    print(f"warp CUDA array device: {array.device}")


def verify_jax() -> None:
    print_section("JAX")
    jax = require_import("jax")
    print(f"jax package version: {jax.__version__}")
    require_version("jax")
    require_version("jaxlib")
    backend = jax.default_backend()
    print(f"jax default backend: {backend}")
    if backend != "gpu":
        fail("JAX default backend is not gpu")
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError as exc:
        fail(f"JAX could not enumerate GPU devices: {exc}")
    if not gpu_devices:
        fail("JAX reports no GPU devices")
    for index, device in enumerate(gpu_devices):
        print(f"jax GPU device {index}: {device}")


def verify_mujoco_stack() -> None:
    print_section("MuJoCo And Playground")
    mujoco = require_import("mujoco")
    mujoco_warp = require_import("mujoco_warp")
    require_import("mujoco.mjx")
    require_import("mujoco_playground")
    require_import("mujoco_playground._src.wrapper_torch")
    require_import("rsl_rl")

    print(f"mujoco __version__: {getattr(mujoco, '__version__', 'unknown')}")
    require_version("mujoco")
    require_version("mujoco-mjx")
    require_version("mujoco-warp")
    print(f"mujoco_warp __version__: {getattr(mujoco_warp, '__version__', 'unknown')}")
    require_version("playground")
    require_version("rsl-rl-lib")

    distribution = importlib.metadata.distribution("playground")
    direct_url_path = next(
        (
            Path(distribution.locate_file(item))
            for item in distribution.files or []
            if str(item).endswith(".dist-info/direct_url.json")
        ),
        None,
    )
    if direct_url_path is None or not direct_url_path.is_file():
        fail("Playground direct_url.json is missing; pinned Git provenance cannot be verified")
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    print(f"playground Git commit: {commit}")
    if commit != EXPECTED_PLAYGROUND_SHA:
        fail(f"expected Playground commit {EXPECTED_PLAYGROUND_SHA}, found {commit}")


def verify_tensorboard() -> None:
    print_section("TensorBoard")
    require_import("tensorboard")
    require_import("tensorboard.main")
    require_import("pkg_resources")
    require_version("tensorboard")
    require_version("setuptools")


def verify_video_encoding() -> None:
    print_section("Video Encoding")
    imageio_ffmpeg = require_import("imageio_ffmpeg")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(ffmpeg).is_file():
        fail(f"imageio-ffmpeg executable is unavailable: {ffmpeg}")
    require_version("imageio-ffmpeg")
    print(f"ffmpeg executable: {ffmpeg}")
    result = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, check=True, text=True
    )
    print(result.stdout.splitlines()[0])


def verify_egl_render() -> None:
    print_section("MuJoCo EGL Render Smoke Test")
    import numpy as np
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 2"/>
        <camera name="verify" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.89"/>
        <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body pos="0 0 0.2">
          <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data, camera="verify")
        frame = renderer.render()
    finally:
        renderer.close()
    if (
        frame.shape != (64, 64, 3)
        or not np.isfinite(frame).all()
        or not np.count_nonzero(frame)
    ):
        fail(f"unexpected EGL frame shape/content: {frame.shape}")
    print(f"MUJOCO_GL: {os.environ['MUJOCO_GL']}")
    print(f"EGL frame: {frame.shape}, nonzero pixels={int(np.count_nonzero(frame))}")


def verify_mjx_warp() -> None:
    print_section("MJX-Warp Smoke Test")
    import jax
    import mujoco
    from mujoco import mjx

    xml = """
    <mujoco>
      <worldbody>
        <body>
          <freejoint/>
          <geom size="0.1" mass="1" type="sphere"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mjx_model = mjx.put_model(mj_model, impl="warp")
    mjx_data = mjx.make_data(mj_model, impl="warp")
    stepped = jax.jit(mjx.step)(mjx_model, mjx_data)
    impl = getattr(getattr(mjx_model, "impl", None), "value", getattr(mjx_model, "impl", "unknown"))
    print(f"mjx model impl: {impl}")
    print(f"mjx stepped qpos shape: {stepped.qpos.shape}")


def verify_playground_warp_load() -> None:
    print_section("Playground Warp Smoke Test")
    from mujoco_playground import dm_control_suite

    env = dm_control_suite.load(
        "CartpoleBalance", config_overrides={"impl": "warp"}
    )
    print(f"loaded Playground env: {env.__class__.__name__}")
    print(f"observation size: {env.observation_size}")
    print(f"action size: {env.action_size}")


def main() -> None:
    verify_python()
    verify_mujoco_stack()
    verify_tensorboard()
    verify_video_encoding()
    verify_torch()
    verify_warp()
    verify_jax()
    verify_mjx_warp()
    verify_playground_warp_load()
    verify_egl_render()
    print("\nStack verification passed.")


if __name__ == "__main__":
    main()
