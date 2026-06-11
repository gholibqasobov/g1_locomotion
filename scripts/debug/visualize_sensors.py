# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug visualizer for the G1 stair-climbing sensors.

Spawns a few envs of a stairs task with the **height scanner** and the **head RealSense depth camera**
debug-visualized, so you can confirm placement/angle in the viewer:

* height scanner  -> grid of ray hit points on the terrain under/around the torso
* depth camera    -> camera frustum on the head, pitched 42.4 deg down
* terrain origins -> generated stairs

It also prints the live sensor data shapes and (optionally) dumps a depth image from env 0 so you can see
exactly what the RealSense "sees".

Examples
--------
    # interactive viewer (recommended) — look at the frustum + scan points
    python scripts/debug/visualize_sensors.py --num_envs 4

    # also save a depth snapshot from env 0
    python scripts/debug/visualize_sensors.py --num_envs 4 --save_depth --enable_cameras

    # inspect the student task instead (same scene/sensors)
    python scripts/debug/visualize_sensors.py --task Template-G1-Stairs-Student-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize the G1 stair-climbing sensors for debugging.")
parser.add_argument("--task", type=str, default="Template-G1-Stairs-Teacher-v0", help="Stairs task to load.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=100000, help="Number of idle steps to run.")
parser.add_argument("--save_depth", action="store_true", default=False, help="Save a depth image from env 0.")
parser.add_argument(
    "--out", type=str, default="logs/debug/depth_env0", help="Output path prefix for the saved depth (no extension)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# the ray-cast depth camera works without RTX cameras, but enabling them is harmless and required for
# saving a clean snapshot in some setups
if args_cli.save_depth:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

import g1_locomotion.tasks  # noqa: F401  (registers the tasks)


def _save_depth(depth_hw: torch.Tensor, out_prefix: str) -> None:
    """Save a single [H, W] depth tensor as .npy and (best-effort) .png."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    arr = depth_hw.detach().float().cpu().numpy()
    import numpy as np

    np.save(out_prefix + ".npy", arr)
    print(f"[debug] saved depth array -> {out_prefix}.npy  (shape={arr.shape})")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        finite = arr[np.isfinite(arr)]
        vmax = float(finite.max()) if finite.size else 1.0
        plt.imsave(out_prefix + ".png", np.nan_to_num(arr, nan=vmax), cmap="viridis", vmin=0.0, vmax=vmax)
        print(f"[debug] saved depth image -> {out_prefix}.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[debug] PNG save skipped ({exc}); use the .npy file.")


def main() -> None:
    # build the env cfg with a handful of envs and turn ON sensor debug visualization
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

    env_cfg.scene.height_scanner.debug_vis = True
    env_cfg.scene.camera.debug_vis = True
    env_cfg.scene.terrain.debug_vis = True
    # nicer vantage for inspecting the head/torso sensors
    env_cfg.viewer.eye = (3.5, 3.5, 2.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.8)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped

    print("\n[debug] sensors in scene:", list(base.scene.sensors.keys()))
    print(f"[debug] camera offset pos={env_cfg.scene.camera.offset.pos} "
          f"rot(wxyz)={env_cfg.scene.camera.offset.rot} convention={env_cfg.scene.camera.offset.convention}")
    print(f"[debug] camera pattern: {env_cfg.scene.camera.pattern_cfg.width}x{env_cfg.scene.camera.pattern_cfg.height}"
          f"  data_types={env_cfg.scene.camera.data_types}\n")

    obs, _ = env.reset()
    action_dim = base.action_manager.total_action_dim
    zero_actions = torch.zeros((base.num_envs, action_dim), device=base.device)

    printed = False
    saved = False
    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            env.step(zero_actions)
        step += 1

        # print sensor data shapes once, after a few steps so buffers are populated
        if not printed and step > 5:
            cam = base.scene.sensors["camera"]
            scan = base.scene.sensors["height_scanner"]
            depth = cam.data.output["distance_to_image_plane"]
            print(f"[debug] height_scanner ray_hits_w: {tuple(scan.data.ray_hits_w.shape)}")
            print(f"[debug] camera depth output:       {tuple(depth.shape)}  "
                  f"(min={float(depth[torch.isfinite(depth)].min()):.2f} "
                  f"max={float(depth[torch.isfinite(depth)].max()):.2f})\n")
            printed = True

            if args_cli.save_depth and not saved:
                _save_depth(depth[0], args_cli.out)
                saved = True

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
