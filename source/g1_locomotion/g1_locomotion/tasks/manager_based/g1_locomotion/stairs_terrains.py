# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generator configuration for the G1 stair-climbing task.

This is a stairs-weighted variant of Isaac Lab's ``ROUGH_TERRAINS_CFG``. With
``curriculum=True`` the per-sub-terrain ranges interpolate from *easy* (row 0,
near-flat) to *hard* (last row, tall steps). Combined with
``max_init_terrain_level=0`` in the scene and the ``terrain_levels_vel``
curriculum, every robot starts on the flat/easy level and is promoted to harder
stairs only as it learns to travel — exactly the "start flat, gradually harder"
schedule requested.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# Difficulty 0 (row 0) -> near-flat; difficulty 1 (last row) -> tall steps.
# Step heights start very low (~2 cm, effectively flat) so early training behaves
# like the flat task, then ramp up to challenging 18 cm steps.
G1_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # difficulty levels (curriculum climbs these)
    num_cols=20,  # terrain variations per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # upward stairs — the primary skill
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.02, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # downward stairs (inverted pyramid) — descending is a distinct skill
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.02, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # mild random roughness for robustness
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        # gentle slopes
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Stairs-weighted, curriculum-enabled terrains for G1 stair climbing."""
