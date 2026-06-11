# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 stair-climbing environment (teacher / student) built for teacher-student distillation.

Two environments share this file:

* ``G1StairTeacherEnvCfg`` — observes a *privileged* top-down **height scanner**. Trained with PPO
  (``OnPolicyRunner``). Fast and stable; this is the *teacher*.
* ``G1StairStudentEnvCfg`` — the ``policy`` (student) observes only a head-mounted **RealSense D435
  depth camera**; a second ``teacher`` observation group reproduces the teacher's height-scan input so
  the pretrained teacher network loads 1:1. Trained with ``DistillationRunner`` (``--load_run <teacher>``).

The flat task (``g1_locomotion_env_cfg.py`` / ``Template-G1-Locomotion-v0``) is left untouched; this file
reuses its commands / actions / events / rewards / terminations.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, RayCasterCameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .stairs_terrains import G1_STAIRS_TERRAINS_CFG

# terrain curriculum lives in the velocity task's mdp (not the core mdp)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel

# reuse the flat task's MDP building blocks unchanged
from .g1_locomotion_env_cfg import ActionsCfg, CommandsCfg, EventCfg, RewardsCfg, TerminationsCfg

from g1_locomotion.robots.unitree_g1 import G1_CFG  # isort:skip


##
# Observation helper
##


def flat_depth_image(env, sensor_cfg, data_type="distance_to_image_plane", normalize=True):
    """Depth image flattened to [num_envs, H*W].

    The observation manager cannot concatenate a 3-D image term with 1-D proprio terms, so the depth
    image is flattened here before it joins the student's ``policy`` group (MLP input, no CNN).
    """
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    return img.reshape(img.shape[0], -1)


##
# RealSense D435 depth camera parameters (head-mounted, looking down)
##
# Down-pitch of 42.4 deg about the head +Y axis (world convention: forward=+X, up=+Z).
# Quaternion (w, x, y, z) = (cos(theta/2), 0, sin(theta/2), 0), theta = 42.4 deg.
_CAM_DOWN_PITCH_QUAT = (0.93251, 0.0, 0.36124, 0.0)
# Depth image resolution (downsampled — flattened straight into the MLP, no CNN).
_CAM_WIDTH = 64
_CAM_HEIGHT = 36
# focal_length + horizontal_aperture chosen so HFOV ~= 87 deg (D435 depth), with the
# 64x36 aspect giving VFOV ~= 58 deg. FOV is resolution-independent.
_CAM_FOCAL_LENGTH = 24.0
_CAM_HORIZONTAL_APERTURE = 45.5
_CAM_MAX_RANGE = 5.0  # metres — D435 usable depth range (values beyond are clipped to max)


##
# Scene
##


@configclass
class G1StairSceneCfg(InteractiveSceneCfg):
    """Scene with generated stairs terrain, a height scanner, and a head depth camera."""

    # generated stairs terrain (curriculum-enabled)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=G1_STAIRS_TERRAINS_CFG,
        max_init_terrain_level=0,  # everyone starts on the easiest (near-flat) level
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # privileged height scanner (TEACHER) — top-down grid of rays from the torso
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # head-mounted RealSense D435 depth camera (STUDENT) — ray-cast depth, angled down
    camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.05),  # forward + slightly up on the head (verify with debug script)
            rot=_CAM_DOWN_PITCH_QUAT,
            convention="world",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=_CAM_FOCAL_LENGTH,
            horizontal_aperture=_CAM_HORIZONTAL_APERTURE,
            width=_CAM_WIDTH,
            height=_CAM_HEIGHT,
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=_CAM_MAX_RANGE,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact sensor (rewards + fall termination)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Observation building blocks
##


@configclass
class _ProprioCfg(ObsGroup):
    """Proprioceptive observation terms shared by the student policy and the teacher network.

    Kept identical (same terms, same order) across the teacher's ``policy`` group and the student's
    ``teacher`` group so the pretrained teacher MLP weights map 1:1 during distillation.
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class _HeightScanPolicyCfg(_ProprioCfg):
    """Proprioception + privileged height scan (teacher input)."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )


@configclass
class _DepthPolicyCfg(_ProprioCfg):
    """Proprioception + head depth image (student input). Image is flattened into the MLP."""

    depth = ObsTerm(
        func=flat_depth_image,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane", "normalize": True},
        noise=Unoise(n_min=-0.05, n_max=0.05),
        clip=(0.0, _CAM_MAX_RANGE),
    )


@configclass
class TeacherObservationsCfg:
    """Observation groups for the teacher (PPO) environment."""

    policy: _HeightScanPolicyCfg = _HeightScanPolicyCfg()


@configclass
class StudentObservationsCfg:
    """Observation groups for the student (distillation) environment.

    ``policy`` = student (depth), ``teacher`` = the height-scan input the pretrained teacher expects.
    """

    policy: _DepthPolicyCfg = _DepthPolicyCfg()
    teacher: _HeightScanPolicyCfg = _HeightScanPolicyCfg()


##
# Curriculum
##


@configclass
class CurriculumCfg:
    """Curriculum terms — promote/demote robots across terrain difficulty levels."""

    terrain_levels = CurrTerm(func=terrain_levels_vel)


##
# Environment configs
##


@configclass
class G1StairTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """Teacher: privileged height-scanner policy trained with PPO on stairs."""

    # Scene
    scene: G1StairSceneCfg = G1StairSceneCfg(num_envs=4096, env_spacing=2.5)
    # MDP
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 200 Hz physics, 50 Hz control — matches the G1 actuator gains
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # stairs need the torso to pitch on steps — relax the fall-orientation limit and
        # forward-only commands (lateral stair walking is not the goal)
        self.terminations.bad_orientation.params["limit_angle"] = 1.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # push the robot less while it is learning to climb
        self.events.push_robot = None

        # sensor update periods (tick with the physics/control step)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # enable the terrain generator curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class G1StairStudentEnvCfg(G1StairTeacherEnvCfg):
    """Student: head depth-camera policy distilled from the teacher."""

    observations: StudentObservationsCfg = StudentObservationsCfg()


@configclass
class G1StairTeacherEnvCfg_PLAY(G1StairTeacherEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # fixed forward command, no corruption / pushes for clean evaluation
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # smaller terrain, no curriculum churn during play
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1StairStudentEnvCfg_PLAY(G1StairStudentEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
