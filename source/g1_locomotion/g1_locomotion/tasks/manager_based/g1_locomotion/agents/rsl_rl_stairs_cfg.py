# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL runner configs for the G1 stair-climbing teacher-student pipeline.

* ``G1StairTeacherPPORunnerCfg`` — PPO teacher on the privileged height scanner (``OnPolicyRunner``).
* ``G1StairDistillRunnerCfg``    — depth-camera student distilled from the teacher (``DistillationRunner``).

The student run loads the teacher checkpoint via ``--load_run`` (train.py auto-loads it for distillation).
The teacher network inside the student-teacher module must match the PPO actor architecture, so
``teacher_hidden_dims`` mirrors the teacher's ``actor_hidden_dims`` ([512, 256, 128]).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.distillation_cfg import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class G1StairTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Teacher: PPO with the privileged height-scanner observation."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_stairs_teacher"
    # actor sees the 'policy' group (proprio + height scan); critic shares it
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1StairDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Student: distill the depth-camera policy from the pretrained height-scan teacher."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_stairs_student"
    # student ('policy') sees proprio + depth; teacher sees proprio + height scan
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # student must have enough capacity for the flattened depth image
        student_hidden_dims=[512, 256, 128],
        # MUST match the teacher PPO actor_hidden_dims so the loaded weights map 1:1
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
