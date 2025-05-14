# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import copy
import os
from dataclasses import asdict

from benchmarks.configs.defaults import (
    default_all_noisy_sensor_module,
    default_evidence_1lm_config,
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import MontyWorldHabitatExperiments
from benchmarks.configs.pretraining_experiments import (
    TBP_ROBOT_LAB_OBJECTS,
    tbp_robot_lab_dataset_args,
)
from benchmarks.configs.ycb_experiments import (
    default_all_noisy_surf_agent_sensor_module,
)
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedNoTransStepS20,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewMontyConfig,
    SurfaceAndViewSOTAMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    RandomRotationObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.two_d_data import NUMENTA_OBJECTS
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.sensor_modules import DetailedLoggingSM
from tbp.monty.simulators.habitat.configs import (
    PatchViewFinderMontyWorldMountHabitatDatasetArgs,
)

test_rotations_one = [[0, 0, 0]]

model_path_numenta_lab_obj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_numenta_lab_obj/pretrained/",
)
model_path_tbp_robot_lab_obj = os.path.join(
    pretrained_dir,
    "surf_agent_1lm_tbp_robot_lab_obj/pretrained/",
)

# More challenging evaluation on photogrammetry objects; serves as a baseline
# for viewing these objects from a fixed, single view, but in simulation rather than
# with real-world data
# Note we therefore use the basic distant-agent policy without hypothesis-driven
# actions, to be more comparable to the constraints of inference on real-world data
randrot_noise_sim_on_scan_monty_world = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_numenta_lab_obj,
        n_eval_epochs=10,
        max_eval_steps=500,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(wandb_group="benchmark_experiments"),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=PatchViewFinderMontyWorldMountHabitatDatasetArgs(),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 12, object_list=NUMENTA_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)


tbp_robot_lab_dist_dataset_args = PatchViewFinderMontyWorldMountHabitatDatasetArgs()
tbp_robot_lab_dist_dataset_args.env_init_args["data_path"] = os.path.join(
    os.environ["MONTY_DATA"], "tbp_robot_lab"
)

randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj = copy.deepcopy(
    randrot_noise_sim_on_scan_monty_world
)
randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_tbp_robot_lab_obj,
        n_eval_epochs=10,
    ),
    monty_config=PatchAndViewMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # Not using the hypothesis-driven motor system here, because the comparison on
        # the iPad images can't move around the object.
        motor_system_config=MotorSystemConfigInformedNoTransStepS20(),
    ),
    dataset_args=tbp_robot_lab_dist_dataset_args,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=TBP_ROBOT_LAB_OBJECTS),
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)

randrot_noise_surf_sim_on_scan_tbp_robot_lab_obj = copy.deepcopy(
    randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj
)
randrot_noise_surf_sim_on_scan_tbp_robot_lab_obj.update(
    monty_config=SurfaceAndViewSOTAMontyConfig(
        sensor_module_configs=dict(
            sensor_module_0=default_all_noisy_surf_agent_sensor_module,
            sensor_module_1=dict(
                sensor_module_class=DetailedLoggingSM,
                sensor_module_args=dict(
                    sensor_module_id="view_finder",
                    save_raw_obs=False,
                ),
            ),
        ),
        learning_module_configs=default_evidence_1lm_config,
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        # In our real-world experiments the sensor is now able to move around the object
        # so we also allow this here for the simlation comparison.
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
    ),
    dataset_args=tbp_robot_lab_dataset_args,
)

experiments = MontyWorldHabitatExperiments(
    randrot_noise_sim_on_scan_monty_world=randrot_noise_sim_on_scan_monty_world,
    randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj=randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj,
    randrot_noise_surf_sim_on_scan_tbp_robot_lab_obj=randrot_noise_surf_sim_on_scan_tbp_robot_lab_obj,
)
CONFIGS = asdict(experiments)
