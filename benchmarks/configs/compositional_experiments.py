import copy
import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.defaults import (
    default_all_noise_params,
    default_all_noisy_sensor_module,
    default_evidence_1lm_config,
    default_evidence_lm_config,
    default_feature_weights,
    default_tolerance_values,
    default_tolerances,
    min_eval_steps,
    pretrained_dir,
)
from benchmarks.configs.names import CompositionalExperiments
from tbp.monty.frameworks.config_utils.config_args import (
    CSVLoggingConfig,
    FiveLMMontySOTAConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    MotorSystemConfigInformedGoalStateDriven,
    MotorSystemConfigNaiveScanSpiral,
    ParallelEvidenceLMLoggingConfig,
    PatchAndViewFartherAwaySOTAMontyConfig,
    PatchAndViewSOTAMontyConfig,
    SurfaceAndViewMontyConfig,
    SurfaceAndViewSOTAMontyConfig,
    TwoLMStackedMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderMultiObjectArgs,
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    RandomRotationObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import (
    DISTINCT_OBJECTS,
    SIMILAR_OBJECTS,
)
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.goal_state_generation import EvidenceGoalStateGenerator
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat.configs import (
    EnvInitArgsTwoLMDistantStackedMount,
    FiveLMMountHabitatDatasetArgs,
    NoisySurfaceViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    PatchViewFinderMultiObjectMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
    TwoLMStackedDistantMountHabitatDatasetArgs,
)

# 14 unique rotations that give good views of the object. Same rotations used
# for supervised pretraining.
# test_rotations_all = get_cube_face_and_corner_views_rotations()
test_rotations_all = [[0.0, 0.0, 0.0]]

min_eval_steps = 20

# Limited number of rotations to use for quicker evaluation when doing longer
# runs with all 77 YCB objects.
test_rotations_3 = test_rotations_all[:3]

model_path_compositional_logos = os.path.join(
    pretrained_dir,
    "supervised_pre_training_compositional_objects_with_logos/pretrained/",
)

model_path_individual_objects = os.path.join(
    pretrained_dir,
    "supervised_pre_training_compositional_logos/pretrained/",
)

model_path_compositional_models = os.path.join(
    pretrained_dir,
    "partial_supervised_pre_training_comp_objects/pretrained/",
)


two_stacked_constrained_lms_inference_config = dict(
    learning_module_0=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,
            tolerances={
                "patch_0": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            # Note graph-delta-thresholds are not used for grid-based models
            feature_weights={},
            max_graph_size=0.3,
            num_model_voxels_per_dim=200,
            max_nodes_per_graph=2000,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-testing goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.03,  # Distance from the object to the
                # agent that is considered "close enough" to the object
            ),
        ),
    ),
    learning_module_1=dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            max_match_distance=0.01,  # TODO: C - Scale with receptive field size
            tolerances={
                "patch_1": {
                    "hsv": np.array([0.1, 1, 1]),
                    "principal_curvatures_log": np.ones(2),
                },
                # object Id currently is an int representation of the strings
                # in the object label so we keep this tolerance high. This is
                # just until we have added a way to encode object ID with some
                # real similarity measure.
                "learning_module_0": {"object_id": 1},
            },
            feature_weights={"learning_module_0": {"object_id": 1}},
            max_graph_size=0.4,
            num_model_voxels_per_dim=200,
            max_nodes_per_graph=2000,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-testing goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a hypothesis
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.03,  # Distance from the object to the
                # agent that is considered "close enough" to the object
            ),
        ),
    ),
)

OBJECT_WITH_LOGOS = [
    "002_cube_tbp_horz",
    "004_cube_numenta_horz",
    "007_disk_tbp_horz",
    "009_disk_numenta_horz",
]


base_config_cube_disk_logos_dist_agent = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_logos,
        n_eval_epochs=len(test_rotations_all),
        # show_sensor_output=True,
    ),
    logging_config=ParallelEvidenceLMLoggingConfig(
        wandb_group="benchmark_experiments",
        # Comment in for quick debugging (turns of wandb and increases logging)
        # wandb_handlers=[],
        # python_log_level="INFO",
    ),
    monty_config=TwoLMStackedMontyConfig(
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
        learning_module_configs=two_stacked_constrained_lms_inference_config,
        motor_system_config=MotorSystemConfigInformedGoalStateDriven(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=TwoLMStackedDistantMountHabitatDatasetArgs(
        env_init_args=EnvInitArgsTwoLMDistantStackedMount(
            data_path=os.path.join(os.environ["MONTY_DATA"], "compositional_objects")
        ).__dict__,
    ),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(OBJECT_WITH_LOGOS), object_list=OBJECT_WITH_LOGOS
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)

INDIVIDUAL_OBJECTS = ["001_cube", "006_disk", "021_logo_tbp", "022_logo_numenta"]

base_config_individual_objects_dist_agent = copy.deepcopy(
    base_config_cube_disk_logos_dist_agent
)
base_config_individual_objects_dist_agent.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_individual_objects,
        n_eval_epochs=len(test_rotations_all),
        show_sensor_output=True,
    ),
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(
            0, len(INDIVIDUAL_OBJECTS), object_list=INDIVIDUAL_OBJECTS
        ),
        object_init_sampler=PredefinedObjectInitializer(
            rotations=test_rotations_all,
        ),
    ),
)


cube_disk_logos_with_pretrained_models = copy.deepcopy(
    base_config_cube_disk_logos_dist_agent
)
cube_disk_logos_with_pretrained_models.update(
    experiment_args=EvalExperimentArgs(
        model_name_or_path=model_path_compositional_models,
        n_eval_epochs=len(test_rotations_all),
        min_lms_match=2,
    ),
)

experiments = CompositionalExperiments(
    base_config_cube_disk_logos_dist_agent=base_config_cube_disk_logos_dist_agent,
    base_config_individual_objects_dist_agent=base_config_individual_objects_dist_agent,
    cube_disk_logos_with_pretrained_models=cube_disk_logos_with_pretrained_models,
)
CONFIGS = asdict(experiments)
