# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""The names of declared experiments grouped by category.

This module keeps experiment names separate from the configuration for the
experiments. The reason for doing this is so that we can import the configurations
selectively to avoid importing uninstalled dependencies (e.g., not installing a
specific simulator).

The use of dataclasses assists in raising early errors when experiment names defined
here and the corresponding experiment configurations drift apart. For additional
discussion, see: https://github.com/thousandbrainsproject/tbp.monty/pull/153.
"""

from dataclasses import dataclass, fields

from benchmarks.configs.follow_ups.names import NAMES as FOLLOW_UP_NAMES

NAMES = []

NAMES.extend(FOLLOW_UP_NAMES)


@dataclass
class MontyWorldExperiments:
    world_image_from_stream_on_scanned_model: dict
    world_image_on_scanned_model: dict
    dark_world_image_on_scanned_model: dict
    bright_world_image_on_scanned_model: dict
    hand_intrusion_world_image_on_scanned_model: dict
    multi_object_world_image_on_scanned_model: dict


NAMES.extend(field.name for field in fields(MontyWorldExperiments))


@dataclass
class MontyWorldHabitatExperiments:
    randrot_noise_sim_on_scan_monty_world: dict
    randrot_noise_dist_sim_on_scan_tbp_robot_lab_obj: dict
    randrot_noise_surf_sim_on_scan_tbp_robot_lab_obj: dict


NAMES.extend(field.name for field in fields(MontyWorldHabitatExperiments))


@dataclass
class PretrainingExperiments:
    supervised_pre_training_base: dict
    supervised_pre_training_5lms: dict
    supervised_pre_training_5lms_all_objects: dict
    only_surf_agent_training_10obj: dict
    only_surf_agent_training_10simobj: dict
    only_surf_agent_training_allobj: dict
    only_surf_agent_training_numenta_lab_obj: dict
    only_surf_agent_training_tbp_robot_lab_obj: dict


NAMES.extend(field.name for field in fields(PretrainingExperiments))


@dataclass
class YcbExperiments:
    base_config_10distinctobj_dist_agent: dict
    base_config_10distinctobj_surf_agent: dict
    randrot_noise_10distinctobj_dist_agent: dict
    randrot_noise_10distinctobj_dist_on_distm: dict
    randrot_noise_10distinctobj_surf_agent: dict
    randrot_10distinctobj_surf_agent: dict
    randrot_noise_10distinctobj_5lms_dist_agent: dict
    base_10simobj_surf_agent: dict
    randrot_noise_10simobj_surf_agent: dict
    randrot_noise_10simobj_dist_agent: dict
    randomrot_rawnoise_10distinctobj_surf_agent: dict
    base_10multi_distinctobj_dist_agent: dict
    surf_agent_unsupervised_10distinctobj: dict
    surf_agent_unsupervised_10distinctobj_noise: dict
    surf_agent_unsupervised_10simobj: dict
    base_77obj_dist_agent: dict
    base_77obj_surf_agent: dict
    randrot_noise_77obj_surf_agent: dict
    randrot_noise_77obj_dist_agent: dict
    randrot_noise_77obj_5lms_dist_agent: dict


NAMES.extend(field.name for field in fields(YcbExperiments))


@dataclass
class UnsupervisedInferenceExperiments:
    unsupervised_inference_distinctobj_surf_agent: dict
    unsupervised_inference_distinctobj_dist_agent: dict


NAMES.extend(field.name for field in fields(UnsupervisedInferenceExperiments))


@dataclass
class MyExperiments:
    # Add your experiment names here
    pass


NAMES.extend(field.name for field in fields(MyExperiments))
