# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
	LoggingConfig,
    MotorSystemConfigInformedNoTrans,
	MotorSystemConfigInformedNoTransStepS1,
    MotorSystemConfigInformedNoTransStepS3,
    MotorSystemConfig,
    MotorSystemConfigNaiveScanSpiral,
    MotorSystemConfigSurface,
	PatchAndViewMontyConfig,
	PretrainLoggingConfig,
    CSVLoggingConfig,
    SurfaceAndViewMontyConfig,
    MontyArgs,

)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
	ExperimentArgs,
	OmniglotDataloaderArgs,   
	OmniglotDatasetArgs,
    get_omniglot_train_dataloader,
    get_omniglot_eval_dataloader,
    RandomRotationObjectInitializer,
    EnvironmentDataloaderPerObjectArgs,
    MnistDatasetArgs,
    get_mnist_train_dataloader,
    get_mnist_eval_dataloader

)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
	MontyObjectRecognitionExperiment,
	MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.evidence_matching import (
	EvidenceGraphLM,
	MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.sensor_modules import (
	DetailedLoggingSM,
	HabitatDistantPatchSM,  

)

from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)


# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)


omniglot_sensor_module_config = dict(
	sensor_module_0=dict(
    	sensor_module_class=HabitatDistantPatchSM,
    	sensor_module_args=dict(
        	sensor_module_id="patch",
        	features=[         
                "rgba",       
            	"pose_vectors",
            	"pose_fully_defined",
            	"on_object",
            	"principal_curvatures_log",
        	],
        	save_raw_obs=False,
        	# Need to set this lower since curvature is generally lower
        	pc1_is_pc2_threshold=1,
    	),
	),
	sensor_module_1=dict(
    	sensor_module_class=DetailedLoggingSM,
    	sensor_module_args=dict(
        	sensor_module_id="view_finder",
        	save_raw_obs=False,
    	),
	),
)

monty_models_dir = os.getenv("MONTY_MODELS")

pretrain_dir = os.path.expanduser(os.path.join(monty_models_dir, "omniglot"))

omniglot_training = dict(
	experiment_class=MontySupervisedObjectPretrainingExperiment,
	experiment_args=ExperimentArgs(
    	n_train_epochs=1,
    	do_eval=False,
	),
	logging_config=PretrainLoggingConfig(
    	#output_dir=pretrain_dir,
        output_dir = "omniglot/log",
	),
    # logging_config=CSVLoggingConfig(
    #     output_dir="omniglot/log",
    #     monty_log_level="BASIC",
    #     monty_handlers=[BasicCSVStatsHandler],                 
    # ),
	monty_config=PatchAndViewMontyConfig(
    	# Take 1 step at a time, following the drawing path of the letter
    	#motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),
        monty_class=MontyForEvidenceGraphMatching,
    	sensor_module_configs=omniglot_sensor_module_config,
	),
	dataset_class=ED.EnvironmentDataset,
	dataset_args=OmniglotDatasetArgs(),
	train_dataloader_class=ED.OmniglotDataLoader,
	# Train on the first version of each character (there are 20 drawings for each
	# character in each alphabet, here we see one of them). The default
	# OmniglotDataloaderArgs specify alphabets = [0, 0, 0, 1, 1, 1] and
    # characters = [1, 2, 3, 1, 2, 3]) so in the first episode we will see version 1
	# of character 1 in alphabet 0, in the next episode version 1 of character 2 in
	# alphabet 0, and so on.
	train_dataloader_args=OmniglotDataloaderArgs(versions=[11,2,3,4,5,6]),
    #train_dataloader_args=OmniglotTrainDataloaderArgs(),
    #train_dataloader = get_omniglot_train_dataloader(alphabet_ids=[0,1,2], num_versions=1),
)

omniglot_inference = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        #model_name_or_path=pretrain_dir + "/omniglot_training/pretrained/",
        model_name_or_path = "omniglot/log/omniglot_training/pretrained",
        do_train=False,
        n_eval_epochs=1,
    ),
    logging_config=LoggingConfig
    (
        output_dir="omniglot/log",
        monty_log_level="BASIC",
        monty_handlers=[BasicCSVStatsHandler], 
    ),
    # logging_config=CSVLoggingConfig(
    #     output_dir="omniglot/log",
    #     monty_log_level="BASIC",
    #     monty_handlers=[BasicCSVStatsHandler],                 
    # ),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    # xyz values are in larger range so need to increase mmd
                    max_match_distance=5,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up, so they are not useful
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 0, 1],
                        }
                    },
                    # We assume the letter is presented upright
                    initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=omniglot_sensor_module_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=OmniglotDatasetArgs(),
    eval_dataloader_class=ED.OmniglotDataLoader,
    # Using version 1 means testing on the same version of the character as trained.
    # Version 2 is a new drawing of the previously seen characters. In this small test
    # setting these are 3 characters from 2 alphabets.
    eval_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    #eval_dataloader_args=OmniglotDataloaderArgs(versions=[2, 2, 2, 2, 2, 2]),
    #eval_dataloader_args=OmniglotEvalDataloaderArgs(),
    #eval_dataloader_args = get_omniglot_eval_dataloader( alphabet_ids=[0,1,2], start_at_version=0,num_versions=20),
)

# Set up config for an evidence graph learning module.
learning_module_0 = dict(
    learning_module_class=EvidenceGraphLM,
    learning_module_args=dict(
        max_match_distance=0.01,
        # Tolerances within which features must match stored values in order to add
        # evidence to a hypothesis.
        tolerances={
            "patch": {
                "hsv": np.array([0.05, 0.1, 0.1]),
                "principal_curvatures_log": np.ones(2),
            }
        },
        feature_weights={
            "patch": {
                # Weighting saturation and value less since these might change
                # under different lighting conditions.
                "hsv": np.array([1, 0.5, 0.5]),
            }
        },
        x_percent_threshold=20,
        # Thresholds to use for when two points are considered different enough to
        # both be stored in memory.
        graph_delta_thresholds=dict(
            patch=dict(
                distance=0.01,
                pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                principal_curvatures_log=[1.0, 1.0],
                hsv=[0.1, 1, 1],
            )
        ),
        # object_evidence_th sets a minimum threshold on the amount of evidence we have
        # for the current object in order to converge; while we can also set min_steps
        # for the experiment, this puts a more stringent requirement that we've had
        # many steps that have contributed evidence.
        object_evidence_threshold=100,
        # Symmetry evidence (indicating possibly symmetry in rotations) increments a lot
        # after 100 steps and easily reaches the default required evidence. The below
        # parameter value partially addresses this, altough we note these are temporary
        # fixes and we intend to implement a more principled approach in the future.
        required_symmetry_evidence=20,
        max_nneighbors=5,
    ),
)
learning_module_configs = dict(learning_module_0=learning_module_0)


# Specify a name for the model.
model_name = "surf_agent_2obj_unsupervised"

# Here we specify which objects to learn. We are going to use the mug and bowl
# from the YCB dataset.
object_names = ["mug", "bowl"]

# The config dictionary for the unsupervised learning experiment.
surf_agent_2obj_unsupervised = dict(
    # Set up unsupervised experiment.
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        # Not running eval here. The only difference between training and evaluation
        # is that during evaluation, no models are updated.
        do_eval=False,
        n_train_epochs=2,
        max_train_steps=2000,
        max_total_steps=5000,
    ),
    logging_config=CSVLoggingConfig(
        python_log_level="INFO",
        output_dir="surf_unsuper/log",
        run_name=model_name,
    ),
    # Set up monty, including LM, SM, and motor system. We will use the default
    # sensor modules (1 habitat surface patch, one logging view finder), motor system,
    # and connectivity matrices given by `SurfaceAndViewMontyConfig`.
    monty_config=SurfaceAndViewMontyConfig(
        #motor_system_config=MotorSystemConfigNaiveScanSpiral(), # by skj
        # Take 1000 exploratory steps after recognizing the object to collect more
        # information about it. Require at least 100 steps before recognizing an object
        # to avoid early misclassifications when we have few objects in memory.
        monty_args=MontyArgs(num_exploratory_steps=1000, min_train_steps=100),
        learning_module_configs=learning_module_configs,
    ),
    # Set up the environment and agent.
    dataset_class=ED.EnvironmentDataset,
    dataset_args=SurfaceViewFinderMountHabitatDatasetArgs(),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
    # Doesn't get used, but currently needs to be set anyways.
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=EnvironmentDataloaderPerObjectArgs(
        object_names=object_names,
        object_init_sampler=RandomRotationObjectInitializer(),
    ),
)


mnist_pretrain_dir = os.path.expanduser(os.path.join(monty_models_dir, "mnist"))

mnist_sensor_module_config = dict(
	sensor_module_0=dict(
    	sensor_module_class=HabitatDistantPatchSM,
    	sensor_module_args=dict(
        	sensor_module_id="patch",
        	features=[ 
                "rgba",               
            	"pose_vectors",
            	"pose_fully_defined",
            	"on_object",
            	"principal_curvatures_log",
        	],
        	save_raw_obs=False,
        	# Need to set this lower since curvature is generally lower
        	pc1_is_pc2_threshold=1,
    	),
	),
	sensor_module_1=dict(
    	sensor_module_class=DetailedLoggingSM,
    	sensor_module_args=dict(
        	sensor_module_id="view_finder",
        	save_raw_obs=False,
    	),
	),
)


mnist_training = dict(
	experiment_class=MontySupervisedObjectPretrainingExperiment,
	experiment_args=ExperimentArgs(
    	n_train_epochs=1,
    	do_train = True,
        do_eval=False,
	),
    logging_config=PretrainLoggingConfig(
    	#output_dir=pretrain_dir,
        output_dir = "mnist/log",
	),
    # logging_config=CSVLoggingConfig(
    #     output_dir="mnist/log",
    #     python_log_level="INFO",
    #     #monty_handlers=[BasicCSVStatsHandler],                 
    # ),

	monty_config=PatchAndViewMontyConfig(
    	# Take 1 step at a time, following the drawing path of the letter
    	#motor_system_config=MotorSystemConfigInformedNoTrans(),
        #motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),                
        monty_class=MontyForEvidenceGraphMatching,
    #     learning_module_configs=dict(
    #         learning_module_0=dict( 
    #             learning_module_class=EvidenceGraphLM,
    #             learning_module_args=dict(               
    #                 max_match_distance=1,
    #                 tolerances={
    #                     "patch": {
    #                         "principal_curvatures_log": np.ones(2),
    #                         "pose_vectors": np.ones(3) * 45,
    #                     }
    #                 },
    #                 # Point normal always points up, so they are not useful
    #                 feature_weights={
    #                     "patch": {
    #                         "pose_vectors": [0, 1, 0],
    #                     }
    #                 },
    #                 # We assume the letter is presented upright
    #                 #initial_possible_poses=[[0, 0, 0]],
    #             ),
    #         )
    #     ),
     	sensor_module_configs=mnist_sensor_module_config,
	),
	dataset_class=ED.EnvironmentDataset,
	dataset_args=MnistDatasetArgs(),
	train_dataloader_class=ED.MnistDataLoader,	
    train_dataloader_args = get_mnist_train_dataloader(start_at_version = 0, number_ids = np.arange(0,10), num_versions=1)
)

mnist_inference = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        #model_name_or_path=pretrain_dir + "/mnist_training/",
        model_name_or_path = "mnist/log/mnist_training/pretrained",
        do_train=False,
        do_eval=True,
        n_train_epochs=3,
        n_eval_epochs=1,
        #max_train_steps=100,
        #max_eval_steps=100,
        max_total_steps=6000,
    ),
    #logging_config=LoggingConfig(),
    logging_config=CSVLoggingConfig(
            output_dir="mnist/log",
            python_log_level="DEBUG",
            #monty_handlers=[BasicCSVStatsHandler],                 
        ),

    monty_config=PatchAndViewMontyConfig(
        #motor_system_config = MotorSystemConfigInformedNoTrans(),
        #motor_system_config = MotorSystemConfigInformedNoTransStepS3(),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(                 
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(              
                    #x_percent_threshold=20, 
                    max_match_distance=1,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up, so they are not useful
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    #initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=mnist_sensor_module_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=MnistDatasetArgs(),
    train_dataloader_class=ED.MnistDataLoader,
	#train_dataloader_args=MnistDataloaderArgs(),
    train_dataloader_args = get_mnist_train_dataloader(start_at_version = 0, number_ids = np.arange(0,2), num_versions=3),
    eval_dataloader_class=ED.MnistDataLoader,
    #eval_dataloader_args=MnistEvalDataloaderArgs(),
    eval_dataloader_args = get_mnist_eval_dataloader(start_at_version = 0, number_ids = np.arange(9,10), num_versions=10)
)

mnist_unsuper = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(        
        
        do_train=False,
        do_eval=True,
        n_train_epochs=3,
        n_eval_epochs=1,
        #max_train_steps=100,
        #max_eval_steps=100,
        max_total_steps=6000,
    ),
    #logging_config=LoggingConfig(),
    logging_config=CSVLoggingConfig(
            output_dir="mnist/log",
            python_log_level="DEBUG",
            #monty_handlers=[BasicCSVStatsHandler],                 
        ),

    monty_config=PatchAndViewMontyConfig(
        #motor_system_config = MotorSystemConfigInformedNoTrans(),
        #motor_system_config = MotorSystemConfigInformedNoTransStepS3(),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(                 
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(              
                    #x_percent_threshold=20, 
                    max_match_distance=1,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up, so they are not useful
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    #initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=mnist_sensor_module_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=MnistDatasetArgs(),
    train_dataloader_class=ED.MnistDataLoader,
	#train_dataloader_args=MnistDataloaderArgs(),
    train_dataloader_args = get_mnist_train_dataloader(start_at_version = 0, number_ids = np.arange(0,2), num_versions=3),
    eval_dataloader_class=ED.MnistDataLoader,
    #eval_dataloader_args=MnistEvalDataloaderArgs(),
    eval_dataloader_args = get_mnist_eval_dataloader(start_at_version = 0, number_ids = np.arange(9,10), num_versions=10)
)


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    omniglot_training=omniglot_training,
	omniglot_inference=omniglot_inference,
    surf_agent_2obj_unsupervised = surf_agent_2obj_unsupervised,
    mnist_training = mnist_training,
    mnist_inference = mnist_inference,
    mnist_unsuper = mnist_unsuper,
)
CONFIGS = asdict(experiments)
