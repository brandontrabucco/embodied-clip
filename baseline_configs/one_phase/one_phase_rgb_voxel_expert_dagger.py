from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.sensors import (
    ExpertRaysSensor,
    ExpertObjectsSensor,
    FeatureMapSensor,
    ExpertVoxelSensor
)
from rearrange.baseline_models import (
    ResNetRearrangeActorCriticRNNWithVoxelExpert,
)
from typing import Sequence
from allenact.base_abstractions.sensor import Sensor
from torch import nn
import gym.spaces


class OnePhaseRGBVoxelExpertDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [*super(OnePhaseRGBVoxelExpertDaggerExperimentConfig, 
                       cls).sensors(), ExpertVoxelSensor()]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBVoxelExpertDagger"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBVoxelExpertDaggerExperimentConfig, 
                       cls)._use_label_to_get_training_params()

        params["lr"] = 1e-4

        return params

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResNetRearrangeActorCriticRNNWithVoxelExpert(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            hidden_size=512,
            positional_features=3,
            voxel_features=256,
            num_octaves=8,
            start_octave=-5)
