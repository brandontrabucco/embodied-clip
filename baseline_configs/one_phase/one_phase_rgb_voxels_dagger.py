from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.sensors import (
    ExpertRaysSensor,
    ExpertObjectsSensor,
    FeatureMapSensor
)
from rearrange.baseline_models import (
    ResNetRearrangeActorCriticRNNWithVoxels,
)
from typing import Sequence
from allenact.base_abstractions.sensor import Sensor
from torch import nn
import gym.spaces


class OnePhaseRGBVoxelsDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [*super(OnePhaseRGBVoxelsDaggerExperimentConfig, 
                       cls).sensors(), FeatureMapSensor()]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBVoxelsDagger_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBVoxelsDaggerExperimentConfig, 
                       cls)._use_label_to_get_training_params()
        params["lr"] = 3e-4
        params["num_train_processes"] = 1
        return params

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResNetRearrangeActorCriticRNNWithVoxels(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            hidden_size=512,
            positional_features=3,
            voxel_features=256,
            num_octaves=8,
            start_octave=-5,
            num_transformer_layers=3,
            dim_head=64,
            dropout=0.0,
            activation='gelu',
            layer_norm_eps=1e-5)
