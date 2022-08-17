from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.sensors import (
    ExpertRaysSensor,
    ExpertObjectsSensor
)
from rearrange.baseline_models import (
    ResNetRearrangeActorCriticRNNWithRays,
)
from typing import Sequence
from allenact.base_abstractions.sensor import Sensor
from torch import nn
import gym.spaces


class OnePhaseRGBNeRFClipResNet50DaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [*super(OnePhaseRGBNeRFClipResNet50DaggerExperimentConfig, 
                       cls).sensors(), ExpertObjectsSensor()]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBNeRFClipResNet50Dagger_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def _training_pipeline_info(cls, **kwargs):
        params = super(OnePhaseRGBNeRFClipResNet50DaggerExperimentConfig, 
                       cls)._training_pipeline_info()
        params["lr"] = 1e-4
        return params

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResNetRearrangeActorCriticRNNWithRays(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            hidden_size=512,
            positional_features=3,
            patch_features=768,
            num_octaves=8,
            start_octave=-5,
            num_transformer_layers=3,
            dim_head=64,
            dropout=0.0,
            activation='gelu',
            layer_norm_eps=1e-5)
