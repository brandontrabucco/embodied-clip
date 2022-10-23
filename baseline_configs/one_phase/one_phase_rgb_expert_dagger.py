from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.map_sensors import (
    GroundTruthHighLevelSensor,
    OBJECT_NAMES
)
from rearrange.expert_models import (
    PretrainedExpertConvRNN, 
)
from typing import Sequence
from allenact.base_abstractions.sensor import Sensor
from torch import nn
import gym.spaces


class OnePhaseRGBExpertDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "80proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [*super(OnePhaseRGBExpertDaggerExperimentConfig, cls).sensors(), GroundTruthHighLevelSensor()]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBExpertDagger"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBExpertDaggerExperimentConfig, cls)._use_label_to_get_training_params()
        params["lr"] = 1e-4
        return params

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PretrainedExpertConvRNN(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            hidden_size=512,
            positional_features=3,
            voxel_features=len(OBJECT_NAMES),
            num_octaves=8,
            start_octave=-5)
