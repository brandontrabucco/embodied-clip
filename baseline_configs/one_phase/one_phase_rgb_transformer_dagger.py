from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.baseline_models import (
    ResNetRearrangeActorCriticTransformer,
)
from torch import nn
import gym.spaces


class OnePhaseRGBTransformerDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBTransformerDagger_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def _use_label_to_get_training_params(cls):
        
        lr = 1e-4
        num_steps = 64
        num_mini_batch = 1
        update_repeats = 3
        use_lr_decay = False
        num_train_processes = 40
        dagger_steps = min(int(1e6), cls.TRAINING_STEPS // 10)
        bc_tf1_steps = min(int(1e5), cls.TRAINING_STEPS // 10)

        return dict(
            lr=lr,
            num_steps=num_steps,
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            use_lr_decay=use_lr_decay,
            num_train_processes=num_train_processes,
            dagger_steps=dagger_steps,
            bc_tf1_steps=bc_tf1_steps,
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResNetRearrangeActorCriticTransformer(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID)
