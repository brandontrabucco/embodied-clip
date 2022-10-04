from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
import torch


class OnePhaseRGBClipResNet50DaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBClipResNet50DaggerExperimentConfig, 
                       cls)._use_label_to_get_training_params()
        params["num_train_processes"] = 4 * torch.cuda.device_count()
        return params

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBClipResNet50Dagger"
