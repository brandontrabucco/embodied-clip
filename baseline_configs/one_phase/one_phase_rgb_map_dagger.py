from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.mapping_models import (
    MappingInputsSensor,
    VoxelMapActorCriticRNN
)
from typing import Sequence, Optional, Dict
from allenact.base_abstractions.sensor import (
    Sensor, 
    SensorSuite, 
    ExpertActionSensor
)
from rearrange.tasks import RearrangeTaskSampler

import torch
import torch.nn as nn
import gym.spaces


class OnePhaseRGBMapDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = None
    IL_PIPELINE_TYPE = "80proc"

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return [ExpertActionSensor(len(cls.actions())), 
                MappingInputsSensor()]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBMapDagger"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBMapDaggerExperimentConfig, 
                       cls)._use_label_to_get_training_params()

        params['num_steps'] = 32
        params["lr"] = 1e-4

        return params

    @classmethod
    def create_model(cls, **kwargs):
        return VoxelMapActorCriticRNN(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=SensorSuite(cls.sensors()).observation_spaces,
            maximum_size_x=240,
            maximum_size_y=240,
            maximum_size_z=28,
            voxel_size=.1,
            fov=90.0,
            image_size=224,
            egocentric_map_size=16
        )

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    "renderDepthImage": True,
                    "renderSemanticSegmentation": True,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            **kwargs,
        )
