from rearrange.tasks import RearrangeTaskSampler
from rearrange.tasks import UnshuffleTask
from rearrange.tasks import WalkthroughTask

from rearrange.environment import RearrangeTHOREnvironment
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.task import SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from ai2thor.platform import CloudRendering

from typing import Optional, Sequence, Dict, Union, Tuple, Any, cast, List
from collections import OrderedDict

from rearrange.sensors import RGBRearrangeSensor
from rearrange.sensors import UnshuffledRGBRearrangeSensor
from rearrange.sensors import DepthRearrangeSensor

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.sensors.vision_sensors import VisionSensor
from allenact_plugins.ithor_plugin.ithor_sensors \
    import RelativePositionChangeTHORSensor

import numpy as np
import torch
import clip

import os

from PIL import Image
from itertools import product
import torch.nn.functional as functional
import torch.distributed as distributed


class ExperimentConfig(RearrangeBaseExperimentConfig):
    """Create a training session using the AI2-THOR Rearrangement task,
    including additional map_depth and semantic segmentation observations
    and expose a task sampling function.

    """

    # interval between successive WalkthroughTasks every next_task call
    TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH: int = 1

    # these sensors define the observation space of the agent
    # the relative pose sensor returns the pose of the agent in the world
    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
            use_resnet_normalization=False
        ),
        DepthRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE
        ),
        RelativePositionChangeTHORSensor()
    ]

    @classmethod
    def make_sampler_fn(cls, stage: str, force_cache_reset: bool,
                        allowed_scenes: Optional[Sequence[str]], seed: int,
                        epochs: Union[str, float, int],
                        scene_to_allowed_rearrange_inds:
                        Optional[Dict[str, Sequence[int]]] = None,
                        x_display: Optional[str] = None,
                        sensors: Optional[Sequence[Sensor]] = None,
                        only_one_unshuffle_per_walkthrough: bool = False,
                        thor_controller_kwargs: Optional[Dict] = None,
                        **kwargs) -> RearrangeTaskSampler:
        """Helper function that creates an object for sampling AI2-THOR 
        Rearrange tasks in walkthrough and unshuffle phases, where additional 
        semantic segmentation and map_depth observations are provided.

        Arguments:

        device: str
            specifies the device used by torch during the color lookup
            operation, which can be accelerated when set to a cuda device.

        Returns:

        sampler: RearrangeTaskSampler
            an instance of RearrangeTaskSampler that implements next_task()
            for generating walkthrough and unshuffle tasks successively.

        """

        # carrying this check over from the example, not sure if required
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        # add a semantic segmentation observation sensor to the list
        sensors = cls.SENSORS if sensors is None else sensors

        # allow default controller arguments to be overridden
        controller_kwargs = dict(**cls.THOR_CONTROLLER_KWARGS)
        if thor_controller_kwargs is not None:
            controller_kwargs.update(thor_controller_kwargs)

        # create a task sampler and carry over settings from the example
        # and ensure the environment will generate a semantic segmentation
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=True,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "platform": CloudRendering,
                    "renderDepthImage": True,
                    "renderSemanticSegmentation": True,
                    **controller_kwargs,
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            unshuffle_runs_per_walkthrough=
            cls.TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH
            if (not only_one_unshuffle_per_walkthrough) and stage == "train"
            else None,
            epochs=epochs, **kwargs)


if __name__ == "__main__":

    task_sampler_args = ExperimentConfig.stagewise_task_sampler_args(
        stage="train", devices=[0], process_ind=0, total_processes=1)

    task_sampler = ExperimentConfig.make_sampler_fn(
        **task_sampler_args, force_cache_reset=False, epochs=1)

    object_vocab = set()

    for task_id in range(task_sampler.length // 2):

        # skip the initial walkthrough phase of each training task
        task = task_sampler.next_task()
        task.step(action=task.action_names().index('done'))

        # set the unshuffle phase to the done state for scene evaluation
        task = task_sampler.next_task()
        task.step(action=task.action_names().index('done'))
        
        # add object types
        for obj in task.env.controller.last_event.metadata['objects']:
            object_vocab.add(obj['objectType'])

        print()
        print()
        print(task_id)
        print()
        print("object_vocab", object_vocab)