from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig, 
    StepwiseLinearDecay
)
from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from rearrange.sensors import (
    ExpertRaysSensor,
    ExpertObjectsSensor,
    IntermediateVoxelSensor
)
from rearrange.hierarchical_models import (
    HierarchicalConvRNN,
    PretrainedHierarchicalConvRNN,
)
from rearrange.tasks import (
    UnshuffleTask,
    WalkthroughTask,
    RearrangeTaskSampler
)
from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
    Callable,
)
try:
    from allenact.embodiedai.sensors.vision_sensors import (
        DepthSensor,
        IMAGENET_RGB_MEANS,
        IMAGENET_RGB_STDS,
    )
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")
from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor, SensorSuite
from torch import nn
import gym.spaces

from typing import Tuple, Sequence, Optional, Dict, Any

import torch

from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor
from allenact.utils.experiment_utils import PipelineStage
from allenact.utils.misc_utils import all_unique
from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig


class WrappedUnshuffleTask(UnshuffleTask):

    MAX_VOXELS = 2

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            action=super(WrappedUnshuffleTask, self).action_space,
            attention=gym.spaces.Discrete(self.MAX_VOXELS))

    def step(self, action: Dict[str, int]):
        return super(WrappedUnshuffleTask, self).step(action=action["action"])

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if kwargs["expert_sensor_group_name"] == "action":
            action, success = super(WrappedUnshuffleTask, self).query_expert(**kwargs)
            self._cached_action_success = success
        else:
            assert hasattr(self, "_cached_action_success")
            action = 0 if self.env.held_object is not None else self.MAX_VOXELS // 2
            success = self._cached_action_success
            del self._cached_action_success
        return action, success


class WrappedWalkthroughTask(WalkthroughTask):

    MAX_VOXELS = 2

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            action=super(WrappedWalkthroughTask, self).action_space,
            attention=gym.spaces.Discrete(self.MAX_VOXELS))

    def step(self, action: Dict[str, int]):
        return super(WrappedWalkthroughTask, self).step(action=action["action"])

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if kwargs["expert_sensor_group_name"] == "action":
            action, success = super(WrappedWalkthroughTask, self).query_expert(**kwargs)
            self._cached_action_success = success
        else:
            assert hasattr(self, "_cached_action_success")
            action = 0 if self.env.held_object is not None else self.MAX_VOXELS // 2
            success = self._cached_action_success
            del self._cached_action_success
        return action, success


class WrappedRearrangeTaskSampler(RearrangeTaskSampler):

    def next_task(
        self, forced_task_spec: Optional = None, **kwargs
    ) -> Optional[WrappedUnshuffleTask]:
        """Return a fresh WrappedUnshuffleTask setup."""

        walkthrough_finished_and_should_run_unshuffle = (
            forced_task_spec is None
            and self.run_unshuffle_phase
            and self.run_walkthrough_phase
            and (
                self.was_in_exploration_phase
                or self.cur_unshuffle_runs_count < self.unshuffle_runs_per_walkthrough
            )
        )

        if (
            self.last_sampled_task is None
            or not walkthrough_finished_and_should_run_unshuffle
        ):
            self.cur_unshuffle_runs_count = 0

            try:
                if forced_task_spec is None:
                    task_spec: RearrangeTaskSpec = next(self.task_spec_iterator)
                else:
                    task_spec = forced_task_spec
            except StopIteration:
                self._last_sampled_task = None
                return self._last_sampled_task

            runtime_sample = task_spec.runtime_sample

            try:
                if self.run_unshuffle_phase:
                    self.unshuffle_env.reset(
                        task_spec=task_spec,
                        force_axis_aligned_start=self.force_axis_aligned_start,
                    )
                    self.unshuffle_env.shuffle()

                    if runtime_sample:
                        unshuffle_task_spec = self.unshuffle_env.current_task_spec
                        starting_objects = unshuffle_task_spec.runtime_data[
                            "starting_objects"
                        ]
                        openable_data = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "objectId": o["objectId"],
                                "start_openness": o["openness"],
                                "target_openness": o["openness"],
                            }
                            for o in starting_objects
                            if o["isOpen"] and not o["pickupable"]
                        ]
                        starting_poses = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "position": o["position"],
                                "rotation": o["rotation"],
                            }
                            for o in starting_objects
                            if o["pickupable"]
                        ]
                        task_spec = RearrangeTaskSpec(
                            scene=unshuffle_task_spec.scene,
                            agent_position=task_spec.agent_position,
                            agent_rotation=task_spec.agent_rotation,
                            openable_data=openable_data,
                            starting_poses=starting_poses,
                            target_poses=starting_poses,
                        )

                self.walkthrough_env.reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )

                if self.run_walkthrough_phase:
                    self.was_in_exploration_phase = True
                    self._last_sampled_task = WrappedWalkthroughTask(
                        sensors=self.sensors,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["walkthrough"],
                        discrete_actions=self.discrete_actions,
                        disable_metrics=self.run_unshuffle_phase,
                    )
                    self._last_sampled_walkthrough_task = self._last_sampled_task
                else:
                    self.cur_unshuffle_runs_count += 1
                    self._last_sampled_task = WrappedUnshuffleTask(
                        sensors=self.sensors,
                        unshuffle_env=self.unshuffle_env,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["unshuffle"],
                        discrete_actions=self.discrete_actions,
                        require_done_action=self.require_done_action,
                        task_spec_in_metrics=self.task_spec_in_metrics,
                    )
            except Exception as e:
                if runtime_sample:
                    get_logger().error(
                        "Encountered exception while sampling a next task."
                        " As this next task was a 'runtime sample' we are"
                        " simply returning the next task."
                    )
                    get_logger().error(traceback.format_exc())
                    return self.next_task()
                else:
                    raise e
        else:
            self.cur_unshuffle_runs_count += 1
            self.was_in_exploration_phase = False

            walkthrough_task = cast(
                WrappedWalkthroughTask, self._last_sampled_walkthrough_task
            )

            if self.cur_unshuffle_runs_count != 1:
                self.unshuffle_env.reset(
                    task_spec=self.unshuffle_env.current_task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )
                self.unshuffle_env.shuffle()

            self._last_sampled_task = WrappedUnshuffleTask(
                sensors=self.sensors,
                unshuffle_env=self.unshuffle_env,
                walkthrough_env=self.walkthrough_env,
                max_steps=self.max_steps["unshuffle"],
                discrete_actions=self.discrete_actions,
                require_done_action=self.require_done_action,
                locations_visited_in_walkthrough=np.array(
                    tuple(walkthrough_task.visited_positions_xzrsh)
                ),
                object_names_seen_in_walkthrough=copy.copy(
                    walkthrough_task.seen_pickupable_objects
                    | walkthrough_task.seen_openable_objects
                ),
                metrics_from_walkthrough=walkthrough_task.metrics(force_return=True),
                task_spec_in_metrics=self.task_spec_in_metrics,
            )

        return self._last_sampled_task


class OnePhaseRGBVoxelsDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")
    IL_PIPELINE_TYPE = "40proc"

    MAX_VOXELS = 2

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        num_actions = len(OnePhaseRGBVoxelsDaggerExperimentConfig.actions())
        return [
            *super(OnePhaseRGBVoxelsDaggerExperimentConfig, cls).sensors()[:2],
            IntermediateVoxelSensor(),
            ExpertActionSensor(
                action_space=gym.spaces.Dict(
                    action=gym.spaces.Discrete(num_actions), 
                    attention=gym.spaces.Discrete(cls.MAX_VOXELS)
                )
            )
        ]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBVoxelsDagger"

    @classmethod
    def _use_label_to_get_training_params(cls, **kwargs):
        params = super(OnePhaseRGBVoxelsDaggerExperimentConfig, 
                       cls)._use_label_to_get_training_params()
        params["lr"] = 1e-4
        params["num_train_processes"] = 4
        return params

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(imitation_loss=Imitation(cls.sensors()[3])),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **params
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        num_actions = len(OnePhaseRGBVoxelsDaggerExperimentConfig.actions())
        return PretrainedHierarchicalConvRNN(
            action_space=gym.spaces.Dict(
                action=gym.spaces.Discrete(num_actions), 
                attention=gym.spaces.Discrete(cls.MAX_VOXELS)
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            hidden_size=512,
            positional_features=3,
            voxel_features=256,
            num_octaves=8,
            start_octave=-5)

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
    ) -> WrappedRearrangeTaskSampler:
        """Return a WrappedRearrangeTaskSampler."""
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return WrappedRearrangeTaskSampler.from_fixed_dataset(
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
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in sensors
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

