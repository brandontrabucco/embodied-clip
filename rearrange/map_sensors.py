from typing import Any, Optional, Union

import gym.spaces
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from collections import OrderedDict, defaultdict

try:
    from allenact.embodiedai.sensors.vision_sensors import RGBSensor
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor

from rearrange.constants import STEP_SIZE
from rearrange.environment import RearrangeTHOREnvironment
from rearrange.tasks import (
    UnshuffleTask,
    WalkthroughTask,
    AbstractRearrangeTask,
)

import einops
import glob
import os
import math


OBJECT_NAMES = ['FreeSpace', 'OccupiedSpace', 'ToiletPaper', 'StoveKnob', 'SinkBasin', 'Bed', 'ScrubBrush', 'Sofa', 'HandTowelHolder', 'Egg', 'AlarmClock', 'Knife', 'Vase', 'Pot', 'Pen', 'WateringCan', 'WineBottle', 'SideTable', 'Curtains', 'SprayBottle', 'Sink', 'Dresser', 'CreditCard', 'ShelvingUnit', 'Cup', 'SoapBottle', 'Microwave', 'Ladle', 'RoomDecor', 'Fridge', 'ToiletPaperHanger', 'Television', 'StoveBurner', 'PepperShaker', 'Candle', 'PaperTowelRoll', 'FloorLamp', 'Desk', 'Bathtub', 'CellPhone', 'TVStand', 'LightSwitch', 'Plunger', 'DiningTable', 'Window', 'Mug', 'TennisRacket', 'Cabinet', 'Stool', 'Spoon', 'Drawer', 'Floor', 'TowelHolder', 'Watch', 'BaseballBat', 'DeskLamp', 'HousePlant', 'Painting', 'Spatula', 'Fork', 'Boots', 'ButterKnife', 'Dumbbell', 'CounterTop', 'ShowerGlass', 'GarbageCan', 'BathtubBasin', 'SaltShaker', 'Shelf', 'DishSponge', 'Poster', 'Chair', 'Bowl', 'Desktop', 'TableTopDecor', 'Bottle', 'TissueBox', 'Pan', 'DogBed', 'ShowerDoor', 'Plate', 'Newspaper', 'Footstool', 'Laptop', 'Book', 'Blinds', 'TeddyBear', 'Faucet', 'Ottoman', 'GarbageBag', 'Safe', 'Pencil', 'ShowerHead', 'Mirror', 'CoffeeTable', 'LaundryHamper', 'CoffeeMachine', 'ShowerCurtain', 'BasketBall', 'Statue', 'Toaster', 'SoapBar', 'Toilet', 'CD', 'Box', 'ArmChair', 'Kettle', 'RemoteControl']


class SemanticMapSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    DATA_DIR = "/home/ubuntu/embodied-clip/semantic_maps"

    WALKTHROUGH_VOXEL_FEATURES_LABEL = "voxel_features_w"
    WALKTHROUGH_VOXEL_POSITIONS_LABEL = "voxel_positions_w"

    UNSHUFFLE_VOXEL_FEATURES_LABEL = "voxel_features_u"
    UNSHUFFLE_VOXEL_POSITIONS_LABEL = "voxel_positions_u"

    def __init__(self, uuid="map", use_egocentric_sensor=True, downsample=5,
                 voxels_per_map=1, voxel_feature_size=len(OBJECT_NAMES), modifier=""):

        self.use_egocentric_sensor = use_egocentric_sensor
        self.downsample = downsample

        self.max_voxels = voxels_per_map
        self.voxel_feature_size = voxel_feature_size

        self.modifier = modifier

        observation_space = gym.spaces.Dict([

            (self.WALKTHROUGH_VOXEL_FEATURES_LABEL, 
                gym.spaces.Box(np.full([self.max_voxels * 2, self.voxel_feature_size], -20.0), 
                               np.full([self.max_voxels * 2, self.voxel_feature_size],  20.0))),

            (self.WALKTHROUGH_VOXEL_POSITIONS_LABEL, 
                gym.spaces.Box(np.full([self.max_voxels * 2, 3], -20.0), 
                               np.full([self.max_voxels * 2, 3],  20.0))),

            (self.UNSHUFFLE_VOXEL_FEATURES_LABEL, 
                gym.spaces.Box(np.full([self.max_voxels * 2, self.voxel_feature_size], -20.0), 
                               np.full([self.max_voxels * 2, self.voxel_feature_size],  20.0))),

            (self.UNSHUFFLE_VOXEL_POSITIONS_LABEL, 
                gym.spaces.Box(np.full([self.max_voxels * 2, 3], -20.0), 
                               np.full([self.max_voxels * 2, 3],  20.0))),

        ])

        self.cache_name = None
        self.cached_object_name = None

        self.cached_coords_w = None
        self.cached_feature_map_w = None

        self.cached_coords_u = None
        self.cached_feature_map_u = None
        
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )
        
        scene = task.env.scene
        index = task.env.current_task_spec.metrics.get("index")
        stage = task.env.current_task_spec.stage

        cache_name = f"{self.DATA_DIR}/thor-{scene}-{index}-{stage}"

        if self.cache_name != cache_name:
            self.cache_name = cache_name

            cached_coords_w = np.load(
                f"{cache_name}-walkthrough-coords{self.modifier}.npy")
            cached_coords_u = np.load(
                f"{cache_name}-unshuffle-coords{self.modifier}.npy")

            cached_feature_map_w = np.load(
                f"{cache_name}-walkthrough-feature_map{self.modifier}.npy")
            cached_feature_map_u = np.load(
                f"{cache_name}-unshuffle-feature_map{self.modifier}.npy")

            original_shape_w = cached_feature_map_w.shape[:2]
            shape_w = [int(math.ceil(x / self.downsample)) * 
                       self.downsample for x in original_shape_w]

            original_shape_u = cached_feature_map_u.shape[:2]
            shape_u = [int(math.ceil(x / self.downsample)) * 
                       self.downsample for x in original_shape_u]

            cached_feature_map_w = np.pad(
                cached_feature_map_w,
                [[0, x1 - x0] for x0, x1 in 
                 zip(original_shape_w, shape_w)] + [[0, 0]])

            cached_feature_map_u = np.pad(
                cached_feature_map_u,
                [[0, x1 - x0] for x0, x1 in 
                 zip(original_shape_u, shape_u)] + [[0, 0]])

            cached_feature_map_w = einops.reduce(
                cached_feature_map_w, "(h n) (w m) c -> h w c", "sum",
                n=self.downsample, m=self.downsample).clip(max=1, min=0)

            cached_feature_map_u = einops.reduce(
                cached_feature_map_u, "(h n) (w m) c -> h w c", "sum",
                n=self.downsample, m=self.downsample).clip(max=1, min=0)

            cached_coords_w = np.pad(
                cached_coords_w,
                [[0, x1 - x0] for x0, x1 in zip(
                    original_shape_w, shape_w)] + [[0, 0]], mode="edge")

            cached_coords_u = np.pad(
                cached_coords_u,
                [[0, x1 - x0] for x0, x1 in zip(
                    original_shape_u, shape_u)] + [[0, 0]], mode="edge")

            cached_coords_w = einops.reduce(
                cached_coords_w, "(h n) (w m) c -> h w c", "mean",
                n=self.downsample, m=self.downsample).clip(max=1, min=0)

            cached_coords_u = einops.reduce(
                cached_coords_u, "(h n) (w m) c -> h w c", "mean",
                n=self.downsample, m=self.downsample).clip(max=1, min=0)

            self.cached_coords_w = cached_coords_w.reshape([-1, 2])
            self.cached_feature_map_w = cached_feature_map_w\
                .reshape([-1, self.voxel_feature_size])

            self.cached_coords_u = cached_coords_u.reshape([-1, 2])
            self.cached_feature_map_u = cached_feature_map_u\
                .reshape([-1, self.voxel_feature_size])

        if task.greedy_expert is None:
            task.query_expert(expert_sensor_group_name="attention")

        if task.greedy_expert._last_to_interact_object_pose is not None:
            self.cached_object_name = task.greedy_expert._last_to_interact_object_pose["name"]

        elif env.held_object is not None:
            self.cached_object_name = env.held_object["name"]

        object_pose_w = env.obj_name_to_walkthrough_start_pose[
            self.cached_object_name]["position"]
        object_pose_u = env.obj_name_to_unshuffle_start_pose [
            self.cached_object_name]["position"]

        object_pose_w = np.array([
            object_pose_w["x"], 
            object_pose_w["z"]
        ])

        object_pose_u = np.array([
            object_pose_u["x"], 
            object_pose_u["z"]
        ])

        voxal_idx_w = np.linalg.norm(
            self.cached_coords_w - 
            object_pose_w[np.newaxis, :], 
            axis=1).argsort()

        voxal_idx_u = np.linalg.norm(
            self.cached_coords_u - 
            object_pose_u[np.newaxis, :], 
            axis=1).argsort()

        voxal_idx_w = np.concatenate([
            voxal_idx_w[:1], np.random.choice(
                voxal_idx_w[1:], size=self.max_voxels - 1, replace=False)
        ], axis=0)

        voxal_idx_u = np.concatenate([
            voxal_idx_u[:1], np.random.choice(
                voxal_idx_u[1:], size=self.max_voxels - 1, replace=False)
        ], axis=0)

        voxel_feature_w = self.cached_feature_map_w[voxal_idx_w]
        voxel_feature_u = self.cached_feature_map_u[voxal_idx_u]

        voxel_position_w = self.cached_coords_w[voxal_idx_w]
        voxel_position_u = self.cached_coords_u[voxal_idx_u]
            
        location = task.env.get_agent_location()

        agent_current_pose = np.array([
            location["x"], 
            location["z"]
        ])

        voxel_position_w -= agent_current_pose[np.newaxis, :]
        voxel_position_u -= agent_current_pose[np.newaxis, :]

        return OrderedDict([  # return rays and discrete labels

            (self.WALKTHROUGH_VOXEL_FEATURES_LABEL, 
                voxel_feature_w.astype(np.float32)),
            (self.UNSHUFFLE_VOXEL_FEATURES_LABEL, 
                voxel_feature_u.astype(np.float32)),    

            (self.WALKTHROUGH_VOXEL_POSITIONS_LABEL, 
                voxel_position_w.astype(np.float32)),
            (self.UNSHUFFLE_VOXEL_POSITIONS_LABEL, 
                voxel_position_u.astype(np.float32)),
        
        ])


class GroundTruthHighLevelSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    GOAL_FEATURE_LABEL = "goal_feature"
    GOAL_POSITION_LABEL = "goal_position"

    def __init__(self, uuid="map", use_egocentric_sensor=True):

        self.use_egocentric_sensor = use_egocentric_sensor

        observation_space = gym.spaces.Dict([

            (self.GOAL_FEATURE_LABEL, 
                gym.spaces.Box(np.full([len(OBJECT_NAMES)], -20.0), 
                               np.full([len(OBJECT_NAMES)],  20.0))),

            (self.GOAL_POSITION_LABEL, 
                gym.spaces.Box(np.full([3], -20.0), 
                               np.full([3],  20.0))),

        ])
        
        super().__init__(**prepare_locals_for_super(locals()))

        self.cached_object_name = None
        self.carrying = False

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )
        
        if task.greedy_expert is None:
            task.query_expert(expert_sensor_group_name="attention")

        if task.greedy_expert._last_to_interact_object_pose is not None:
            self.cached_object_name = task.greedy_expert._last_to_interact_object_pose["name"]
            self.carrying = False

        elif env.held_object is not None:
            self.cached_object_name = env.held_object["name"]
            self.carrying = True

        object_pose_w = env.obj_name_to_walkthrough_start_pose[
            self.cached_object_name]["position"]
        object_pose_u = env.obj_name_to_unshuffle_start_pose [
            self.cached_object_name]["position"]

        object_pose_w = np.array([
            object_pose_w["x"], 
            object_pose_w["z"], 
            object_pose_w["y"]
        ])

        object_pose_u = np.array([
            object_pose_u["x"], 
            object_pose_u["z"], 
            object_pose_u["y"]
        ])
            
        location = task.env.get_agent_location()

        agent_current_pose = np.array([
            location["x"], 
            location["z"], 
            location["y"]
        ])

        object_pose_w -= agent_current_pose
        object_pose_u -= agent_current_pose

        position_feature = object_pose_w if self.carrying else object_pose_u

        semantic_feature = np.zeros([len(OBJECT_NAMES)])
        if self.cached_object_name.split("_")[0] in OBJECT_NAMES:
            semantic_feature[OBJECT_NAMES.index(self.cached_object_name.split("_")[0])] = 1

        return OrderedDict([

            (self.GOAL_FEATURE_LABEL, 
                semantic_feature.astype(np.float32)),
            (self.GOAL_POSITION_LABEL, 
                position_feature.astype(np.float32)),
        
        ])