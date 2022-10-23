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


PICKABLE_TO_COLOR = OrderedDict([
    ('Candle', (233, 102, 178)),
    ('SoapBottle', (168, 222, 137)),
    ('ToiletPaper', (162, 204, 152)),
    ('SoapBar', (43, 97, 155)),
    ('SprayBottle', (89, 126, 121)),
    ('TissueBox', (98, 43, 249)),
    ('DishSponge', (166, 58, 136)),
    ('PaperTowelRoll', (144, 173, 28)),
    ('Book', (43, 31, 148)),
    ('CreditCard', (56, 235, 12)),
    ('Dumbbell', (45, 57, 144)),
    ('Pen', (239, 130, 152)),
    ('Pencil', (177, 226, 23)),
    ('CellPhone', (227, 98, 136)),
    ('Laptop', (20, 107, 222)),
    ('CD', (65, 112, 172)),
    ('AlarmClock', (184, 20, 170)),
    ('Statue', (243, 75, 41)),
    ('Mug', (8, 94, 186)),
    ('Bowl', (209, 182, 193)),
    ('TableTopDecor', (126, 204, 158)),
    ('Box', (60, 252, 230)),
    ('RemoteControl', (187, 19, 208)),
    ('Vase', (83, 152, 69)),
    ('Watch', (242, 6, 88)),
    ('Newspaper', (19, 196, 2)),
    ('Plate', (188, 154, 128)),
    ('WateringCan', (147, 67, 249)),
    ('Fork', (54, 200, 25)),
    ('PepperShaker', (5, 204, 214)),
    ('Spoon', (235, 57, 90)),
    ('ButterKnife', (135, 147, 55)),
    ('Pot', (132, 237, 87)),
    ('SaltShaker', (36, 222, 26)),
    ('Cup', (35, 71, 130)),
    ('Spatula', (30, 98, 242)),
    ('WineBottle', (53, 130, 252)),
    ('Knife', (211, 157, 122)),
    ('Pan', (246, 212, 161)),
    ('Ladle', (174, 98, 216)),
    ('Egg', (240, 75, 163)),
    ('Kettle', (7, 83, 48)),
    ('Bottle', (64, 80, 115))])


OPENABLE_TO_COLOR = OrderedDict([
    ('Drawer', (155, 30, 210)),
    ('Toilet', (21, 27, 163)),
    ('ShowerCurtain', (60, 12, 39)),
    ('ShowerDoor', (36, 253, 61)),
    ('Cabinet', (210, 149, 89)),
    ('Blinds', (214, 223, 197)),
    ('LaundryHamper', (35, 109, 26)),
    ('Safe', (198, 238, 160)),
    ('Microwave', (54, 96, 202)),
    ('Fridge', (91, 156, 207))])


CLASS_TO_COLOR = OrderedDict(
    [("OccupiedSpace", (243, 246, 208))]
    + list(PICKABLE_TO_COLOR.items())
    + list(OPENABLE_TO_COLOR.items()))


CLASS_TO_ID = OrderedDict([(key, idx) for idx, key in enumerate(CLASS_TO_COLOR.keys())])


import glob
import os


class ExpertVoxelSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):
    
    VOXEL_FEATURE_SIZE = 256

    DATA_DIR = "/home/ubuntu/embodied-clip/maps"

    VOXEL_FEATURES_LABEL = "voxel_features"
    VOXEL_POSITIONS_LABEL = "voxel_positions"

    def __init__(self, uuid="map", use_egocentric_sensor=True):

        self.use_egocentric_sensor = use_egocentric_sensor

        observation_space = gym.spaces.Dict([

            (self.VOXEL_FEATURES_LABEL, 
                gym.spaces.Box(np.full([self.VOXEL_FEATURE_SIZE], -20.0), 
                               np.full([self.VOXEL_FEATURE_SIZE],  20.0))),

            (self.VOXEL_POSITIONS_LABEL, 
                gym.spaces.Box(np.full([3], -20.0), 
                               np.full([3],  20.0))),

        ])

        self.cache_name = None

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
                f"{cache_name}-walkthrough-coords.npy")
            cached_coords_u = np.load(
                f"{cache_name}-unshuffle-coords.npy")

            cached_feature_map_w = np.load(
                f"{cache_name}-walkthrough-feature_map.npy")
            cached_feature_map_u = np.load(
                f"{cache_name}-unshuffle-feature_map.npy")

            cached_hits_per_voxel_w = np.load(
                f"{cache_name}-walkthrough-hits_per_voxel.npy")
            cached_hits_per_voxel_u = np.load(
                f"{cache_name}-unshuffle-hits_per_voxel.npy")

            indices_w = np.nonzero(cached_hits_per_voxel_w[..., 0])
            indices_u = np.nonzero(cached_hits_per_voxel_u[..., 0])

            self.cached_coords_w = cached_coords_w[indices_w]
            self.cached_feature_map_w = cached_feature_map_w[indices_w]

            self.cached_coords_u = cached_coords_u[indices_u]
            self.cached_feature_map_u = cached_feature_map_u[indices_u]
            
        location = task.env.get_agent_location()
        crouch_height_offset = 0.675 if location["standing"] else 0.0

        agent_current_pose = np.array([
            location["x"], 
            location["z"], 
            location["y"] + 
            crouch_height_offset
        ])

        object_current_pose = agent_current_pose
        voxel_feature = np.zeros([self.VOXEL_FEATURE_SIZE])

        if env.held_object is not None:

            object_current_pose = env.obj_name_to_walkthrough_start_pose[
                env.held_object["name"]]["position"]

            object_current_pose = np.array([
                object_current_pose["x"], 
                object_current_pose["z"], 
                object_current_pose["y"]
            ])

            voxal_idx = np.linalg.norm(
                self.cached_coords_w - 
                object_current_pose[np.newaxis, :], 
                axis=1).argmin()

            voxel_feature = self.cached_feature_map_w[voxal_idx]

        elif task.greedy_expert._last_to_interact_object_pose is not None:

            object_current_pose = task.greedy_expert\
                ._last_to_interact_object_pose["position"]

            object_current_pose = np.array([
                object_current_pose["x"], 
                object_current_pose["z"], 
                object_current_pose["y"]
            ])

            voxal_idx = np.linalg.norm(
                self.cached_coords_u - 
                object_current_pose[np.newaxis, :], 
                axis=1).argmin()

            voxel_feature = self.cached_feature_map_u[voxal_idx]

        return OrderedDict([  # return rays and discrete labels

            (self.VOXEL_FEATURES_LABEL, 
                voxel_feature.astype(np.float32)),

            (self.VOXEL_POSITIONS_LABEL, (
                object_current_pose - 
                agent_current_pose).astype(np.float32)),
        
        ])


class IntermediateVoxelSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    DATA_DIR = "/home/ubuntu/embodied-clip/maps"

    WALKTHROUGH_VOXEL_FEATURES_LABEL = "voxel_features_w"
    WALKTHROUGH_VOXEL_POSITIONS_LABEL = "voxel_positions_w"

    UNSHUFFLE_VOXEL_FEATURES_LABEL = "voxel_features_u"
    UNSHUFFLE_VOXEL_POSITIONS_LABEL = "voxel_positions_u"

    def __init__(self, uuid="map", use_egocentric_sensor=True, 
                 voxels_per_map=1, voxel_feature_size=512, modifier=""):

        self.use_egocentric_sensor = use_egocentric_sensor
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

            self.cached_coords_w = cached_coords_w
            self.cached_feature_map_w = cached_feature_map_w

            self.cached_coords_u = cached_coords_u
            self.cached_feature_map_u = cached_feature_map_u

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
            object_pose_w["z"], 
            object_pose_w["y"]
        ])

        object_pose_u = np.array([
            object_pose_u["x"], 
            object_pose_u["z"], 
            object_pose_u["y"]
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
        crouch_height_offset = 0.675 if location["standing"] else 0.0

        agent_current_pose = np.array([
            location["x"], 
            location["z"], 
            location["y"] + 
            crouch_height_offset
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


class ExpertObjectsSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    MAX_OBJECTS_SHUFFLED = 8

    WALKTHROUGH_OBJECTS_LABEL = "walkthrough_rays"
    WALKTHROUGH_CLASSES_LABEL = "walkthrough_classes"
    WALKTHROUGH_INSTANCES_LABEL = "walkthrough_instances"

    UNSHUFFLE_OBJECTS_LABEL = "unshuffle_rays"
    UNSHUFFLE_CLASSES_LABEL = "unshuffle_classes"
    UNSHUFFLE_INSTANCES_LABEL = "unshuffle_instances"

    def __init__(self, uuid="nerf", use_egocentric_sensor=True):

        self.use_egocentric_sensor = use_egocentric_sensor

        observation_space = gym.spaces.Dict([

            (self.WALKTHROUGH_OBJECTS_LABEL, 
                gym.spaces.Box(np.full([self.MAX_OBJECTS_SHUFFLED, 3], -20.0), 
                               np.full([self.MAX_OBJECTS_SHUFFLED, 3],  20.0))),
            (self.UNSHUFFLE_OBJECTS_LABEL, 
                gym.spaces.Box(np.full([self.MAX_OBJECTS_SHUFFLED, 3], -20.0), 
                               np.full([self.MAX_OBJECTS_SHUFFLED, 3],  20.0))),

            (self.WALKTHROUGH_CLASSES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED], len(CLASS_TO_ID)))),
            (self.UNSHUFFLE_CLASSES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED], len(CLASS_TO_ID)))),
                                            
            (self.WALKTHROUGH_INSTANCES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED], len(CLASS_TO_ID)))),
            (self.UNSHUFFLE_INSTANCES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED], len(CLASS_TO_ID)))),

        ])
        
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )

        location = task.env.get_agent_location()
        crouch_height_offset = 0.675 if location["standing"] else 0.0

        agent_current_pose = np.array([location["x"], location["y"] + 
                                       crouch_height_offset, location["z"]])

        reachable_positions = task.env.controller.step(
            action="GetReachablePositions").metadata["actionReturn"]

        unshuffle_objects = []
        walkthrough_objects = []

        unshuffle_classes = []
        walkthrough_classes = []

        unshuffle_instances = []
        walkthrough_instances = []

        class_counts = defaultdict(int)

        unshuffle_poses, walkthrough_poses, _ = task.env.poses
        for object_u, object_w in zip(unshuffle_poses, walkthrough_poses):

            if task.env.are_poses_equal(object_u, object_w):
                continue  # only compute positions for misplaced objects 

            pose_u = np.array([object_u["position"]["x"], 
                               object_u["position"]["y"], 
                               object_u["position"]["z"]])

            if self.use_egocentric_sensor:
                pose_u = pose_u - agent_current_pose

            pose_w = np.array([object_w["position"]["x"], 
                               object_w["position"]["y"], 
                               object_w["position"]["z"]])

            if self.use_egocentric_sensor:
                pose_w = pose_w - agent_current_pose

            unshuffle_objects.append(pose_u)
            walkthrough_objects.append(pose_w)

            unshuffle_classes.append(CLASS_TO_ID[object_u["type"]])
            walkthrough_classes.append(CLASS_TO_ID[object_w["type"]])

            unshuffle_instances.append(class_counts[object_u["type"]])
            walkthrough_instances.append(class_counts[object_w["type"]])

            class_counts[object_u["type"]] += 1
            
        unshuffle_objects = np.stack(unshuffle_objects, axis=0)
        walkthrough_objects = np.stack(walkthrough_objects, axis=0)

        unshuffle_classes = np.array(unshuffle_classes)
        walkthrough_classes = np.array(walkthrough_classes)

        unshuffle_instances = np.array(unshuffle_instances)
        walkthrough_instances = np.array(walkthrough_instances)

        add_padding = self.MAX_OBJECTS_SHUFFLED - unshuffle_classes.size

        return OrderedDict([  # return rays and discrete labels

            (self.WALKTHROUGH_OBJECTS_LABEL, np.concatenate([
                walkthrough_objects,
                np.full([add_padding, 3], 0.0)], axis=0)),
            (self.UNSHUFFLE_OBJECTS_LABEL, np.concatenate([
                unshuffle_objects, 
                np.full([add_padding, 3], 0.0)], axis=0)),

            (self.WALKTHROUGH_CLASSES_LABEL, np.concatenate([
                walkthrough_classes,
                np.full([add_padding], 0)], axis=0)),
            (self.UNSHUFFLE_CLASSES_LABEL, np.concatenate([
                unshuffle_classes, 
                np.full([add_padding], 0)], axis=0)),

            (self.WALKTHROUGH_INSTANCES_LABEL, np.concatenate([
                walkthrough_instances,
                np.full([add_padding], 0)], axis=0)),
            (self.UNSHUFFLE_INSTANCES_LABEL, np.concatenate([
                unshuffle_instances, 
                np.full([add_padding], 0)], axis=0))])


class ExpertRaysSensor(
    Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]
):

    NUM_PATCHES = 3
    MAX_OBJECTS_SHUFFLED = 8

    WALKTHROUGH_RAYS_LABEL = "walkthrough_rays"

    WALKTHROUGH_CLASSES_LABEL = "walkthrough_classes"
    WALKTHROUGH_INSTANCES_LABEL = "walkthrough_instances"

    UNSHUFFLE_RAYS_LABEL = "unshuffle_rays"

    UNSHUFFLE_CLASSES_LABEL = "unshuffle_classes"
    UNSHUFFLE_INSTANCES_LABEL = "unshuffle_instances"

    @staticmethod
    def distance_to_object(x, object_i):
        return np.sqrt((x["x"] - object_i["position"]["x"]) ** 2 + 
                       (x["z"] - object_i["position"]["z"]) ** 2)

    def __init__(self, uuid="nerf", use_egocentric_sensor=False):

        self.use_egocentric_sensor = use_egocentric_sensor

        observation_space = gym.spaces.Dict([

            (self.WALKTHROUGH_RAYS_LABEL, 
                gym.spaces.Box(np.full([self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES, 5], -20.0), 
                               np.full([self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES, 5],  20.0))),
            (self.UNSHUFFLE_RAYS_LABEL, 
                gym.spaces.Box(np.full([self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES, 5], -20.0), 
                               np.full([self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES, 5],  20.0))),

            (self.WALKTHROUGH_CLASSES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),
            (self.UNSHUFFLE_CLASSES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),
                                            
            (self.WALKTHROUGH_INSTANCES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),
            (self.UNSHUFFLE_INSTANCES_LABEL, 
                gym.spaces.MultiDiscrete(np.full([
                    self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),

        ])
        
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )

        location = task.env.get_agent_location()
        crouch_height_offset = 0.675 if location["standing"] else 0.0

        agent_current_pose = np.array([
            location["x"], location["y"] + crouch_height_offset, location["z"],
            location["horizon"] / 180.0 * np.pi, location["rotation"] / 180.0 * np.pi])

        reachable_positions = task.env.controller.step(
            action="GetReachablePositions").metadata["actionReturn"]

        unshuffle_rays = []
        walkthrough_rays = []

        unshuffle_class = []
        walkthrough_class = []

        unshuffle_instance = []
        walkthrough_instance = []

        class_counts = defaultdict(int)

        unshuffle_poses, walkthrough_poses, _ = task.env.poses
        for object_u, object_w in zip(unshuffle_poses, walkthrough_poses):

            if task.env.are_poses_equal(object_u, object_w):
                continue  # only compute rays for objects that are misplaced

            distances_u = [self.distance_to_object(x, object_u) 
                           for x in reachable_positions]

            positions_u = [reachable_positions[idx] for idx in 
                           np.argsort(distances_u)[:self.NUM_PATCHES]]

            distances_w = [self.distance_to_object(x, object_w) 
                           for x in reachable_positions]

            positions_w = [reachable_positions[idx] for idx in 
                           np.argsort(distances_w)[:self.NUM_PATCHES]]

            for pui, pwi in zip(positions_u, positions_w):

                position = pui
                position["y"] = 1.275

                end = np.array([object_u["position"]["x"], 
                                object_u["position"]["y"], 
                                object_u["position"]["z"]])

                start = np.array([position["x"], 
                                  position["y"], 
                                  position["z"]])

                look_at = end - start
                look_at = look_at / np.linalg.norm(look_at)

                rx = np.arccos(look_at[1]) - np.pi/2
                ry = np.arctan2(look_at[0], look_at[2])
                
                pose_u = np.array([position["x"], 
                                   position["y"], 
                                   position["z"], rx, ry])

                if self.use_egocentric_sensor:
                    pose_u = pose_u - agent_current_pose

                position = pwi
                position["y"] = 1.275

                end = np.array([object_w["position"]["x"], 
                                object_w["position"]["y"], 
                                object_w["position"]["z"]])

                start = np.array([position["x"], 
                                  position["y"], 
                                  position["z"]])

                look_at = end - start
                look_at = look_at / np.linalg.norm(look_at)

                rx = np.arccos(look_at[1]) - np.pi/2
                ry =  np.arctan2(look_at[0], look_at[2])
                
                pose_w = np.array([position["x"], 
                                   position["y"], 
                                   position["z"], rx, ry])

                if self.use_egocentric_sensor:
                    pose_w = pose_w - agent_current_pose

                    print(pose_w[3:])

                unshuffle_rays.append(pose_u)
                walkthrough_rays.append(pose_w)

                unshuffle_class.append(CLASS_TO_ID[object_u["type"]])
                walkthrough_class.append(CLASS_TO_ID[object_w["type"]])

                unshuffle_instance.append(class_counts[object_u["type"]])
                walkthrough_instance.append(class_counts[object_w["type"]])

            class_counts[object_u["type"]] += 1
            
        unshuffle_rays = np.stack(unshuffle_rays, axis=0)
        walkthrough_rays = np.stack(walkthrough_rays, axis=0)

        unshuffle_class = np.array(unshuffle_class)
        walkthrough_class = np.array(walkthrough_class)

        unshuffle_instance = np.array(unshuffle_instance)
        walkthrough_instance = np.array(walkthrough_instance)

        add_padding = (self.MAX_OBJECTS_SHUFFLED * 
                       self.NUM_PATCHES - unshuffle_class.size)

        return OrderedDict([  # return rays and discrete labels

            (self.WALKTHROUGH_RAYS_LABEL, np.concatenate([
                walkthrough_rays,
                np.full([add_padding, 5], 0.0)], axis=0)),
            (self.UNSHUFFLE_RAYS_LABEL, np.concatenate([
                unshuffle_rays, 
                np.full([add_padding, 5], 0.0)], axis=0)),

            (self.WALKTHROUGH_CLASSES_LABEL, np.concatenate([
                walkthrough_class,
                np.full([add_padding], 0)], axis=0)),
            (self.UNSHUFFLE_CLASSES_LABEL, np.concatenate([
                unshuffle_class, 
                np.full([add_padding], 0)], axis=0)),

            (self.WALKTHROUGH_INSTANCES_LABEL, np.concatenate([
                walkthrough_instance,
                np.full([add_padding], 0)], axis=0)),
            (self.UNSHUFFLE_INSTANCES_LABEL, np.concatenate([
                unshuffle_instance, 
                np.full([add_padding], 0)], axis=0))])

                
class RGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return task.walkthrough_env.last_event.frame.copy()
        elif isinstance(task, UnshuffleTask):
            return task.unshuffle_env.last_event.frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )


class DepthRearrangeSensor(DepthSensorThor):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return task.walkthrough_env.last_event.depth_frame.copy()
        elif isinstance(task, UnshuffleTask):
            return task.unshuffle_env.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )


class UnshuffledRGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
        return walkthrough_env.last_event.frame.copy()


class ClosestUnshuffledRGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    ROT_TO_FORWARD = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            walkthrough_visited_locs = (
                task.locations_visited_in_walkthrough
            )  # A (num unique visited) x 4 matrix
            assert walkthrough_visited_locs is not None

            current_loc = np.array(task.agent_location_tuple).reshape((1, -1))

            diffs = walkthrough_visited_locs - current_loc

            xz_dist = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

            rot_diff = np.array(diffs[:, 2].round(), dtype=int) % 360
            rot_diff = np.minimum(rot_diff, 360 - rot_diff)
            rot_dist = 100 * (180 == rot_diff) + 2 * (90 == rot_diff)

            stand_dist = np.abs(diffs[:, 3]) * STEP_SIZE / 2

            horizon_dist = np.abs(diffs[:, 4]) * STEP_SIZE / 2

            x, z, rotation, standing, horizon = tuple(
                walkthrough_visited_locs[
                    np.argmin(xz_dist + rot_dist + stand_dist + horizon_dist), :
                ]
            )

            walkthrough_env = task.walkthrough_env
            assert task.unshuffle_env.scene == walkthrough_env.scene

            walkthrough_agent_loc = walkthrough_env.get_agent_location()
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )
            if walkthrough_loc_tuple != (x, z, rotation, standing, horizon):
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=x,
                    y=walkthrough_agent_loc["y"],
                    z=z,
                    horizon=horizon,
                    rotation={"x": 0, "y": rotation, "z": 0},
                    standing=standing == 1,
                    forceAction=True,
                )
        return walkthrough_env.last_event.frame.copy()


class InWalkthroughPhaseSensor(
    Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask, WalkthroughTask]]
):
    def __init__(self, uuid: str = "in_walkthrough_phase", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=False, high=True, shape=(1,), dtype=np.bool
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: RearrangeTHOREnvironment,
        task: Optional[UnshuffleTask],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return np.array([True], dtype=bool)
        elif isinstance(task, UnshuffleTask):
            return np.array([False], dtype=bool)
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )
