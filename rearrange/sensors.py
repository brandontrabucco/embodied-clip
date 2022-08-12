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


class ExpertRaysSensor(
    Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]
):

    NUM_PATCHES = 3
    MAX_OBJECTS_SHUFFLED = 8

    EXPERT_RAYS_LABEL = "expert_rays"

    EXPERT_CLASSES_LABEL = "expert_classes"
    EXPERT_INSTANCES_LABEL = "expert_instances"

    @staticmethod
    def distance_to_object(x, object_i):
        return np.sqrt((x["x"] - object_i["position"]["x"]) ** 2 + 
                       (x["z"] - object_i["position"]["z"]) ** 2)

    def __init__(self, uuid="nerf"):

        observation_space = gym.spaces.Dict(
            [
                (self.EXPERT_RAYS_LABEL, 
                 gym.spaces.Box(np.full([2 * self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES * 5], -20.0), 
                                np.full([2 * self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES * 5],  20.0))),

                (self.EXPERT_CLASSES_LABEL, 
                 gym.spaces.MultiDiscrete(np.full([
                     2 * self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),
                                                
                (self.EXPERT_INSTANCES_LABEL, 
                 gym.spaces.MultiDiscrete(np.full([
                     2 * self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES], len(CLASS_TO_ID)))),
            ]
        )
        
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )

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

                unshuffle_rays.append(pose_u)
                walkthrough_rays.append(pose_w)

                unshuffle_class.append(CLASS_TO_ID[object_u["type"]])
                walkthrough_class.append(CLASS_TO_ID[object_w["type"]])

                unshuffle_instance.append(class_counts[object_u["type"]])
                walkthrough_instance.append(class_counts[object_w["type"]])

            class_counts[object_u["type"]] += 1
            
        unshuffle_rays = np.concatenate(unshuffle_rays, axis=0)
        walkthrough_rays = np.concatenate(walkthrough_rays, axis=0)

        unshuffle_class = np.array(unshuffle_class)
        walkthrough_class = np.array(walkthrough_class)

        unshuffle_instance = np.array(unshuffle_instance)
        walkthrough_instance = np.array(walkthrough_instance)

        add_padding = 2 * self.MAX_OBJECTS_SHUFFLED * self.NUM_PATCHES - 2 * unshuffle_class.size

        return OrderedDict([  # return rays and discrete labels

            (self.EXPERT_RAYS_LABEL, np.concatenate([
                unshuffle_rays, 
                walkthrough_rays,
                np.full([add_padding * 5], 0.0)], axis=0)),

            (self.EXPERT_CLASSES_LABEL, np.concatenate([
                unshuffle_class, 
                walkthrough_class,
                np.full([add_padding], 0)], axis=0)),

            (self.EXPERT_INSTANCES_LABEL, np.concatenate([
                unshuffle_instance, 
                walkthrough_instance,
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
