from typing import (
    Sequence,
    Dict,
    Union,
    cast,
    List,
    Callable,
    Optional,
    Tuple,
    Any,
)

import gym
import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from gym.spaces.dict import Dict as SpaceDict
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, DistributionType
from allenact.base_abstractions.distributions import CategoricalDistr, Distr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from allenact.utils.system import get_logger

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

from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
    Callable,
)

import copy
import math

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from rearrange.baseline_models import RearrangeActorCriticSimpleConvRNN


OBJECT_NAMES = ['FreeSpace', 'OccupiedSpace', 'ToiletPaper', 'StoveKnob', 'SinkBasin', 'Bed', 'ScrubBrush', 'Sofa', 'HandTowelHolder', 'Egg', 'AlarmClock', 'Knife', 'Vase', 'Pot', 'Pen', 'WateringCan', 'WineBottle', 'SideTable', 'Curtains', 'SprayBottle', 'Sink', 'Dresser', 'CreditCard', 'ShelvingUnit', 'Cup', 'SoapBottle', 'Microwave', 'Ladle', 'RoomDecor', 'Fridge', 'ToiletPaperHanger', 'Television', 'StoveBurner', 'PepperShaker', 'Candle', 'PaperTowelRoll', 'FloorLamp', 'Desk', 'Bathtub', 'CellPhone', 'TVStand', 'LightSwitch', 'Plunger', 'DiningTable', 'Window', 'Mug', 'TennisRacket', 'Cabinet', 'Stool', 'Spoon', 'Drawer', 'Floor', 'TowelHolder', 'Watch', 'BaseballBat', 'DeskLamp', 'HousePlant', 'Painting', 'Spatula', 'Fork', 'Boots', 'ButterKnife', 'Dumbbell', 'CounterTop', 'ShowerGlass', 'GarbageCan', 'BathtubBasin', 'SaltShaker', 'Shelf', 'DishSponge', 'Poster', 'Chair', 'Bowl', 'Desktop', 'TableTopDecor', 'Bottle', 'TissueBox', 'Pan', 'DogBed', 'ShowerDoor', 'Plate', 'Newspaper', 'Footstool', 'Laptop', 'Book', 'Blinds', 'TeddyBear', 'Faucet', 'Ottoman', 'GarbageBag', 'Safe', 'Pencil', 'ShowerHead', 'Mirror', 'CoffeeTable', 'LaundryHamper', 'CoffeeMachine', 'ShowerCurtain', 'BasketBall', 'Statue', 'Toaster', 'SoapBar', 'Toilet', 'CD', 'Box', 'ArmChair', 'Kettle', 'RemoteControl']


def spherical_to_cartesian(yaw, elevation):
    """Helper function to convert from a spherical coordinate system
    parameterized by a yaw and elevation to the xyz cartesian coordinate
    with a unit radius, where the z-axis points upwards.

    Arguments:

    yaw: torch.Tensor
        a tensor representing the top-down yaw in radians of the coordinate,
        starting from the positive x-axis and turning counter-clockwise.
    elevation: torch.Tensor
        a tensor representing the elevation in radians of the coordinate,
        about the x-axis, with positive corresponding to upwards tilt.

    Returns:

    point: torch.Tensor
        a tensor corresponding to a point specified by the given yaw and
        elevation, in spherical coordinates.

    """

    # zero elevation and zero yaw points along the positive x-axis
    return np.stack([np.cos(yaw) * np.cos(elevation),
                     np.sin(yaw) * np.cos(elevation),
                     np.sin(elevation)], axis=-1)


def project_camera_rays(image_height, image_width,
                        focal_length_y, focal_length_x,
                        dtype=torch.float32, device='cpu'):
    """Generate a ray for each pixel in an image with a particular map_height
    and map_width by setting up a pinhole camera and sending out a ray from
    the camera to the imaging plane at each pixel location

    Arguments:

    image_height: int
        an integer that described the map_height of the imaging plane in
        pixels and determines the number of rays sampled vertically
    image_width: int
        an integer that described the map_width of the imaging plane in
        pixels and determines the number of rays sampled horizontally
    focal_length_y: float
        the focal length of the pinhole camera, which corresponds to
        the distance to the imaging plane in units of y pixels
    focal_length_x: float
        the focal length of the pinhole camera, which corresponds to
        the distance to the imaging plane in units of x pixels

    Returns:

    rays: torch.Tensor
        a tensor that represents the directions of sampled rays in
        the coordinate system of the camera with shape: [height, width, 3]

    """

    # generate pixel locations for every ray in the imaging plane
    # where the returned shape is: [image_height, image_width]
    kwargs = dict(dtype=dtype, device=device)
    y, x = torch.meshgrid(torch.arange(image_height, **kwargs),
                          torch.arange(image_width, **kwargs), indexing='ij')

    # convert pixel coordinates to the camera coordinate system
    # y is negated to conform to OpenGL convention in computer graphics
    rays_y = (y - 0.5 * float(image_height - 1)) / focal_length_y
    rays_x = (x - 0.5 * float(image_width - 1)) / focal_length_x
    return torch.stack([rays_x, -rays_y, -torch.ones_like(rays_x)], dim=-1)


def transform_rays(rays, eye_vector, up_vector):
    """Given a batch of camera orientations, specified with a viewing
    direction and up vector, convert rays from the camera coordinate
    system to the world coordinate system using a rotation matrix.

    Arguments:

    rays: torch.Tensor
        a batch of rays that have been generated in the coordinate system
        of the camera with shape: [batch, map_height, map_width, 3]
    eye_vector: torch.Tensor
        a batch of viewing directions that are represented as three
        vectors in the world coordinate system with shape: [batch, 3]
    up_vector: torch.Tensor
        a batch of up directions in the imaging plane represented as
        three vectors in the world coordinate system with shape: [batch, 3]

    Returns:

    rays: torch.Tensor
        a batch of rays that have been converted to the coordinate system
        of the world with shape: [batch, map_height, map_width, 3]

    """

    # create a rotation matrix that transforms rays from the camera
    # coordinate system to the world coordinate system
    rotation = torch.stack([torch.cross(
        eye_vector, up_vector), up_vector, -eye_vector], dim=-1)

    # transform the rays using the rotation matrix such that rays project
    # out of the camera in world coordinates in the viewing direction
    return (rays.unsqueeze(-2) *
            rotation.unsqueeze(-3).unsqueeze(-3)).sum(dim=-1)


def bin_rays(bins0, bins1, bins2, origin, rays, depth,
             *features, min_ray_depth=0.1, max_ray_depth=1.5):
    """Given a set of rays and bins that specify the location and size of a
    grid of voxels, return the index of which voxel the end of each ray
    falls into, using a map_depth image to compute this point.

    Arguments:

    bins0: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 0th axis of the coordinate system.
    bins1: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 1st axis of the coordinate system.
    bins2: torch.FloatTensor
        a 1D tensor whose elements specify the locations of boundaries of a
        set of voxels along the 2nd axis of the coordinate system.

    origin: torch.FloatTensor
        the origin of the rays in world coordinates, represented as a batch
        of 3-vectors, shaped like [batch_size, 3].
    rays: torch.FloatTensor
        rays projecting outwards from the origin, ending at a point specified
        by the map_depth, shaped like: [batch_size, height, width, 3].
    map_depth: torch.FloatTensor
        the length of the corresponding ray in world coordinates before
        intersecting a surface, shaped like: [batch_size, height, width].

    features: List[torch.Tensor]
        a list of features for every pixel in the image, such as class
        probabilities shaped like: [batch_size, height, width, num_features]

    min_ray_depth: float
        the minimum distance rays can be to the camera focal point, used to
        handle special cases, such as when the distance is zero.
    max_ray_depth: float
        the maximum distance rays can be to the camera focal point, used to
        handle special cases, such as when the distance is infinity.

    Returns:

    ind_batch: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        in the current batch, shaped like: [num_points].
    ind_x: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [num_points].
    ind_y: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [num_points].
    ind_z: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [num_points].

    binned_features: List[torch.Tensor]
        a list of features for every pixel in the image, such as class
        probabilities, shaped like: [num_points, num_features].

    """

    # bin the point cloud according to which voxel points occupy in space
    # the xyz convention must be known by who is using the function
    rays = origin.unsqueeze(-2).unsqueeze(-2) + rays * depth.unsqueeze(-1)
    ind_x = torch.bucketize(rays[..., 0].contiguous(), bins0, right=True) - 1
    ind_y = torch.bucketize(rays[..., 1].contiguous(), bins1, right=True) - 1
    ind_z = torch.bucketize(rays[..., 2].contiguous(), bins2, right=True) - 1

    # certain rays will be out of bounds of the map or will have a special
    # map_depth value used to signal invalid points, identify them
    criteria = [torch.logical_and(torch.ge(ind_x, 0),
                                  torch.lt(ind_x, bins0.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind_y, 0),
                                  torch.lt(ind_y, bins1.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind_z, 0),
                                  torch.lt(ind_z, bins2.size(dim=0) - 1))]

    # merge each of the criteria into a single mask tensor that will
    # have the shape [batch, image_height, image_width]
    criterion = torch.logical_and(torch.ge(depth.squeeze(-1), min_ray_depth),
                                  torch.le(depth.squeeze(-1), max_ray_depth))
    for next_criterion in criteria:
        criterion = torch.logical_and(criterion, next_criterion)

    # indices where the criterion is true and binned points are valid
    indices = torch.nonzero(criterion, as_tuple=True)
    ind_x, ind_y, ind_z = ind_x[indices], ind_y[indices], ind_z[indices]

    # select a subset of the voxel indices and voxel feature predictions
    # in order to remove points that correspond to invalid voxels
    return indices[0], ind_x, ind_y, ind_z, *[
        features_i[indices] for features_i in features]


def update_feature_map(ind_batch, ind_x, ind_y, ind_z, features, feature_map):
    """Scatter add feature vectors associated with a point cloud onto a
    voxel feature map by adding the features to the locations of each voxel
    using the voxel ids returned by the bin_rays function.

    Arguments:

    ind_batch: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        in the current batch, shaped like: [num_points].
    ind_x: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [num_points].
    ind_y: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [num_points].
    ind_z: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [num_points].

    features: torch.Tensor
        tensor of features that will be added to the feature map using
        torch.scatter_add on the map: [num_points, feature_dim].
    feature_map: torch.Tensor
        tensor of features organized as a three dimensional grid of
        voxels with shape: [batch_size, height, width, depth, num_features].

    """

    # get the size of the spatial dimensions of the map, used to
    # infer flattened map indices for scatter operation
    size0, size1, size2 = feature_map.shape[-3:]
    
    indices = ((ind_batch * size0 + ind_x) * size1 + ind_y) * size2 + ind_z
    feature_map.flatten().scatter_(0, indices, features)

    return feature_map  # return even though map is modified in-place


class VoxelMapEncoder(nn.Module):

    def __init__(
        self,
        maximum_size_x: int = 480,
        maximum_size_y: int = 480,
        maximum_size_z: int = 56,
        voxel_size: float = 0.05,
        fov: float = 90.0,
        image_size: int = 224,
    ):

        super(VoxelMapEncoder, self).__init__()

        self._maximum_size_x = maximum_size_x
        self._maximum_size_y = maximum_size_y
        self._maximum_size_z = maximum_size_z
        self._voxel_size = voxel_size
        self._fov = fov
        self._image_size = image_size

        focal_length = (
            self._image_size / 2.0 / 
            np.tan(np.radians(self._fov) / 2.0)
        )

        self.register_buffer(
            "rays",
            project_camera_rays(
                self._image_size, 
                self._image_size, 
                focal_length, 
                focal_length
            )
        )

        self.register_buffer(
            "bins_x",
            torch.linspace(
                0, 
                self._maximum_size_x * self._voxel_size, 
                self._maximum_size_x + 1
            )
        )
        
        self.register_buffer(
            "bins_y",
            torch.linspace(
                0, 
                self._maximum_size_y * self._voxel_size, 
                self._maximum_size_y + 1
            )
        )

        self.register_buffer(
            "bins_z",
            torch.linspace(
                0, 
                self._maximum_size_z * self._voxel_size,
                self._maximum_size_z + 1
            )
        )

    def single_forward(
        self,
        scene_bounds: torch.FloatTensor,
        features: torch.IntTensor,
        depth: torch.FloatTensor,
        position_vector: torch.FloatTensor,
        at_vector: torch.FloatTensor,
        up_vector: torch.FloatTensor,
        hidden_states: torch.IntTensor,
        masks: torch.IntTensor,
    ):

        ind_batch, ind_x, ind_y, ind_z, features = bin_rays(
            self.bins_x, 
            self.bins_y, 
            self.bins_z, 
            position_vector - scene_bounds[:, 0], 
            transform_rays(self.rays, at_vector, up_vector), 
            depth, 
            features
        )

        return update_feature_map(
            ind_batch,
            ind_x, 
            ind_y, 
            ind_z, 
            features, 
            hidden_states * masks.to(hidden_states.dtype)
        )

    def seq_forward(
        self,
        scene_bounds: torch.FloatTensor,
        features: torch.IntTensor,
        depth: torch.FloatTensor,
        position_vector: torch.FloatTensor,
        at_vector: torch.FloatTensor,
        up_vector: torch.FloatTensor,
        hidden_states: torch.IntTensor,
        masks: torch.IntTensor,
    ):
    
        outputs = []
        nsteps = masks.shape[0]

        for timestep in range(nsteps):

            hidden_states = self.single_forward(
                scene_bounds[timestep],
                features[timestep],
                depth[timestep],
                position_vector[timestep],
                at_vector[timestep],
                up_vector[timestep],
                hidden_states,
                masks[timestep],
            )

            outputs.append(hidden_states)

        return torch.stack(outputs, dim=0), hidden_states

    def forward(
        self,
        scene_bounds: torch.FloatTensor,
        features: torch.IntTensor,
        depth: torch.FloatTensor,
        position_vector: torch.FloatTensor,
        at_vector: torch.FloatTensor,
        up_vector: torch.FloatTensor,
        hidden_states: torch.IntTensor,
        masks: torch.IntTensor,
    ):

        return self.seq_forward(
            scene_bounds,
            features,
            depth,
            position_vector,
            at_vector,
            up_vector,
            hidden_states,
            masks,
        )


class VoxelMapActorCriticRNN(RearrangeActorCriticSimpleConvRNN):

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        maximum_size_x: int = 240,
        maximum_size_y: int = 240,
        maximum_size_z: int = 28,
        voxel_size: float = .1,
        fov: float = 90.0,
        image_size: int = 224,
        egocentric_map_size: int = 16
    ):
    
        self.visual_attention: Optional[nn.Module] = None
        rgb_uuid = unshuffled_rgb_uuid = None
        locals_dict = prepare_locals_for_super(locals())

        locals_dict.pop("maximum_size_x")
        locals_dict.pop("maximum_size_y")
        locals_dict.pop("maximum_size_z")
        locals_dict.pop("voxel_size")
        locals_dict.pop("fov")
        locals_dict.pop("image_size")
        locals_dict.pop("egocentric_map_size")
        super().__init__(**locals_dict)

        self.voxel_map_encoder = VoxelMapEncoder(
            maximum_size_x=maximum_size_x,
            maximum_size_y=maximum_size_y,
            maximum_size_z=maximum_size_z,
            voxel_size=voxel_size,
            fov=fov,
            image_size=image_size
        )

        self._egocentric_map_size = egocentric_map_size

        grid = torch.linspace(
            -self._egocentric_map_size * voxel_size / 2, 
             self._egocentric_map_size * voxel_size / 2, 
             self._egocentric_map_size, dtype=torch.float32
        )

        self.register_buffer(
            'grid', torch.stack(torch.meshgrid(
                grid, 
                grid, 
                grid - (self._egocentric_map_size * voxel_size / 2), 
                indexing='ij'), 
                dim=-1
            )
        )

    def _create_visual_encoder(self) -> nn.Module:

        visual_encoder = nn.Sequential(
            nn.Embedding(256, 32),
            Rearrange('s b h w d c q -> (s b) (c q) h w d'),

            nn.Conv3d(64, 64, 3, padding=1, stride=2), 
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, padding=1), 
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, padding=1, stride=2), 
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 64, 3, padding=1), 
            nn.ReLU(inplace=True),

            Rearrange('b c h w d -> b (c h w d)'),
            nn.Linear(4 * 4 * 4 * 64, self._hidden_size)
        )

        visual_encoder.apply(simple_conv_and_linear_weights_init)

        return visual_encoder

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            unshuffle_map=(
                (
                    ("sampler", None),
                    ("hidden", (
                        self.voxel_map_encoder._maximum_size_x * 
                        self.voxel_map_encoder._maximum_size_y * 
                        self.voxel_map_encoder._maximum_size_z
                    )),
                ),
                torch.uint8,
            ),
            walkthrough_map=(
                (
                    ("sampler", None),
                    ("hidden", (
                        self.voxel_map_encoder._maximum_size_x * 
                        self.voxel_map_encoder._maximum_size_y * 
                        self.voxel_map_encoder._maximum_size_z
                    )),
                ),
                torch.uint8,
            )
        )

    def get_map_indices(self, ind_x, ind_y, ind_z):
        
        return (  # tensor shape: [seq_len, batch, 40, 40, 40]
            ind_x.clamp(min=0, max=self.voxel_map_encoder._maximum_size_x - 1) * 
            self.voxel_map_encoder._maximum_size_y + 
            ind_y.clamp(min=0, max=self.voxel_map_encoder._maximum_size_y - 1)
        ) * self.voxel_map_encoder._maximum_size_z + \
            ind_z.clamp(min=0, max=self.voxel_map_encoder._maximum_size_z - 1)

    def get_egocentric_map(
        self,
        map_outputs,
        scene_bounds, 
        position_vector, 
        at_vector, 
        up_vector
    ):

        rotation_matrix = torch.stack([
            torch.cross(at_vector, up_vector), 
            up_vector, -at_vector
        ], dim=-1)

        grid = torch.einsum(
            'xynm,abcm->xyabcn', 
            rotation_matrix, 
            self.grid
        )

        batch_shape = map_outputs.shape[:2]
        
        grid += position_vector.view(*batch_shape, 1, 1, 1, 3)
        grid -= scene_bounds[:, :, 0].view(*batch_shape, 1, 1, 1, 3)
        
        ind_x = torch.bucketize(
            grid[..., 0], 
            self.voxel_map_encoder.bins_x, 
            right=True
        ) - 1
        ind_y = torch.bucketize(
            grid[..., 1], 
            self.voxel_map_encoder.bins_y, 
            right=True
        ) - 1
        ind_z = torch.bucketize(
            grid[..., 2], 
            self.voxel_map_encoder.bins_z, 
            right=True
        ) - 1

        map_indices = self.get_map_indices(ind_x, ind_y, ind_z)

        egocentric_map_outputs = torch.gather(
            map_outputs.view(*batch_shape, -1), 2,
            map_indices.view(*batch_shape, -1)
        ).view(*batch_shape, 
               self._egocentric_map_size, 
               self._egocentric_map_size, 
               self._egocentric_map_size)

        map_indices_valid_x = torch.logical_and(
            ind_x >= 0, ind_x < self.voxel_map_encoder._maximum_size_x
        )
        map_indices_valid_y = torch.logical_and(
            ind_y >= 0, ind_y < self.voxel_map_encoder._maximum_size_y
        )
        map_indices_valid_z = torch.logical_and(
            ind_z >= 0, ind_z < self.voxel_map_encoder._maximum_size_z
        )
        
        map_indices_valid = torch.logical_and(
            map_indices_valid_x, torch.logical_and(
                map_indices_valid_y, map_indices_valid_z
            )
        )

        return torch.where(map_indices_valid, 
                           egocentric_map_outputs, 0)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        unshuffle_features = observations['map']['unshuffle_features']
        unshuffle_depth = observations['map']['unshuffle_depth']
        walkthrough_features = observations['map']['walkthrough_features']
        walkthrough_depth = observations['map']['walkthrough_depth']

        scene_bounds = observations['map']['scene_bounds']
        position_vector = observations['map']['position_vector']
        at_vector = observations['map']['at_vector']
        up_vector = observations['map']['up_vector']

        batch_shape = masks.shape[:2]

        unshuffle_map_hidden_states = memory.tensor("unshuffle_map").view(
            batch_shape[1], 
            self.voxel_map_encoder._maximum_size_x,
            self.voxel_map_encoder._maximum_size_y, 
            self.voxel_map_encoder._maximum_size_z
        )
        walkthrough_map_hidden_states = memory.tensor("walkthrough_map").view(
            batch_shape[1], 
            self.voxel_map_encoder._maximum_size_x,
            self.voxel_map_encoder._maximum_size_y, 
            self.voxel_map_encoder._maximum_size_z
        )

        unshuffle_map_outputs, unshuffle_map_hidden_states = self.voxel_map_encoder(
            scene_bounds,
            unshuffle_features,
            unshuffle_depth,
            position_vector,
            at_vector,
            up_vector,
            unshuffle_map_hidden_states,
            masks.view(*batch_shape, 1, 1, 1)
        )
        walkthrough_map_outputs, walkthrough_map_hidden_states = self.voxel_map_encoder(
            scene_bounds,
            walkthrough_features,
            walkthrough_depth,
            position_vector,
            at_vector,
            up_vector,
            walkthrough_map_hidden_states,
            masks.view(*batch_shape, 1, 1, 1)
        )
        
        unshuffle_map_hidden_states = unshuffle_map_hidden_states.view(
            batch_shape[1], 
            self.voxel_map_encoder._maximum_size_x * 
            self.voxel_map_encoder._maximum_size_y * 
            self.voxel_map_encoder._maximum_size_z
        )
        walkthrough_map_hidden_states = walkthrough_map_hidden_states.view(
            batch_shape[1], 
            self.voxel_map_encoder._maximum_size_x * 
            self.voxel_map_encoder._maximum_size_y * 
            self.voxel_map_encoder._maximum_size_z
        )

        egocentric_unshuffle_map_outputs = self.get_egocentric_map(
            unshuffle_map_outputs, scene_bounds, 
            position_vector, at_vector, up_vector
        )
        egocentric_walkthrough_map_outputs = self.get_egocentric_map(
            walkthrough_map_outputs, scene_bounds, 
            position_vector, at_vector, up_vector
        )

        egocentric_map_outputs = torch.stack([
            egocentric_unshuffle_map_outputs, 
            egocentric_walkthrough_map_outputs
        ], dim=-1)

        x = self.visual_encoder(egocentric_map_outputs.long())

        x, rnn_hidden_states = self.state_encoder(
            x.view(*batch_shape, -1), memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        memory = memory.set_tensor("unshuffle_map", unshuffle_map_hidden_states)
        memory = memory.set_tensor("walkthrough_map", walkthrough_map_hidden_states)
        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class MappingInputsSensor(Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    def __init__(self, uuid="map"):

        observation_space = gym.spaces.Dict([

            ("scene_bounds", gym.spaces.Box(
                np.full([2, 3], -20.0, dtype=np.float32), 
                np.full([2, 3],  20.0, dtype=np.float32), dtype=np.float32)),

            ("unshuffle_features", gym.spaces.MultiDiscrete(
                np.full([224, 224], len(OBJECT_NAMES), dtype=np.uint8), dtype=np.uint8)),

            ("unshuffle_depth", gym.spaces.Box(
                np.full([224, 224], -20.0, dtype=np.float32), 
                np.full([224, 224],  20.0, dtype=np.float32), dtype=np.float32)),

            ("walkthrough_features", gym.spaces.MultiDiscrete(
                np.full([224, 224], len(OBJECT_NAMES), dtype=np.uint8), dtype=np.uint8)),

            ("walkthrough_depth", gym.spaces.Box(
                np.full([224, 224], -20.0, dtype=np.float32), 
                np.full([224, 224],  20.0, dtype=np.float32), dtype=np.float32)),

            ("position_vector", gym.spaces.Box(
                np.full([3], -20.0, dtype=np.float32), 
                np.full([3],  20.0, dtype=np.float32), dtype=np.float32)),

            ("at_vector", gym.spaces.Box(
                np.full([3], -20.0, dtype=np.float32), 
                np.full([3],  20.0, dtype=np.float32), dtype=np.float32)),

            ("up_vector", gym.spaces.Box(
                np.full([3], -20.0, dtype=np.float32), 
                np.full([3],  20.0, dtype=np.float32), dtype=np.float32)),

        ])

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )
        
        metadata = task.env.controller.step(action="GetReachablePositions").metadata
        bounds = metadata["sceneBounds"]["cornerPoints"]

        bounds = np.stack([np.min(bounds, axis=0), 
                           np.max(bounds, axis=0)], axis=0)[:, [0, 2, 1]]

        location = task.env.get_agent_location()
        crouch_height_offset = 0.675 if location["standing"] else 0.0
        
        position_vector = np.array([
            location["x"], location["z"], 
            location["y"] + crouch_height_offset])

        rotation = -location["rotation"] / 180.0 * np.pi + np.pi / 2
        horizon = -location["horizon"] / 180.0 * np.pi

        at_vector = spherical_to_cartesian(rotation, horizon)
        up_vector = spherical_to_cartesian(rotation, horizon + np.pi / 2)
 
        semantic_image_colors = task.unshuffle_env.last_event.semantic_segmentation_frame
        semantic_features = np.ones([224, 224], dtype=np.uint8)

        for unique_color in np.unique(
                semantic_image_colors.reshape(224 * 224, 3), axis=0):

            if tuple(unique_color.tolist()) not in \
                task.unshuffle_env.last_event.color_to_object_id: continue

            object_id = task.unshuffle_env.last_event\
                .color_to_object_id[tuple(unique_color.tolist())]

            if object_id not in OBJECT_NAMES:
                continue

            object_class_idx = OBJECT_NAMES.index(
                object_id
            )

            unique_color = unique_color[np.newaxis, np.newaxis]

            ind_y, ind_x = np.nonzero(np.equal(
                semantic_image_colors, unique_color).all(axis=2))

            ind_c = np.full(ind_y.shape, object_class_idx)

            semantic_features[ind_y, ind_x] = ind_c

        unshuffle_semantic_features = semantic_features
 
        semantic_image_colors = task.walkthrough_env.last_event.semantic_segmentation_frame
        semantic_features = np.ones([224, 224], dtype=np.uint8)

        for unique_color in np.unique(
                semantic_image_colors.reshape(224 * 224, 3), axis=0):

            if tuple(unique_color.tolist()) not in \
                task.walkthrough_env.last_event.color_to_object_id: continue

            object_id = task.walkthrough_env.last_event\
                .color_to_object_id[tuple(unique_color.tolist())]

            if object_id not in OBJECT_NAMES:
                continue

            object_class_idx = OBJECT_NAMES.index(
                object_id
            )

            unique_color = unique_color[np.newaxis, np.newaxis]

            ind_y, ind_x = np.nonzero(np.equal(
                semantic_image_colors, unique_color).all(axis=2))

            ind_c = np.full(ind_y.shape, object_class_idx)

            semantic_features[ind_y, ind_x] = ind_c

        walkthrough_semantic_features = semantic_features

        return OrderedDict([

            ("scene_bounds", bounds.astype(np.float32)),

            ("unshuffle_features", unshuffle_semantic_features.astype(np.uint8)),
            ("unshuffle_depth", task.unshuffle_env.last_event.depth_frame.astype(np.float32)),
            
            ("walkthrough_features", walkthrough_semantic_features.astype(np.uint8)),
            ("walkthrough_depth", task.walkthrough_env.last_event.depth_frame.astype(np.float32)),

            ("position_vector", position_vector.astype(np.float32)),
            ("at_vector", at_vector.astype(np.float32)),
            ("up_vector", up_vector.astype(np.float32)),
        
        ])