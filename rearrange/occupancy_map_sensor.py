from typing import Any, Optional, Union

import os
import gym.spaces
import numpy as np
import torch
import torch.nn.functional as functional
from allenact.base_abstractions.sensor import Sensor
from collections import OrderedDict, defaultdict

from allenact.utils.misc_utils import prepare_locals_for_super
import torch.distributed as dist

from rearrange.constants import STEP_SIZE
from rearrange.environment import RearrangeTHOREnvironment
from rearrange.tasks import (
    UnshuffleTask,
    WalkthroughTask,
    AbstractRearrangeTask,
)


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
    return torch.stack([torch.cos(yaw) * torch.cos(elevation),
                        torch.sin(yaw) * torch.cos(elevation),
                        torch.sin(elevation)], dim=-1)


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
        intersecting a surface, shaped like: [batch_size, height, width, 1].

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

    ind0: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [batch_size, num_points].
    ind1: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [batch_size, num_points].
    ind2: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [batch_size, num_points].

    binned_features: List[torch.Tensor]
        a list of features for every pixel in the image, such as class
        probabilities, shaped like: [batch_size, num_points, num_features].

    """

    # bin the point cloud according to which voxel points occupy in space
    # the xyz convention must be known by who is using the function
    rays = origin.unsqueeze(-2).unsqueeze(-2) + rays * depth
    ind0 = torch.bucketize(rays[..., 0].contiguous(), bins0, right=True) - 1
    ind1 = torch.bucketize(rays[..., 1].contiguous(), bins1, right=True) - 1
    ind2 = torch.bucketize(rays[..., 2].contiguous(), bins2, right=True) - 1

    # certain rays will be out of bounds of the map or will have a special
    # map_depth value used to signal invalid points, identify them
    criteria = [torch.logical_and(torch.ge(ind0, 0),
                                  torch.lt(ind0, bins0.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind1, 0),
                                  torch.lt(ind1, bins1.size(dim=0) - 1)),
                torch.logical_and(torch.ge(ind2, 0),
                                  torch.lt(ind2, bins2.size(dim=0) - 1))]

    # merge each of the criteria into a single mask tensor that will
    # have the shape [batch, image_height, image_width]
    criterion = torch.logical_and(torch.ge(depth.squeeze(-1), min_ray_depth),
                                  torch.le(depth.squeeze(-1), max_ray_depth))
    for next_criterion in criteria:
        criterion = torch.logical_and(criterion, next_criterion)

    # indices where the criterion is true and binned points are valid
    indices = torch.nonzero(criterion, as_tuple=True)
    ind0, ind1, ind2 = ind0[indices], ind1[indices], ind2[indices]

    # select a subset of the voxel indices and voxel feature predictions
    # in order to remove points that correspond to invalid voxels
    return ind0, ind1, ind2, *[features_i[indices] for features_i in features]


def update_feature_map(ind0, ind1, ind2, features, feature_map):
    """Scatter add feature vectors associated with a point cloud onto a
    voxel feature map by adding the features to the locations of each voxel
    using the voxel ids returned by the bin_rays function.

    Arguments:

    ind0: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 0, shaped like: [batch_size, num_points].
    ind1: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 1, shaped like: [batch_size, num_points].
    ind2: torch.LongTensor
        voxel ids associated with the point cloud according to their position
        along axis 2, shaped like: [batch_size, num_points].

    features: torch.Tensor
        tensor of features that will be added to the feature map using
        torch.scatter_add on the map: [batch_size, num_points, feature_dim].
    feature_map: torch.Tensor
        tensor of features organized as a three dimensional grid of
        voxels with shape: [batch_size, height, width, depth, num_features].

    """

    # get the size of the spatial dimensions of the map, used to
    # infer the batch size of the map if present
    size0, size1, size2, num_features = feature_map.shape[-4:]
    feature_map_flat = feature_map.view(-1, size0 * size1 * size2, num_features)

    # expand the indices tensor with an additional axis so that
    indices = (ind0 * size1 + ind1) * size2 + ind2
    indices = indices.unsqueeze(-1).expand(*(list(indices.shape) + [num_features]))

    if len(indices.shape) < 3:
        indices = indices.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)

    # zero the features at all observed voxels in the feature
    # map and assign voxels to interpolated features
    feature_map_flat.scatter_(-2, indices, features)


class OccupancyMapSensor(Sensor[
        RearrangeTHOREnvironment, Union[UnshuffleTask]]):

    OBSERVATION_LABEL = "top_down"

    def __init__(self, uuid="occupancy_map", 
                 window_size=1, voxel_size=0.05, room_height=2.75):

        self.device = (f'cuda:{dist.get_rank()}' 
                       if dist.is_initialized() else 'cuda')

        self.window_size = window_size

        self.voxel_size = voxel_size
        self.room_height = room_height

        observation_space = gym.spaces.Dict([

            (self.OBSERVATION_LABEL, 

                gym.spaces.Box(
                    
                    np.full([
                        window_size *
                        window_size *
                        window_size
                    ], -1.0), 

                    np.full([
                        window_size *
                        window_size *
                        window_size
                    ],  1.0))),

            ])
        
        super().__init__(**prepare_locals_for_super(locals()))

        self.cached_task = None
        self.cached_occupancy_map = None

        self.cached_bins_x = None
        self.cached_bins_y = None
        self.cached_bins_z = None

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        focal_length = 224 / 2.0 / np.tan(np.radians(90.0) / 2.0)
        self.rays = project_camera_rays(224, 224, focal_length, focal_length, device=self.device)

    def create_map(self, env, task, voxel_size=0.05, room_height=2.75):
        
        metadata = env.controller.step(action="GetReachablePositions").metadata
        bounds = metadata["sceneBounds"]["cornerPoints"]
        bounds = np.stack([np.min(bounds, axis=0), np.max(bounds, axis=0)], axis=0)
        
        size_x = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / voxel_size))
        size_y = int(np.ceil((bounds[1, 2] - bounds[0, 2]) / voxel_size))
        size_z = int(np.ceil(room_height / voxel_size))

        del self.cached_occupancy_map
        del self.cached_bins_x
        del self.cached_bins_y
        del self.cached_bins_z

        torch.cuda.empty_cache()

        self.cached_occupancy_map = torch.zeros(size_x, size_y, size_z, 1, dtype=torch.uint8, device=self.device)

        self.cached_bins_x = torch.linspace(
            bounds[0, 0], bounds[1, 0], size_x + 1, device=self.device)
        self.cached_bins_y = torch.linspace(
            bounds[0, 2], bounds[1, 2], size_y + 1, device=self.device)
        self.cached_bins_z = torch.linspace(
            bounds[0, 1], bounds[0, 1] + 
            room_height, size_z + 1, device=self.device)

    def get_observation(self, env, task) -> Any:

        if not isinstance(task, UnshuffleTask):
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `UnshuffleTask`."
            )
        
        index = task.env.current_task_spec.metrics.get("index")
        stage = task.env.current_task_spec.stage
        scene = task.env.scene
        
        if self.cached_task != (index, stage, scene):

            self.create_map(env, task, voxel_size=
                            self.voxel_size, room_height=self.room_height)

            self.cached_task = (index, stage, scene)

        with torch.no_grad():

            location = task.env.get_agent_location()
            crouch_height_offset = 0.675 if location["standing"] else 0.0

            position = torch.tensor(
                [location["x"], 
                 location["z"],
                 location["y"] + crouch_height_offset], device=self.device)

            yaw = task.env.get_agent_location()["rotation"] / 180 * np.pi
            yaw = torch.tensor(-yaw + np.pi / 2, dtype=torch.float32, device=self.device)

            elevation = torch.tensor(-task.env.get_agent_location()["horizon"]
                                     / 180.0 * np.pi, dtype=torch.float32, device=self.device)

            to_vector = spherical_to_cartesian(yaw, elevation)
            up_vector = spherical_to_cartesian(yaw, elevation + np.pi / 2)

            rays = transform_rays(self.rays, to_vector, up_vector)

            depth = task.env.last_event.depth_frame
            depth = torch.tensor(depth, device=self.device).unsqueeze(2)

            ind0, ind1, ind2, features = bin_rays(
                self.cached_bins_x, self.cached_bins_y, 
                self.cached_bins_z, position, rays, depth, torch.ones_like(depth, dtype=torch.uint8))

            update_feature_map(ind0, ind1, ind2, features, self.cached_occupancy_map)

        return OrderedDict([

            (self.OBSERVATION_LABEL, 
                np.zeros([self.window_size ** 3])),
        
        ])