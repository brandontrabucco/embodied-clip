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

    # select the indices and ray coordinates
    ind0, ind1, ind2, rays = ind0[  # that are within the world bounds
        indices], ind1[indices], ind2[indices], rays[indices]

    # select a subset of the voxel indices and voxel feature predictions
    # in order to remove points that correspond to invalid voxels
    return ind0, ind1, ind2, *[features_i[indices] for features_i in features]


def update_feature_map(ind0, ind1, ind2, features, feature_map, hits_per_voxel):
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
    hits_per_voxel_flat = hits_per_voxel.view(-1, size0 * size1 * size2, 1)

    # expand the indices tensor with an additional axis so that
    indices = (ind0 * size1 + ind1) * size2 + ind2
    indices = indices.unsqueeze(-1).expand(*(list(indices.shape) + [num_features]))

    if len(indices.shape) < 3:
        indices = indices.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)

    # zero the features at all observed voxels in the feature
    # map and assign voxels to interpolated features
    feature_map_flat.scatter_(-2, indices, features, reduce="add")
    hits_per_voxel_flat.scatter_(-2, indices[..., :1], 1.0, reduce="add")


def get_feature_map(task, voxel_size=0.2, max_images=1000, feature_size=512, room_height=2.75):

    metadata = task.env.controller.step(action="GetReachablePositions").metadata

    bounds = metadata["sceneBounds"]["cornerPoints"]
    bounds = np.stack([np.min(bounds, axis=0), np.max(bounds, axis=0)], axis=0)
    
    size0 = int(np.ceil((bounds[1, 0] - bounds[0, 0]) / voxel_size))
    size1 = int(np.ceil((bounds[1, 2] - bounds[0, 2]) / voxel_size))
    size2 = int(np.ceil(room_height / voxel_size))

    valid_positions = [dict(position=position,
                            rotation=dict(x=0, y=rotation, z=0),
                            horizon=horizon, standing=standing)
                    for position in metadata["actionReturn"]
                    for rotation in (0, 90, 180, 270)
                    for horizon in (-30, 0, 30, 60)
                    for standing in (True, False)]

    valid_positions = [valid_positions[idx] for idx in
                       np.random.permutation(len(valid_positions))]

    feature_map = torch.zeros(size0, size1, size2, feature_size).cuda()
    hits_per_voxel = torch.zeros(size0, size1, size2, 1).cuda()

    bins0 = torch.linspace(bounds[0, 0], bounds[1, 0], size0 + 1).cuda()
    bins1 = torch.linspace(bounds[0, 2], bounds[1, 2], size1 + 1).cuda()
    bins2 = torch.linspace(bounds[0, 1], bounds[0, 1] + room_height, size2 + 1).cuda()

    for config in valid_positions[:max_images]:
        task.env.controller.step(action="TeleportFull", **config)

        obs = task.get_observations()

        # grab the position of the agent and copy to the right device
        # the absolute position of the agent in meters
        location = task.env.get_agent_location()
        crouch_height_offset = 0.675 if location["standing"] else 0.0
        position = torch.FloatTensor([location["x"], location["z"],
                                      location["y"] + crouch_height_offset]).cuda()

        # grab the yaw of the agent ensuring that zero yaw corresponds to
        # the x axis, and positive yaw is rotates counterclockwise
        yaw = task.env.get_agent_location()["rotation"] / 180.0 * np.pi
        yaw = torch.tensor(-yaw + np.pi / 2, dtype=torch.float32).cuda()

        # grab the yaw of the agent ensuring that zero yaw corresponds to
        # the x axis, and positive yaw is rotates counterclockwise
        elevation = torch.tensor(-task.env.get_agent_location()["horizon"]
                                / 180.0 * np.pi, dtype=torch.float32).cuda()

        eye_vector = spherical_to_cartesian(yaw, elevation)
        up_vector = spherical_to_cartesian(yaw, elevation + np.pi / 2)

        rotation_matrix = torch.stack([torch.cross(
            eye_vector, up_vector), up_vector, -eye_vector], dim=-1)

        pose = torch.cat([rotation_matrix, position.unsqueeze(-1)], dim=-1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]], 
                                             dtype=torch.float32, 
                                             device=pose.device)], dim=-2)

        with torch.no_grad():

            image = Image.fromarray(np.uint8(255.0 * obs["rgb"]))
            image = preprocess(image).unsqueeze(0).cuda()

            image_features = model.encode_image(image)
            image_features = image_features[0].float().permute(1, 2, 0)

            focal_length = (28 / 2.0 / np.tan(np.radians(90.0) / 2.0))

            rays = project_camera_rays(28, 28, focal_length, focal_length).cuda()
            rays = transform_rays(rays, eye_vector, up_vector)

            depth = torch.tensor(obs["depth"][4::8, 4::8]).cuda()

            ind0, ind1, ind2, image_features = bin_rays(
                bins0, bins1, bins2, position, rays, depth, image_features)

            update_feature_map(ind0, ind1, ind2, image_features, 
                               feature_map, hits_per_voxel)
            
    coords0 = (bins0[1:] + bins0[:-1]) / 2
    coords1 = (bins1[1:] + bins1[:-1]) / 2
    coords2 = (bins2[1:] + bins2[:-1]) / 2
            
    coords0 = coords0.view(size0, 1, 1).expand(size0, size1, size2)
    coords1 = coords1.view(1, size1, 1).expand(size0, size1, size2)
    coords2 = coords2.view(1, 1, size2).expand(size0, size1, size2)
    
    coords = torch.stack((coords0, coords1, coords2), dim=3) 
    feature_map /= hits_per_voxel.clamp(min=1)

    return (coords.cpu().numpy(), 
            feature_map.cpu().numpy(), 
            hits_per_voxel.cpu().numpy())


if __name__ == "__main__":

    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1
    else:
        distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    device_id = rank % torch.cuda.device_count()
    print(f'Initialized process {rank} / {world_size}')

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if rank == 0 or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    model, preprocess = clip.load("RN50", device='cuda')

    model.visual.attnpool = torch.nn.Identity()

    model.visual.layer4 = torch.nn.Identity()
    model.visual.layer3 = torch.nn.Identity()

    model.eval()

    os.makedirs("maps", exist_ok=True)

    for stage in ["train", "val", "test"]:

        task_sampler_args = ExperimentConfig.stagewise_task_sampler_args(
            stage=stage, devices=[device_id], process_ind=rank, total_processes=world_size)

        task_sampler = ExperimentConfig.make_sampler_fn(
            **task_sampler_args, force_cache_reset=False, epochs=1)
        
        num_tasks = task_sampler.length // 2

        for task_id in range(num_tasks):

            # walkthrough phase

            task = task_sampler.next_task()
            coords, feature_map, hits_per_voxel = get_feature_map(task)

            hits_per_voxel_flat = hits_per_voxel.sum(axis=2)

            coords_flat = (coords * hits_per_voxel).sum(axis=2) / hits_per_voxel_flat.clip(min=1)
            feature_map_flat = (feature_map * hits_per_voxel).sum(axis=2) / hits_per_voxel_flat.clip(min=1)

            scene = task.env.scene
            index = task.env.current_task_spec.metrics.get("index")
            stage = task.env.current_task_spec.stage

            prefix = f"thor-{scene}-{index}-{stage}-walkthrough"

            occupied_indices = np.nonzero(hits_per_voxel[..., 0])
            occupied_indices_flat = np.nonzero(hits_per_voxel_flat[..., 0])
            
            coords = coords[occupied_indices]
            feature_map = feature_map[occupied_indices]

            coords_flat = coords_flat[occupied_indices_flat]
            feature_map_flat = feature_map_flat[occupied_indices_flat]

            print(f"[{stage}: {task_id}/{num_tasks}] \
{prefix} {feature_map.shape[0]} / {np.prod(hits_per_voxel.shape)} voxels are occupied")

            np.save(os.path.join(
                "maps", f"{prefix}-coords.npy"), coords)
            np.save(os.path.join(
                "maps", f"{prefix}-feature_map.npy"), feature_map)

            np.save(os.path.join(
                "maps", f"{prefix}-coords_flat.npy"), coords_flat)
            np.save(os.path.join(
                "maps", f"{prefix}-feature_map_flat.npy"), feature_map_flat)

            # unshuffle phase

            task = task_sampler.next_task()
            coords, feature_map, hits_per_voxel = get_feature_map(task)

            hits_per_voxel_flat = hits_per_voxel.sum(axis=2)

            coords_flat = (coords * hits_per_voxel).sum(axis=2) / hits_per_voxel_flat.clip(min=1)
            feature_map_flat = (feature_map * hits_per_voxel).sum(axis=2) / hits_per_voxel_flat.clip(min=1)

            scene = task.env.scene
            index = task.env.current_task_spec.metrics.get("index")
            stage = task.env.current_task_spec.stage

            prefix = f"thor-{scene}-{index}-{stage}-unshuffle"

            occupied_indices = np.nonzero(hits_per_voxel[..., 0])
            occupied_indices_flat = np.nonzero(hits_per_voxel_flat[..., 0])

            coords = coords[occupied_indices]
            feature_map = feature_map[occupied_indices]

            coords_flat = coords_flat[occupied_indices_flat]
            feature_map_flat = feature_map_flat[occupied_indices_flat]

            print(f"[{stage}: {task_id}/{num_tasks}] \
{prefix} {feature_map.shape[0]} / {np.prod(hits_per_voxel.shape)} voxels are occupied")

            np.save(os.path.join(
                "maps", f"{prefix}-coords.npy"), coords)
            np.save(os.path.join(
                "maps", f"{prefix}-feature_map.npy"), feature_map)

            np.save(os.path.join(
                "maps", f"{prefix}-coords_flat.npy"), coords_flat)
            np.save(os.path.join(
                "maps", f"{prefix}-feature_map_flat.npy"), feature_map_flat)