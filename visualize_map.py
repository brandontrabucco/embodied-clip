import numpy as np
from PIL import Image


color_map = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)


if __name__ == "__main__":

    semantic_map = np.load(
        "semantic_maps/thor-FloorPlan306-7-train-walkthrough-feature_map.npy"
    )

    occupancy = np.not_equal(semantic_map, 0).any(axis=3).astype(np.float32)




    mask = (occupancy[:, :, ::-1].cumsum(axis=2)[:, :, ::-1] <= 1).astype(np.float32)

    top_down_voxels = occupancy * mask

    assert top_down_voxels.sum(axis=2).max() == 1.0

    top_down_semantic_map = (semantic_map * top_down_voxels[..., np.newaxis]).sum(axis=2)

    top_down_semantic_map_image = color_map[top_down_semantic_map.argmax(axis=-1)]

    Image.fromarray(top_down_semantic_map_image).save("top_down_semantic_map_image.png")


    

    mask = (occupancy[:, ::-1, :].cumsum(axis=1)[:, ::-1, :] <= 1).astype(np.float32)

    side_view_voxels = occupancy * mask

    assert side_view_voxels.sum(axis=1).max() == 1.0

    side_view_semantic_map = (semantic_map * side_view_voxels[..., np.newaxis]).sum(axis=1)

    side_view_semantic_map_image = color_map[side_view_semantic_map.argmax(axis=-1)]

    Image.fromarray(side_view_semantic_map_image).save("side_view_semantic_map_image.png")