from baseline_configs.one_phase.one_phase_rgb_expert_dagger import (
    OnePhaseRGBExpertDaggerExperimentConfig
)
from allenact.utils.inference import InferenceAgent

import torch
import numpy as np
from collections import defaultdict, namedtuple

import stringcase
import pickle as pkl
from PIL import Image
import os


if __name__ == "__main__":

    exp_config = OnePhaseRGBExpertDaggerExperimentConfig()

    task_sampler_args = exp_config.stagewise_task_sampler_args(
        stage="train", devices=[0], process_ind=0, total_processes=1)

    task_sampler = exp_config.make_sampler_fn(
        **task_sampler_args, force_cache_reset=False, epochs=1)

    os.makedirs("examples/", exist_ok=True)

    max_size = np.zeros([3])

    for task_id in range(80):

        task = task_sampler.next_task()
        
        metadata = task.env.controller.step(action="GetReachablePositions").metadata
        bounds = metadata["sceneBounds"]["cornerPoints"]

        bounds = np.stack([np.min(bounds, axis=0), 
                           np.max(bounds, axis=0)], axis=0)[:, [0, 2, 1]]

        max_size = np.maximum(max_size, bounds[1] - bounds[0])

        print(max_size)

        for _ in range(50):
            next(task_sampler.task_spec_iterator)

