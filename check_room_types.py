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
        stage="val", devices=[0], process_ind=0, total_processes=1)

    task_sampler = exp_config.make_sampler_fn(
        **task_sampler_args, force_cache_reset=False, epochs=1)

    os.makedirs("examples/", exist_ok=True)

    for task_id in range(20):

        task = task_sampler.next_task()
        observations = task.get_observations()

        for i in range(4):

            task.step(task.action_names().index("rotate_left"))

            image = task.env.controller.last_event.frame

            Image.fromarray(image).save(f"examples/{task_id}-{i}.png")
            print(f"saved: examples/{task_id}-{i}.png")

        for _ in range(50):
            next(task_sampler.task_spec_iterator)

