from baseline_configs.one_phase.one_phase_rgb_il_base \
    import OnePhaseRGBILBaseExperimentConfig as BaseExperimentConfig

from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from rearrange.attention_models import TransformerXL
import datagen.datagen_utils as datagen_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence as pad
from torch.distributions.categorical import Categorical

import os
import sys
import time

import queue
import torch.multiprocessing as multiprocessing

from torch.multiprocessing import Process, Queue
from collections import defaultdict

import glob
import tqdm
import tree
import numpy as np

import argparse


class ExperimentConfig(BaseExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")


class RolloutEngine(Process):

    def __init__(self, device: int, 
                 rank: int, 
                 world_size: int, 
                 data_dir: str = "bc_dataset/", 
                 stage: str = "train", 
                 context_length: int = 64):

        super(RolloutEngine, self).__init__()

        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.in_queue = Queue()
        self.result_queue = Queue()
        
        os.makedirs(data_dir, exist_ok=True)

        self.data_dir = data_dir
        self.stage = stage

        self.context_length = context_length
        self.task_to_num_samples = defaultdict(int)

    def wait_for_result(self):

        return self.result_queue.get()[1]

    def remote_get_episode(self, teacher_ratio: float = 1.0):

        self.task = self.train_sampler.next_task()
        
        scene = self.task.unshuffle_env.scene
        index = self.task.unshuffle_env.current_task_spec.metrics.get("index")
        stage = self.task.unshuffle_env.current_task_spec.stage

        num_samples = self.task_to_num_samples[(scene, index, stage)]
        self.task_to_num_samples[(scene, index, stage)] += 1

        if num_samples > 0:

            metadata = self.task.env.controller.step(
                action="GetReachablePositions").metadata

            valid_positions = [dict(
                position=position,
                rotation=dict(x=0, y=rotation, z=0), 
                horizon=horizon,
                standing=standing)
                for position in metadata["actionReturn"]
                for rotation in (0, 90, 180, 270)
                for horizon in (-30, 0, 30, 60)
                for standing in (True, False)]

            sampled_position = valid_positions[
                np.random.choice(len(valid_positions))]
            
            self.task.env.controller.step(
                action="TeleportFull", **sampled_position)

        episode = []

        while not self.task.is_done():

            observation = self.task.get_observations()

            observation["expert_action"] = torch.tensor(
                observation["expert_action"][0], 
                dtype=torch.int64).unsqueeze(0)

            observation["rgb"] = torch.tensor(
                observation["rgb"], 
                dtype=torch.float32).unsqueeze(0)

            observation["unshuffled_rgb"] = torch.tensor(
                observation["unshuffled_rgb"], 
                dtype=torch.float32).unsqueeze(0)

            observation = self.preprocessor\
                .get_observations(observation)

            self.task.step(observation[
                "expert_action"][0].cpu().numpy().item())

            episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

        episode_length = len(episode)

        episode = tree.map_structure(
            lambda *x: torch.cat(x, dim=0), *episode)

        prefix = f"{scene}-{index}-{stage}-{num_samples}"

        for idx in range(0, episode_length, self.context_length):

            chunk = tree.map_structure(lambda x: x[
                idx:idx + self.context_length], episode)

            path = f"{prefix}-{idx // self.context_length}.pt"

            torch.save(chunk, os.path.join(self.data_dir, path))

    def get_episode(self, teacher_ratio: float = 1.0):

        self.in_queue.put(("remote_get_episode", (teacher_ratio,), dict()))

    def _is_alive(self):

        parent = multiprocessing.parent_process()
        return not (parent is None or not parent.is_alive())

    def _check_input_queue(self):

        command, args, kwargs = self.in_queue.get(timeout=2.0)
        fn = RolloutEngine.__dict__[command]

        self.result_queue.put(
            (command, fn(self, *args, **kwargs)))

        del args
        del kwargs

    def _init_thor(self):

        train_args = ExperimentConfig.stagewise_task_sampler_args(
            stage=self.stage, devices=[self.device], 
            total_processes=self.world_size,
            process_ind=self.rank)

        self.train_sampler = ExperimentConfig.make_sampler_fn(
            **train_args, force_cache_reset=False, epochs=float('inf'))

        self.preprocessor = ExperimentConfig\
            .resnet_preprocessor_graph(mode=self.stage)

        preproc = self.preprocessor.preprocessors
        cuda_device = f"cuda:{self.device}"

        preproc["rgb_resnet"].device = cuda_device
        preproc["unshuffled_rgb_resnet"].device = cuda_device

        preproc["rgb_resnet"].resnet.to(cuda_device)
        preproc["unshuffled_rgb_resnet"].resnet.to(cuda_device)
            
        preproc["rgb_resnet"]._resnet = \
            preproc["unshuffled_rgb_resnet"]._resnet

    def run(self):

        self._init_thor()

        while self._is_alive():

            try:
                self._check_input_queue()
            except queue.Empty:
                continue


class BatchRolloutEngine(object):

    def __init__(self, device: int, rank: int, world_size: int, 
                 num_processes: int = 5, **kwargs):

        self.workers = [
            RolloutEngine(
                device, i + rank * num_processes, 
                world_size * num_processes, **kwargs
            ) for i in range(num_processes)
        ]

        for w in self.workers:
            w.start()

    def wait_for_result(self):

        return [w.wait_for_result() 
                for w in self.workers]

    def get_episode(self, teacher_ratio: float = 1.0):

        for w in self.workers:
            w.get_episode(teacher_ratio=teacher_ratio)


if __name__ == "__main__":

    multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="large_bc_dataset")
    parser.add_argument("--samplers-per-gpu", type=int, default=5)

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--context-length", type=int, default=64)

    args = parser.parse_args()
    
    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1
    else:
        
        distributed.init_process_group(backend="nccl")
        print(f'Initialized process {rank} / {world_size}')
        torch.cuda.set_device(rank)

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if rank == 0 or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    engine = BatchRolloutEngine(
        rank, rank, world_size, 
        context_length=args.context_length,
        data_dir=args.data_dir,
        num_processes=args.samplers_per_gpu
    )

    for i in range(args.episodes // 
                   args.samplers_per_gpu):

        engine.get_episode(teacher_ratio=1.0)

    for i in tqdm.trange(args.episodes // 
                         args.samplers_per_gpu, 
                         desc=f'GPU {rank}'):

        engine.wait_for_result()
