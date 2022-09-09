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
import json

import queue
import multiprocessing
from multiprocessing import Process, Queue

import glob
import tqdm
import tree
import numpy as np

import argparse


class ExperimentConfig(BaseExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "clip")


class Policy(nn.Module):

    def __init__(self, num_outputs: int, 
                 hidden_size: int = 768, 
                 num_transformer_layers: int = 6, 
                 nhead: int = 12, 
                 dim_head: int = 64, 
                 dim_feedforward: int = 1536, 
                 context_length: int = 64, 
                 dropout: float = 0.):

        super(Policy, self).__init__()

        self.hidden_size = hidden_size
        self.num_transformer_layers = num_transformer_layers
        self.nhead = nhead
        self.dim_head = dim_head
        self.dim_feedforward = dim_feedforward
        self.context_length = context_length
        self.dropout = dropout

        self.visual_attention = nn.Sequential(
            nn.Conv2d(3 * 2048, 32, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 1, 1)
        )
        self.visual_attention.apply(
            simple_conv_and_linear_weights_init
        )

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3 * 2048, hidden_size, 1), 
            nn.ReLU(inplace=True)
        )
        self.visual_encoder.apply(
            simple_conv_and_linear_weights_init
        )

        self.transformer = TransformerXL(
            hidden_size,
            num_transformer_layers=num_transformer_layers,
            nhead=nhead,
            dim_head=dim_head,
            dim_feedforward=dim_feedforward,
            context_length=context_length,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, num_outputs)
        
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def create_memory(self, batch_size: int = 1, 
                      context_length: int = 0, 
                      device: str = 'cpu'):

        shape = (batch_size, context_length, self.hidden_size)

        return [torch.zeros(*shape, dtype=torch.float32, device=device) 
                for layer in range(self.num_transformer_layers)]

    def forward(self, observations, *hidden_states, mask=None):

        cur_img_resnet = observations["rgb_resnet"]
        unshuffled_img_resnet = observations["unshuffled_rgb_resnet"]

        concat_img = torch.cat(
            (
                cur_img_resnet,
                unshuffled_img_resnet,
                cur_img_resnet * unshuffled_img_resnet,
            ),
            dim=-3,
        )

        batch_shape = concat_img.shape[:-3]
        features_shape = concat_img.shape[-3:]

        concat_img_reshaped = concat_img.view(-1, *features_shape)
        attention_probs = torch.softmax(
            self.visual_attention(concat_img_reshaped).view(
                concat_img_reshaped.shape[0], -1
            ),
            dim=-1,
        )

        attention_probs = attention_probs.view(
            concat_img_reshaped.shape[0], 
            1, *concat_img_reshaped.shape[-2:])

        x = (self.visual_encoder(concat_img_reshaped) * 
             attention_probs).sum(-1).sum(-1).view(*batch_shape, -1)

        x, *hidden_states = self.transformer(x, *hidden_states, mask=mask)
        
        return self.linear(self.norm(x)), *hidden_states


def init_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run
        
    distributed.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_dist():
    return distributed.is_available() and distributed.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return distributed.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return distributed.get_rank()


class RolloutEngine(Process):

    def __init__(self, model: Policy, device: int, rank: int, world_size: int, 
                 temperature: float = 0.01, queue_timeout: float = 2.0):

        super(RolloutEngine, self).__init__()

        self.model = model
        self.temperature = temperature

        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.in_queue = Queue()
        self.result_queue = Queue()

        self.queue_timeout = queue_timeout

    def wait_for_result(self):

        return self.result_queue.get()[1]

    def run(self):

        self._init_thor()

        while self._is_alive():

            try:
                self._check_input_queue()
            except queue.Empty:
                continue

    def _is_alive(self):

        parent = multiprocessing.parent_process()
        return not (parent is None or not parent.is_alive())

    def _check_input_queue(self):

        command, args, kwargs = self.in_queue.get(timeout=self.queue_timeout)
        fn = RolloutEngine.__dict__[command]

        self.result_queue.put(
            (command, fn(self, *args, **kwargs)))

    def _init_thor(self):

        train_args = ExperimentConfig.stagewise_task_sampler_args(
            stage="test", devices=[self.device], 
            total_processes=self.world_size,
            process_ind=self.rank)

        self.train_sampler = ExperimentConfig.make_sampler_fn(
            **train_args, force_cache_reset=False, epochs=1)

        self.preprocessor = ExperimentConfig\
            .resnet_preprocessor_graph(mode="test")

        preproc = self.preprocessor.preprocessors
        cuda_device = f"cuda:{self.device}"

        preproc["rgb_resnet"].device = cuda_device
        preproc["unshuffled_rgb_resnet"].device = cuda_device

        preproc["rgb_resnet"].resnet.to(cuda_device)
        preproc["unshuffled_rgb_resnet"].resnet.to(cuda_device)
            
        preproc["rgb_resnet"]._resnet = \
            preproc["unshuffled_rgb_resnet"]._resnet

    def _metrics(self):

        return self.task.metrics()

    def metrics(self):

        self.in_queue.put(("_metrics", (), dict()))

    def _update_model(self, state_dict):

        self.model.load_state_dict(state_dict)

    def update_model(self, state_dict):

        self.in_queue.put(("_update_model", (state_dict,), dict()))

    def _get_episode(self, teacher_ratio: float = 1.0):

        self.task = self.train_sampler.next_task()

        device = f"cuda:{self.device}"
        self.model.eval()
        self.model.to(device)
        
        memory = self.model.create_memory(
            batch_size=1, context_length=0, device=device)

        episode = []

        while not self.task.is_done():

            observation = self.task.get_observations()

            if "expert_action" in observation:
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

            observation["mask"] = \
                torch.ones([1], dtype=torch.float32)

            inputs = tree.map_structure(
                lambda x: x.to(device).unsqueeze(0), observation)
        
            with torch.no_grad():
                logits, *new_memory = \
                    self.model(inputs, *memory)

            memory = tree.map_structure(
                lambda h0, h1: torch.cat((
                    h0, h1), dim=1), memory, new_memory)

            if np.random.uniform() < teacher_ratio:

                act = observation["expert_action"][0]

            else:

                act = Categorical(logits=logits[
                    0, 0] / self.temperature).sample()

            self.task.step(act.cpu().numpy().item())

            episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

        return tree.map_structure(
            lambda *x: torch.cat(x, dim=0), *episode)

    def get_episode(self, teacher_ratio: float = 1.0):

        self.in_queue.put(("_get_episode", (teacher_ratio,), dict()))


class BatchRolloutEngine(object):

    def __init__(self, model: Policy, device: int, rank: int, world_size: int, 
                 temperature: float = 0.01, num_processes: int = 5):

        self.workers = [
            RolloutEngine(
                model, device, i + rank * num_processes, 
                world_size * num_processes, temperature=temperature
            ) for i in range(num_processes)
        ]

        for w in self.workers:
            w.start()

    def update_model(self, state_dict):

        for w in self.workers:
            w.update_model(state_dict)

        for w in self.workers:
            w.wait_for_result()

    def get_episode(self, teacher_ratio: float = 1.0):

        for w in self.workers:
            w.get_episode(teacher_ratio=teacher_ratio)

        return [w.wait_for_result() 
                for w in self.workers]

    def metrics(self):

        for w in self.workers:
            w.metrics()

        return [w.wait_for_result() 
                for w in self.workers]


class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.bool):
            return bool(obj)

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NumpyJSONEncoder,
                     self).default(obj)


if __name__ == "__main__":

    multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="results_debug/transformer-24999.pt")

    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--samplers-per-gpu", type=int, default=5)

    parser.add_argument("--episodes", type=int, default=1000)

    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-transformer-layers", type=int, default=6)

    args = parser.parse_args()
    
    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    model = Policy(
        len(ExperimentConfig.actions()), 
        context_length=args.context_length,
        num_transformer_layers=args.num_transformer_layers
    )

    model.to(device)

    model.load_state_dict(torch.load(
        args.ckpt, map_location=device)["model"])

    rollout_engine = BatchRolloutEngine(
        model, rank, rank, world_size, 
        temperature=args.temperature,
        num_processes=args.samplers_per_gpu
    )

    metrics = []

    for i in tqdm.trange(args.episodes // world_size // 
                         args.samplers_per_gpu,
                         desc=f"GPU {rank}"):

        rollout_engine.get_episode(teacher_ratio=0.0)
        metrics.extend(rollout_engine.metrics())

        print("Prop Fixed Strict:", np.array([
            x["unshuffle/prop_fixed_strict"] for x in metrics]).mean())

        with open(args.ckpt + f".{rank}.test.json", "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyJSONEncoder)
