from baseline_configs.one_phase.one_phase_rgb_il_base \
    import OnePhaseRGBILBaseExperimentConfig as ExperimentConfig

from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from allenact.utils.misc_utils import partition_sequence
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


if __name__ == "__main__":

    multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type=str, default="results_large_dataset")
    parser.add_argument("--data-dir", type=str, default="bc_dataset")
    parser.add_argument("--save-period", type=int, default=5000)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-per-iteration", type=int, default=0)
    
    parser.add_argument("--temperature", type=float, default=.001)
    parser.add_argument("--samplers-per-gpu", type=int, default=5)

    parser.add_argument("--episodes-per-iteration", type=int, default=0)
    parser.add_argument("--episode-capacity", type=int, default=5000)

    parser.add_argument("--teacher-iterations", type=int, default=0)
    parser.add_argument("--decay-iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)

    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-transformer-layers", type=int, default=6)

    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

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

    device = torch.device(f"cuda:{rank}")

    unwrapped_model = model = Policy(
        len(ExperimentConfig.actions()), 
        context_length=args.context_length,
        num_transformer_layers=args.num_transformer_layers
    )

    model.to(device)

    start_iteration = 0
      
    for ckpt in glob.glob(os.path.join(args.logdir, f"*.pt")):

        ckpt = torch.load(ckpt, map_location=device)

        if ckpt["iteration"] > start_iteration:
            start_iteration = ckpt["iteration"] + 1
            unwrapped_model.load_state_dict(ckpt["model"])

    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    training_steps = start_iteration * args.batch_per_iteration

    scenes = list(
        sorted(
            partition_sequence(
                seq=datagen_utils.get_scenes("train"), 
                parts=world_size
            )[rank]
        )
    )

    dataset_files = []
    for scene in scenes:
        dataset_files.extend(
            list(
                sorted(
                    list(
                        glob.glob(os.path.join(args.data_dir, f"{scene}-*-train-*-*.pt"))
                    )
                )
            )
        )

    hidden_states = unwrapped_model.create_memory(
        batch_size=len(dataset_files), 
        context_length=args.context_length)

    not_first_chunk = np.array([int(x[:-3].split("-")[-1]) != 0 for x in dataset_files])
    sampler_probabilities = 1.0 - not_first_chunk.astype(np.float32)

    def add_mask(x):
        return dict(mask=torch.ones(x["expert_action"].shape[0], 
                    device=x["expert_action"].device, dtype=torch.float32), **x)

    for iteration in range(start_iteration, 5000):

        model.train()

        batch_chunk_ids = np.random.choice(
            sampler_probabilities.size, size=args.batch_size, 
            replace=args.batch_size > sampler_probabilities.sum(), 
            p=sampler_probabilities / sampler_probabilities.sum()
        )

        batch = [
            add_mask(torch.load(dataset_files[idx]))
            for idx in batch_chunk_ids
        ]

        batch = tree.map_structure(
            lambda *x: pad(x, batch_first=True), *batch)

        batch_hidden_states = [
            layer[batch_chunk_ids].to(device) 
            for layer in hidden_states
        ]

        training_steps += 1

        batch = tree.map_structure(lambda x: x.to(device), batch)

        first_chunk = torch.logical_not(torch.tensor(
            not_first_chunk[batch_chunk_ids])
            ).unsqueeze(1).unsqueeze(1)

        first_chunk_mask = torch.cat((
            torch.full((1, 1, args.context_length), float("-inf")), 
            torch.zeros(1, 1, batch["mask"].shape[1])), dim=2)

        other_chunk_mask = torch.zeros(
            1, 1, args.context_length + batch["mask"].shape[1])

        chunk_mask = torch.where(
            first_chunk, first_chunk_mask, other_chunk_mask)

        logits, *batch_hidden_states = model(
            batch, *batch_hidden_states, mask=chunk_mask.to(device))

        batch_next_chunk_ids = (
            batch_chunk_ids + 1) % sampler_probabilities.size

        sampler_probabilities[batch_next_chunk_ids] = 1.0

        next_not_first_chunk = not_first_chunk[batch_next_chunk_ids]
        next_not_first_chunk = torch.tensor(
            next_not_first_chunk).unsqueeze(1).unsqueeze(1)

        if next_not_first_chunk.any():

            for layer, layer_batch in zip(
                    hidden_states, batch_hidden_states):

                layer[batch_next_chunk_ids] = torch.where(
                    next_not_first_chunk, 
                    layer_batch.detach().cpu(), 
                    layer[batch_next_chunk_ids]
                )

        logits = logits.permute(0, 2, 1)

        loss = F.cross_entropy(logits, batch[
            "expert_action"], reduction='none')

        loss = loss * batch["mask"]
        loss = loss.sum() / batch["mask"].sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss = loss.detach().cpu().numpy().item()
        print(f"Training Step: {training_steps} Loss {loss}")

        if (iteration + 1) % args.save_period == 0 and rank == 0:

            model_path = os.path.join(
                args.logdir, f"transformer-{iteration}.pt")

            torch.save(dict(model=unwrapped_model.state_dict(), 
                            iteration=iteration), model_path)
