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

import os
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-initial", type=int, default=20)
    parser.add_argument("--batch-per-episode", type=int, default=3)

    parser.add_argument("--episode-capacity", type=int, default=500)
    parser.add_argument("--initial-episodes", type=int, default=500)

    parser.add_argument("--teacher-episodes", type=int, default=500)
    parser.add_argument("--teacher-decay", type=int, default=500)
    parser.add_argument("--model-episodes", type=int, default=10000)

    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-transformer-layers", type=int, default=6)

    args = parser.parse_args()
    
    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}")

    train_args = ExperimentConfig.stagewise_task_sampler_args(
        stage="train", devices=[rank], process_ind=rank, total_processes=world_size)

    train_sampler = ExperimentConfig.make_sampler_fn(
        **train_args, force_cache_reset=False, epochs=float('inf'))

    clip_preprocessor = ExperimentConfig\
        .resnet_preprocessor_graph(mode="train")

    episodes = []

    for iteration in tqdm.tqdm(list(range(args.initial_episodes))):

        task = train_sampler.next_task()

        current_episode = []

        while not task.is_done():

            observation = task.get_observations()

            observation["expert_action"] = torch.tensor(
                observation["expert_action"], dtype=torch.int64).unsqueeze(0)
            observation["rgb"] = torch.tensor(
                observation["rgb"], dtype=torch.float32).unsqueeze(0)
            observation["unshuffled_rgb"] = torch.tensor(
                observation["unshuffled_rgb"], dtype=torch.float32).unsqueeze(0)

            observation = clip_preprocessor.get_observations(observation)
            observation["mask"] = torch.ones([1], dtype=torch.float32)

            current_episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

            act = task.query_expert()[0]

            task.step(act)

        episodes.append(tree.map_structure(
            lambda *x: torch.cat(x, dim=0), *current_episode))

    model = Policy(
        len(ExperimentConfig.actions()), 
        context_length=args.context_length,
        num_transformer_layers=args.num_transformer_layers
    )

    model.to(device)

    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    training_steps = 0

    for iteration in range(args.teacher_episodes + 
                           args.teacher_decay + 
                           args.model_episodes):

        task = train_sampler.next_task()

        hidden_states = [torch.zeros(
            1, 0, 768, dtype=torch.float32).to(device)
            for layer in range(args.num_transformer_layers)]

        model.eval()

        current_episode = []

        while not task.is_done() and iteration > 0:

            observation = task.get_observations()

            observation["expert_action"] = torch.tensor(
                observation["expert_action"], dtype=torch.int64).unsqueeze(0)
            observation["rgb"] = torch.tensor(
                observation["rgb"], dtype=torch.float32).unsqueeze(0)
            observation["unshuffled_rgb"] = torch.tensor(
                observation["unshuffled_rgb"], dtype=torch.float32).unsqueeze(0)

            observation = clip_preprocessor.get_observations(observation)
            observation["mask"] = torch.ones([1], dtype=torch.float32)

            with torch.no_grad():

                logits, *new_hidden_states = model(tree.map_structure(
                    lambda x: x.to(device).unsqueeze(0), observation), *hidden_states)

            hidden_states = tree.map_structure(lambda h0, h1: torch.cat(
                (h0, h1), dim=1), hidden_states, new_hidden_states)

            act = logits.argmax(dim=2).view(1).cpu().numpy().item()

            if np.random.uniform() < (1.0 - (
                    iteration - 1 - 
                    args.teacher_episodes) / args.teacher_decay):

                act = task.query_expert()[0]

            task.step(act)

            current_episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

        if iteration > 0:

            episodes.append(tree.map_structure(
                lambda *x: torch.cat(x, dim=0), *current_episode))

            metrics = task.metrics()
            print(f"Iteration {iteration} Metrics:", metrics)

        if len(episodes) > args.episode_capacity:
            episodes.pop(0)

        model.train()

        chunks = np.array([
            [i, j * args.context_length] 
            for i in range(len(episodes)) 
            for j in range(int(np.ceil(episodes[i]["mask"].shape[0] / args.context_length)))
        ])

        not_first_chunk = chunks[:, 1] != 0
        sampler_probabilities = (chunks[:, 1] == 0).astype(np.float32)

        hidden_states = [torch.zeros(
            chunks.shape[0], args.context_length, 768, 
            dtype=torch.float32) for layer in range(args.num_transformer_layers)]

        for repeat in range(args.batch_per_episode 
                            if iteration > 0 else 
                            args.batch_initial):

            batch_chunk_ids = np.random.choice(
                sampler_probabilities.size, size=args.batch_size, 
                replace=args.batch_size > sampler_probabilities.sum(), 
                p=sampler_probabilities / sampler_probabilities.sum()
            )

            batch_chunk = chunks[batch_chunk_ids]

            batch = [
                tree.map_structure(lambda x: x[
                    start:start + args.context_length], episodes[idx]) 
                for idx, start in batch_chunk
            ]

            batch = tree.map_structure(
                lambda *x: pad(x, batch_first=True), *batch)

            batch_hidden_states = [
                layer[batch_chunk_ids].to(device) 
                for layer in hidden_states
            ]

            training_steps += 1

            batch = tree.map_structure(lambda x: x.to(device), batch)

            first_chunk = torch.tensor(not_first_chunk[batch_chunk_ids])
            first_chunk = torch.logical_not(first_chunk)
            first_chunk = first_chunk.unsqueeze(1).unsqueeze(1)

            first_chunk_mask = torch.cat((
                torch.full((1, 1, args.context_length), float("-inf")), 
                torch.zeros(1, 1, batch["mask"].shape[1])), dim=2)

            other_chunk_mask = torch.zeros(
                1, 1, args.context_length + batch["mask"].shape[1])

            chunk_mask = torch.where(
                first_chunk, first_chunk_mask, other_chunk_mask)

            logits, *batch_hidden_states = model(
                batch, *batch_hidden_states, mask=chunk_mask.to(device))

            batch_next_chunk_ids = (batch_chunk_ids + 1) % sampler_probabilities.size
            next_not_first_chunk = not_first_chunk[batch_next_chunk_ids]
            next_not_first_chunk = torch.tensor(
                next_not_first_chunk).unsqueeze(1).unsqueeze(1)

            if next_not_first_chunk.any():

                for layer, layer_batch in zip(hidden_states, batch_hidden_states):

                    layer[batch_next_chunk_ids] = torch.where(
                        next_not_first_chunk, 
                        layer_batch.detach().cpu(), 
                        layer[batch_next_chunk_ids]
                    )

            sampler_probabilities[batch_next_chunk_ids] = 1.0

            logits = logits.permute(0, 2, 1)
            labels = batch["expert_action"][:, :, 0]

            loss = F.cross_entropy(logits, labels, reduction='none')
            loss = loss * batch["mask"]
            loss = loss.sum() / batch["mask"].sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss = loss.detach().cpu().numpy().item()
            print(f"Training Step: {training_steps} Loss {loss}")

        if (iteration + 1) % 100 == 0 and rank == 0:

            torch.save(model.state_dict(), f"model-{iteration}.pt")
