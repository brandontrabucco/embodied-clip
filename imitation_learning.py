from baseline_configs.one_phase.one_phase_rgb_il_base \
    import OnePhaseRGBILBaseExperimentConfig as BaseExperimentConfig

from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from rearrange.attention_models import TransformerXL
import datagen.datagen_utils as datagen_utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence as pad

import tqdm
import tree
import numpy as np


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
            nn.Conv2d(32, 1, 1,)
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
        batch_shape, features_shape = concat_img.shape[:-3], concat_img.shape[-3:]
        concat_img_reshaped = concat_img.view(-1, *features_shape)
        attention_probs = torch.softmax(
            self.visual_attention(concat_img_reshaped).view(
                concat_img_reshaped.shape[0], -1
            ),
            dim=-1,
        ).view(concat_img_reshaped.shape[0], 1, *concat_img_reshaped.shape[-2:])
        x = (
            (self.visual_encoder(concat_img_reshaped) * attention_probs)
            .sum(-1)
            .sum(-1)
        )
        x = x.view(*batch_shape, -1)

        x, *hidden_states = self.transformer(x, *hidden_states, mask=mask)

        return self.linear(self.norm(x)), *hidden_states


if __name__ == "__main__":

    allowed_rearrange_inds_subset = list(range(10))
    allowed_scenes = datagen_utils.get_scenes("train")[:1]

    train_args = ExperimentConfig.stagewise_task_sampler_args(
        stage="train", devices=[0], process_ind=0, total_processes=1,
        allowed_rearrange_inds_subset=allowed_rearrange_inds_subset, 
        allowed_scenes=allowed_scenes)

    train_sampler = ExperimentConfig.make_sampler_fn(
        **train_args, force_cache_reset=False, epochs=float('inf'))

    clip_preprocessor = ExperimentConfig\
        .resnet_preprocessor_graph(mode="train")

    model = Policy(len(ExperimentConfig.actions())).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    episodes = []
    training_steps = 0

    random_action_eps = 0.0

    for iteration in tqdm.tqdm(list(range(10))):

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
            observation["indicator"] = torch.ones([1], dtype=torch.float32)

            current_episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

            act = task.query_expert()[0]

            task.step(act)

        episodes.append(tree.map_structure(
            lambda *x: torch.cat(x, dim=0), *current_episode))

    for iteration in range(10000):

        task = train_sampler.next_task()

        current_episode = []

        hidden_states = [
            torch.zeros(1, 0, 768, dtype=torch.float32).cuda()
        ] * 6

        model.eval()

        while not task.is_done() and iteration > 0:

            observation = task.get_observations()

            observation["expert_action"] = torch.tensor(
                observation["expert_action"], dtype=torch.int64).unsqueeze(0)
            observation["rgb"] = torch.tensor(
                observation["rgb"], dtype=torch.float32).unsqueeze(0)
            observation["unshuffled_rgb"] = torch.tensor(
                observation["unshuffled_rgb"], dtype=torch.float32).unsqueeze(0)

            observation = clip_preprocessor.get_observations(observation)
            observation["indicator"] = torch.ones([1], dtype=torch.float32)

            with torch.no_grad():

                logits, *new_hidden_states = model(tree.map_structure(
                    lambda x: x.cuda().unsqueeze(0), observation), *hidden_states)

            hidden_states = tree.map_structure(lambda h0, h1: torch.cat(
                (h0, h1), dim=1), hidden_states, new_hidden_states)

            act = logits.argmax(dim=2).view(1).cpu().numpy().item()

            task.step(act)

            current_episode.append(tree.map_structure(
                lambda x: x.cpu(), observation))

        if iteration > 0:

            episodes.append(tree.map_structure(
                lambda *x: torch.cat(x, dim=0), *current_episode))

            metrics = task.metrics()
            print(metrics)

        model.train()

        chunks = [
            (i, j * 64, min((j + 1) * 64, episodes[i]["expert_action"].shape[0])) 
            for i in range(len(episodes))
            for j in range(int(np.ceil(episodes[i]["expert_action"].shape[0] / 64))) 
        ]

        chunks = np.array(chunks)
        probabilities = (chunks[:, 1] == 0).astype(np.float32)

        should_write_hidden_state = chunks[:, 1] != 0

        hidden_states = [torch.zeros(chunks.shape[0], 64, 768, dtype=torch.float32) for layer in range(6)]

        for repeat in range(200):

            batch_chunk_ids = np.random.choice(probabilities.size, size=int(probabilities.sum()), replace=False, 
                                               p=probabilities / probabilities.sum())

            batch_chunk = chunks[batch_chunk_ids]

            batch = [tree.map_structure(lambda x: x[start: end], episodes[idx]) for idx, start, end in batch_chunk]
            batch = tree.map_structure(lambda *x: pad(x, batch_first=True), *batch)

            batch_hidden_states = [layer[batch_chunk_ids].cuda() for layer in hidden_states]

            training_steps += 1

            batch = tree.map_structure(lambda x: x.cuda(), batch)
        
            should_write_next_hidden_state = should_write_hidden_state[(batch_chunk_ids + 1) % probabilities.size]
            should_write_next_hidden_state = torch.tensor(
                should_write_next_hidden_state).unsqueeze(1).unsqueeze(1)

            mask = torch.tensor(should_write_hidden_state[batch_chunk_ids])
            mask = torch.logical_not(mask).unsqueeze(1).unsqueeze(1)
            mask = torch.where(mask, torch.cat((
                float("-inf") * torch.ones(1, 1, 64), 
                torch.zeros(1, 1, batch["expert_action"].shape[1])), dim=2), 
                torch.zeros(1, 1, 64 + batch["expert_action"].shape[1]))

            logits, *batch_hidden_states = model(batch, *batch_hidden_states, mask=mask.cuda())

            if should_write_next_hidden_state.any():

                for layer, layer_batch in zip(hidden_states, batch_hidden_states):

                    layer[(batch_chunk_ids + 1) % probabilities.size] = torch.where(
                        should_write_next_hidden_state, layer_batch.detach().cpu(), layer[(batch_chunk_ids + 1) % probabilities.size]
                    )

            probabilities[(batch_chunk_ids + 1) % probabilities.size] = 1.0

            print(logits.argmax(dim=2)[0].view(-1).cpu().numpy(), 
                  batch["expert_action"][0, :, 0].view(-1).cpu().numpy())
                    
            loss = F.cross_entropy(
                logits.permute(0, 2, 1), 
                batch["expert_action"][:, :, 0], 
                reduction='none'
            )

            loss = (loss * batch["indicator"]).sum() / batch["indicator"].sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss = loss.detach().cpu().numpy().item()

            print(f"Training Step: {training_steps} Loss {loss}")

