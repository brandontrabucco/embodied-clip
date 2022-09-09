from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
    Callable,
)

import copy
import math

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr, ConditionalDistr, SequentialDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init

from .baseline_models import PositionalEncoding


class HierarchicalConvRNN(ActorCriticModel[SequentialDistr]):
    """A CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        positional_features=3,
        voxel_features=512,
        num_octaves=8,
        start_octave=-5,
        dropout=0.2,
    ):
        """Initialize a `RearrangeActorCriticSimpleConvRNN` object.

        # Parameters
        action_space : The action space of the agent.
            Should equal `gym.spaces.Discrete(# actions available to the agent)`.
        observation_space : The observation space available to the agent.
        rgb_uuid : The unique id of the RGB image sensor (see `RGBSensor`).
        unshuffled_rgb_uuid : The unique id of the `UnshuffledRGBRearrangeSensor` available to the agent.
        hidden_size : The size of the hidden layer of the RNN.
        num_rnn_layers: The number of hidden layers in the RNN.
        rnn_type : The RNN type, should be "GRU" or "LSTM".
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid

        self.concat_rgb_uuid = "concat_rgb"
        assert self.concat_rgb_uuid not in observation_space

        self.visual_encoder = self._create_visual_encoder()

        self.state_encoder = RNNStateEncoder(
            self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        
        self.critic = LinearCriticHead(self._hidden_size)

        self.pos_encoding = PositionalEncoding(
            num_octaves=num_octaves, start_octave=start_octave)

        positional_features = (
            num_octaves * 2 * positional_features)

        self.obs_to_hidden_w = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size +
                      voxel_features +
                      positional_features, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1 + hidden_size),
        )

        self.obs_to_hidden_u = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size +
                      voxel_features +
                      positional_features, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1 + hidden_size),
        )

        self.actor = nn.Sequential(
            nn.GELU(), 
            nn.Dropout(p=dropout),
            LinearActorHead(
                self._hidden_size, 
                action_space["action"].n
            )
        )
            
        self.train()

    def _create_visual_encoder(self) -> nn.Module:
        """Create the visual encoder for the model."""
        img_space: gym.spaces.Box = self.observation_space[self.rgb_uuid]
        return SimpleCNN(
            observation_space=gym.spaces.Dict(
                {
                    self.concat_rgb_uuid: gym.spaces.Box(
                        low=np.tile(img_space.low, (1, 1, 2)),
                        high=np.tile(img_space.high, (1, 1, 2)),
                        shape=img_space.shape[:2] + (img_space.shape[2] * 2,),
                    )
                }
            ),
            output_size=self._hidden_size,
            rgb_uuid=self.concat_rgb_uuid,
            depth_uuid=None,
        )

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img = observations[self.rgb_uuid]
        unshuffled_img = observations[self.unshuffled_rgb_uuid]
        concat_img = torch.cat((cur_img, unshuffled_img), dim=-1)

        x = self.visual_encoder({self.concat_rgb_uuid: concat_img})
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ## START

        voxel_positions_w = observations["map"]["voxel_positions_w"]
        voxel_features_w = observations["map"]["voxel_features_w"]

        voxel_positions_u = observations["map"]["voxel_positions_u"]
        voxel_features_u = observations["map"]["voxel_features_u"]

        logits_w, state_w = self.obs_to_hidden_w(torch.cat((
            x.view(x.shape[0], x.shape[1], 1, x.shape[2]).expand(
                x.shape[0], x.shape[1], voxel_features_w.shape[2], x.shape[2]),
            voxel_features_w, self.pos_encoding(voxel_positions_w)
        ), dim=-1)).split([1, x.shape[2]], dim=3)

        logits_u, state_u = self.obs_to_hidden_u(torch.cat((
            x.view(x.shape[0], x.shape[1], 1, x.shape[2]).expand(
                x.shape[0], x.shape[1], voxel_features_u.shape[2], x.shape[2]),
            voxel_features_u, self.pos_encoding(voxel_positions_u)
        ), dim=-1)).split([1, x.shape[2]], dim=3)

        attention_distr = CategoricalDistr(
            logits=torch.cat((logits_w, logits_u), dim=2).squeeze(3)
        )

        def policy_distr(*args, **kwargs):

            assert "attention" in kwargs

            attention_idx = kwargs["attention"]

            attention_idx = attention_idx.view(x.shape[0], x.shape[1], 1, 1)
            attention_idx = attention_idx.expand(
                x.shape[0], x.shape[1], 1, x.shape[2])

            emb = torch.cat((state_w, state_u), dim=2)

            return self.actor(torch.gather(
                emb, 2, attention_idx).squeeze(2))

        attention_distr = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=attention_distr,
            action_group_name="attention",
        )

        policy_distr = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=policy_distr,
            action_group_name="action",
        )

        distr = SequentialDistr(attention_distr, policy_distr)

        ## END

        ac_output = ActorCriticOutput(
            distributions=distr, values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PretrainedHierarchicalConvRNN(HierarchicalConvRNN):

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        num_rnn_layers=1,
        rnn_type="GRU",
        hidden_size=512,
        positional_features=3,
        voxel_features=512,
        num_octaves=8,
        start_octave=-5,
        dropout=0.2,
    ):
        """A CNN->RNN rearrangement model that expects ResNet features instead
        of RGB images.

        Nearly identical to `RearrangeActorCriticSimpleConvRNN` but
        `rgb_uuid` should now be the unique id of the ResNetPreprocessor
        used to featurize RGB images using a pretrained ResNet before
        they're passed to this model.
        """

        self.visual_attention: Optional[nn.Module] = None
        super().__init__(**prepare_locals_for_super(locals()))
        
    def _create_visual_encoder(self) -> nn.Module:
        a, b = [
            self.observation_space[k].shape[0]
            for k in [self.rgb_uuid, self.unshuffled_rgb_uuid]
        ]
        assert a == b
        self.visual_attention = nn.Sequential(
            nn.Conv2d(3 * a, 32, 1,), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1,),
        )
        visual_encoder = nn.Sequential(
            nn.Conv2d(3 * a, self._hidden_size, 1,), nn.ReLU(inplace=True),
        )
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        visual_encoder.apply(simple_conv_and_linear_weights_init)

        return visual_encoder

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img_resnet = observations[self.rgb_uuid]
        unshuffled_img_resnet = observations[self.unshuffled_rgb_uuid]
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

        x, rnn_hidden_states = self.state_encoder(
            x, memory.tensor("rnn"), masks)

        ## START

        voxel_positions_w = observations["map"]["voxel_positions_w"]
        voxel_features_w = observations["map"]["voxel_features_w"]

        voxel_positions_u = observations["map"]["voxel_positions_u"]
        voxel_features_u = observations["map"]["voxel_features_u"]

        logits_w, state_w = self.obs_to_hidden_w(torch.cat((
            x.view(x.shape[0], x.shape[1], 1, x.shape[2]).expand(
                x.shape[0], x.shape[1], voxel_features_w.shape[2], x.shape[2]),
            voxel_features_w, self.pos_encoding(voxel_positions_w)
        ), dim=-1)).split([1, x.shape[2]], dim=3)

        logits_u, state_u = self.obs_to_hidden_u(torch.cat((
            x.view(x.shape[0], x.shape[1], 1, x.shape[2]).expand(
                x.shape[0], x.shape[1], voxel_features_u.shape[2], x.shape[2]),
            voxel_features_u, self.pos_encoding(voxel_positions_u)
        ), dim=-1)).split([1, x.shape[2]], dim=3)

        attention_distr = CategoricalDistr(
            logits=torch.cat((logits_w, logits_u), dim=2).squeeze(3)
        )

        def policy_distr(*args, **kwargs):

            assert "attention" in kwargs

            attention_idx = kwargs["attention"]

            attention_idx = attention_idx.view(x.shape[0], x.shape[1], 1, 1)
            attention_idx = attention_idx.expand(
                x.shape[0], x.shape[1], 1, x.shape[2])

            emb = torch.cat((state_w, state_u), dim=2)

            return self.actor(torch.gather(
                emb, 2, attention_idx).squeeze(2))

        attention_distr = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=attention_distr,
            action_group_name="attention",
        )

        policy_distr = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=policy_distr,
            action_group_name="action",
        )

        distr = SequentialDistr(attention_distr, policy_distr)

        ## END

        ac_output = ActorCriticOutput(
            distributions=distr, values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)