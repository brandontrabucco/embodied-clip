import torch
import numpy as np

from torch import nn
from einops import rearrange


class PreNorm(nn.Module):

    def __init__(self, dim, fn):

        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, dim_feedforward: int = 1536, dropout = 0.):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim: int, nhead: int = 8, dim_head: int = 64, 
                 dropout: float = 0., context_length: int = 64):

        super().__init__()

        inner_dim = dim_head *  nhead
        num_octaves = int(np.ceil(np.log2(context_length))) + 1

        self.nhead = nhead
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_rel = nn.Linear(2 * num_octaves, inner_dim, bias = False)

        self.attend = nn.Sequential(nn.Softmax(dim = -1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.u_embedding = nn.Parameter(torch.randn(1, 1, 1, dim_head))
        self.v_embedding = nn.Parameter(torch.randn(1, 1, 1, dim_head))

        self.context_length = context_length

        R_embedding = torch.arange(context_length)

        self.register_buffer('R_embedding', Attention.positional_encoding(
            R_embedding.view(context_length, 1), 1 - num_octaves, num_octaves))

    def get_shifts(self, memory_dim, sequence_dim):

        coords = torch.arange(start = memory_dim + sequence_dim - 1, 
                              end = -1, step = -1) - (sequence_dim - 1)

        shifts = coords.unsqueeze(0) + torch.arange(sequence_dim).unsqueeze(1)
        out_of_bounds = torch.logical_or(shifts < 0, shifts > self.context_length - 1)

        attention_bias = torch.where(out_of_bounds, float('-inf'), 0.0)
        return shifts.clamp(min = 0, max = self.context_length - 1), attention_bias

    @staticmethod
    def positional_encoding(coords, start_octave, num_octaves):

        coords_shape = coords.shape

        octaves = torch.arange(start_octave, start_octave + num_octaves)
        octaves = octaves.float().to(coords.device)

        multipliers = (2 ** octaves) * np.pi

        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = (coords * multipliers).view(
            *coords_shape[:-1], int(coords_shape[-1]) * num_octaves)
 
        return torch.cat((torch.sin(scaled_coords),
                          torch.cos(scaled_coords)), dim = -1)

    def forward(self, x, memory, mask = None):

        memory_dim, sequence_dim = memory.shape[1], x.shape[1]
        shifts, attention_bias = self.get_shifts(memory_dim, sequence_dim)
        
        qkv = self.to_qkv(torch.cat([memory, x], dim = 1)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.nhead), qkv)
        
        q = q[:, :, memory_dim:]
        k_transpose = k.transpose(-1, -2)

        kr = rearrange(self.to_rel(self.R_embedding)[shifts], 
                       'n m (h d) -> h n m d', h = self.nhead)

        term_a = torch.matmul(q, k_transpose)
        term_b = torch.einsum("bhnd,hnmd->bhnm", q, kr)
        term_c = torch.matmul(self.u_embedding.expand(q.shape), k_transpose)
        term_d = torch.einsum("bhnd,hnmd->bhnm", self.v_embedding.expand(q.shape), kr)

        dots = (term_a + term_b + term_c + term_d) * self.scale
        dots = dots + attention_bias.to(x.device)

        if mask is not None:
            dots = dots + mask.unsqueeze(1).unsqueeze(1)
        
        result = torch.matmul(self.attend(dots), v)
        return self.to_out(rearrange(result, 'b h n d -> b n (h d)'))


class TransformerXL(nn.Module):

    def __init__(self, dim, num_transformer_layers: int = 6, nhead: int = 12, dim_head: int = 64, 
                 dim_feedforward: int = 1536, context_length: int = 64, dropout: float = 0.):

        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(num_transformer_layers):

            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(
                    dim, nhead = nhead, dim_head = dim_head, context_length = context_length, dropout = dropout
                )),
                PreNorm(dim, FeedForward(
                    dim, dim_feedforward = dim_feedforward, dropout = dropout
                ))
            ]))

    def forward(self, x, *memory, mask = None):

        new_memory = []
        for m, (attention, ff) in zip(memory, self.layers):
            new_memory.append(x)

            x = x + attention(x, m, mask = mask)
            x = x + ff(x)

        return x, *new_memory