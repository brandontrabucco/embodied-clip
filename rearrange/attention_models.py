import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., max_len=1024):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_kr = nn.Linear(dim, inner_dim, bias = False)

        self.R_embedding = nn.Parameter(torch.randn(2 * max_len, dim))
        self.u_embedding = nn.Parameter(torch.randn(1, 1, 1, dim_head))
        self.v_embedding = nn.Parameter(torch.randn(1, 1, 1, dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.register_buffer('shifts', ((
            torch.arange(2 * max_len).view(1, 2 * max_len) -
            torch.arange(2 * max_len).view(2 * max_len, 1)
        ) % (2 * max_len))[:max_len, :max_len])

    def get_attention_bias(self, memory_dim, sequence_dim, device):

        autoregressive_mask = nn.Transformer.generate_square_subsequent_mask(sequence_dim)
        memory_mask = nn.Transformer.generate_square_subsequent_mask(memory_dim)

        autoregressive_mask = autoregressive_mask.to(device)
        memory_mask = memory_mask.to(device)

        memory_mask = memory_mask.transpose(0, 1)[:sequence_len]

        autoregressive_mask = autoregressive_mask.view(1, 1, sequence_dim, sequence_dim)
        memory_mask = memory_mask.view(1, 1, sequence_dim, memory_dim)
        
        return torch.cat([memory_mask, autoregressive_mask], dim=3)

    def forward(self, x, h = None, attn_bias = None):
        qkv = self.to_qkv(torch.cat([h, x], dim = 1)).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        shifts = self.shifts[h.shape[1]:h.shape[1] + x.shape[1], :h.shape[1] + x.shape[1]]

        kr = self.to_kr(self.R_embedding)[shifts]
        kr = rearrange(kr, 'n m (h d) -> h n m d', h = self.heads)
        
        q = q[:, :, h.shape[1]:]
        k_transpose = k.transpose(-1, -2)

        term_a = torch.matmul(q, k_transpose)
        term_b = torch.einsum("bhnd,hnmd->bhnm", q, kr)
        term_c = torch.matmul(self.u_embedding.expand(q.shape), k_transpose)
        term_d = torch.einsum("bhnd,hnmd->bhnm", self.v_embedding.expand(q.shape), kr)

        dots = (term_a + term_b + term_c + term_d) * self.scale
        dots = dots + self.get_attention_bias(h.shape[1], x.shape[1], x.device)

        if attn_bias is not None:
            dots = dots + attn_bias.unsqueeze(1)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerXL(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, h, attn_bias = None):
        new_h = []
        for i, (attn, ff) in enumerate(self.layers):
            new_h.append(x)
            x = attn(x, h = h[:, :, i], attn_bias = attn_bias) + x
            x = ff(x) + x
        h = torch.stack(new_h, dim=2)
        return x, h