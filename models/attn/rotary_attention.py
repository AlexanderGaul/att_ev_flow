import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from perceiverpytorch.perceiver_pytorch.perceiver_io import default, exists

from repositories.rotary_embedding_torch_repo.rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding


class RotaryAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.rotary_embedding = RotaryEmbedding(inner_dim)

        # Add relative position encoding

    def forward(self, x, context = None, mask = None, block_size=1, context_block_size=None,
                rot_offset=0, context_rot_offset=None):
        if context_block_size is None :
            context_block_size = block_size
        if context_rot_offset is None :
            context_rot_offset = rot_offset
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q = self.rotary_embedding.rotate_queries_or_keys(q, block_size=block_size, offset=rot_offset)
        k = self.rotary_embedding.rotate_queries_or_keys(k, block_size=context_block_size, offset=context_rot_offset)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # rotate query and key
        # can have different block sizes
        # coudl compute from block size

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b ...')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)