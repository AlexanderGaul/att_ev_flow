from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        latent_init_normal = True,
        cross_heads = 1,
        cross_layers = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        decoder_ff_norm = True,
        to_logits_prenorm = False,
        transformer_encoder = False,
        no_query_return = 'latents',
        latent_long_range_query = False,
        decoder_query_refine = False,
        low_level_latents = False
    ):
        super().__init__()
        if not transformer_encoder :
            if latent_init_normal :
                self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
            else :
                self.latents = nn.Parameter(torch.rand(num_latents, latent_dim) * 2 - 1)
        else :
            self.latents = None
            assert dim == latent_dim

        self.cross_attend_layers = nn.ModuleList([])
        for i in range(cross_layers) :
            self.cross_attend_layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ]))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)

        if decoder_query_refine :
            self.decoder_cross_attn_refine = PreNorm(queries_dim,
                                                     Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head),
                                                     context_dim = latent_dim)
        else :
            self.decoder_cross_attn_refine = None

        self.latent_long_range_query = latent_long_range_query
        self.low_level_latents = low_level_latents

        if self.latent_long_range_query or self.low_level_latents:
            self.decoder_cross_attn_long_range = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)

        if self.low_level_latents :
            if latent_init_normal:
                self.more_latents = nn.Parameter(torch.randn(num_latents, latent_dim))
            else:
                self.more_latents = nn.Parameter(torch.rand(num_latents, latent_dim) * 2 - 1)
            self.more_cross = nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ])
            self.more_self = nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ])

        if decoder_ff :
            if decoder_ff_norm :
                self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim))
            else :
                self.decoder_ff = FeedForward(queries_dim)
        else :
            self.decoder_ff = None

        if not to_logits_prenorm :
            self.to_logits = nn.Linear(queries_dim, logits_dim, bias=True) if exists(logits_dim) else nn.Identity()
        else :
            self.to_logits = PreNorm(queries_dim, nn.Linear(queries_dim, logits_dim, bias=True)) if exists(logits_dim) else nn.Identity()

        self.no_query_return = no_query_return
        if self.no_query_return != 'latents' :
            self.latents_to_logits = nn.Linear(latent_dim, logits_dim, bias=True)

    def forward(
        self,
        data,
        mask = None,
        queries = None
    ):
        if type(data) is list:
            b = len(data)
            device = data[0].device
            #if b == 1 :
            #    data = data[0].unsqueeze(0)
        else:
            if data.ndim == 2 :
                data = data.unsqueeze(0)
            b, *_, device = *data.shape, data.device

        if self.latents is not None :
            x = repeat(self.latents, 'n d -> b n d', b = b)
        else :
            x = data

        # TODO: create batch dimensions
        if type(x) is list :
            x = [x_i.unsqueeze(0) if x_i.ndim == 2 else x_i for x_i in x]
        if type(data) is list :
            data = [d_i.unsqueeze(0) if d_i.ndim == 2 else d_i for d_i in data]
        
        def apply_cross(cross_attn, cross_ff, x, data, mask) :
            if type(x) is list :
                x = [cross_attn(x[i],
                                context = data[i],
                                mask = mask) + x[i]
                     for i in range(b)]
                x = [cross_ff(x_i) for x_i in x]
            elif type(data) is list :
                x = torch.cat([cross_attn(x[[i], :],
                                          context = data[i],
                                          mask = mask) + x[[i], :]
                               for i in range(b)], dim=0)
                x = cross_ff(x) + x
            else :
                x = cross_attn(x, context = data, mask = mask) + x
                x = cross_ff(x) + x
            return x

        for cross_attn, cross_ff in self.cross_attend_layers :
            x = apply_cross(cross_attn, cross_ff, x, data, mask)

        x_after_cross = x

        if type(x) is list :
            for self_attn, self_ff in self.layers :
                x = [self_attn(x_i) + x_i for x_i in x]
                x = [self_ff(x_i) + x_i for x_i in x]

            if not exists(queries) :
                if self.no_query_return == 'latents' :
                    return x
                elif self.no_query_return == 'logits_mean' :
                    x_means = torch.cat([x_i.mean(dim=1, keepdim=True)
                                         #if x_i.shape[1] > 0 else
                                         #torch.zeros((x_i.shape[0], 1, *x_i.shape[2:]), device=x_i.device)
                                         for x_i in x],
                                        dim=0)
                    # TOOD: handle zero length inputs
                    return self.latents_to_logits(x_means)
                elif self.no_query_return == 'logits_all' :
                    # TODO: these might have batch dimension of 1, do we want that??
                    return [self.latents_to_logits(x_i) for x_i in x]

        else :
            # layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

            if not exists(queries):
                if self.no_query_return == 'latents' :
                    return x
                elif self.no_query_return == 'logits_mean' :
                    return self.latents_to_logits(x.mean(dim=1, keepdim=True))
                elif self.no_query_return == 'logits_all' :
                    return self.latents_to_logits(x)

        # make sure queries contains batch dimension
        if type(queries) is list:
            pass
        elif queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        if type(x) is list :
            latents = [self.decoder_cross_attn(queries[i].unsqueeze(0),
                                               context=x[i])
                       for i in range(b)]
        elif type(queries) is list:
            # TODO: batch/concatenate? for feed forward
            # NOTE: concat at dim 1 to keep one batch
            # TODO: is this a good idea, what about batch norm
            latents = [self.decoder_cross_attn(queries[i].unsqueeze(0),
                                               context = x[[i], :])
                       for i in range(b)]
        else :
            latents = self.decoder_cross_attn(queries, context = x)

        if exists(self.decoder_cross_attn_refine) :
            for _ in range(3) :
                if type(x) is list:
                    latents = [self.decoder_cross_attn_refine(latents[i],
                                                              context=x[i])
                               + latents[i]
                               for i in range(b)]
                elif type(queries) is list:
                    # TODO: batch/concatenate? for feed forward
                    # NOTE: concat at dim 1 to keep one batch
                    # TODO: is this a good idea, what about batch norm
                    latents = [self.decoder_cross_attn_refine(latents[i],
                                                              context=x[[i], :])
                               + latents[i]
                               for i in range(b)]
                else:
                    latents = self.decoder_cross_attn_refine(latents, context=x) + latents


        if self.latent_long_range_query :
            # TODO: what is already unsqueezed here?
            if type(x_after_cross) is list:
                latents = [self.decoder_cross_attn_long_range(latents[i],
                                                              context=x_after_cross[i])
                           + latents[i].unsqueeze(0)
                           for i in range(b)]
            elif type(latents) is list:
                # TODO: batch/concatenate? for feed forward
                # NOTE: concat at dim 1 to keep one batch
                # TODO: is this a good idea, what about batch norm
                latents = [self.decoder_cross_attn(latents[i],
                                                   context=x_after_cross[[i], :])
                           + latents[i].unsqueeze(0)
                           for i in range(b)]
            else:
                latents = latents + self.decoder_cross_attn_long_range(latents, context=x_after_cross)

        if self.low_level_latents :
            # TODO: assume batched maybe?
            x_more = repeat(self.more_latents, 'n d -> b n d', b = b)
            cross_attn, cross_ff = self.more_cross
            x_more = apply_cross(cross_attn, cross_ff, x_more, data, mask)
            self_attn, self_ff = self.more_self
            x_more = self_attn(x_more) + x_more
            x_more = self_ff(x_more) + x_more
            latents = latents + self.decoder_cross_attn_long_range(latents, context=x_more)


        # TODO: insert latent residual update latents + cross_attn(latents, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            if type(latents) is list :
                latents = [latents[i] + self.decoder_ff(latents[i])
                           for i in range(len(latents))]
            else :
                latents = latents + self.decoder_ff(latents)

        # final linear out
        if type(latents) is list :
            return [self.to_logits(latents[i]).squeeze(0)
                    for i in range(len(latents))]
        else :
            return list(self.to_logits(latents).unbind(dim=0))

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits
