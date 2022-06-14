import torch
import torch.nn as nn

from perceiverpytorch.perceiver_pytorch.perceiver_io import PreNorm, Attention, FeedForward, cache_fn, exists
from einops import repeat


class PerceiverIO(nn.Module):
    def __init__(
            self,
            *,
            depth,
            dim,
            queries_dim,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            initial_cross_is_residual=False
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
                    context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        self.cross_attend_blocks_update = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
                    context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads,
                                                                 dim_head=cross_dim_head), context_dim=latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

        self.initial_cross_is_residual = initial_cross_is_residual

    def forward(
            self,
            data,
            data_update,
            mask=None,
            queries=None
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        if self.initial_cross_is_residual :
            x = cross_attn(x, context=data, mask=mask) + x
        else :
            x = cross_attn(x, context=data, mask=mask)
        x = cross_ff(x) + x



        cross_attn, cross_ff = self.cross_attend_blocks_update

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context=data_update, mask=mask) + x
        x = cross_ff(x) + x


        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        # cross attend from decoder queries to latents

        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)