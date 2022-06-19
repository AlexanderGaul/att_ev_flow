import torch
import torch.nn as nn

from repositories.vit_pytorch_repo.vit_pytorch.twins_svt import *


class SS(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1',):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            layers.append(nn.Sequential(
                PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
                Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
                PEG(dim = dim_next, kernel_size = peg_kernel_size),
                Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
            ))

            dim = dim_next

        self.layers = nn.Sequential(
            *layers,
            Rearrange('... d h w -> ... h w d'),
            nn.Linear(dim, num_classes),
            Rearrange('... h w d -> ... d h w')
        )

    def forward(self, x):
        for l in self.layers :
            x = l(x)
        return x


class AGTLayer(torch.nn.Module) :
    def __init__(self) :
        super().__init__()

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.self_attend = nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ])


class AGT(torch.nn.Module) :
    def __init__(self,
                 depth,
                 num_latents,
                 latent_dim,
                 cross_heads,
                 latent_heads,
                 cross_dim_head,
                 latent_dim_head) :
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend = nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
                PreNorm(latent_dim, FeedForward(latent_dim))
            ])

        pass
