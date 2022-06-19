import torch

from perceiverpytorch.perceiver_pytorch.perceiver_io import PreNorm, Attention

from ERAFT.model.utils import coords_grid, bilinear_sampler

class QueryRefine(torch.nn.Module) :
    def __init__(self, queries_dim, latent_dim, cross_heads, cross_dim_head) :
        super().__init__()


        self.decoder_cross_attn_refine = PreNorm(queries_dim,
                                                 Attention(queries_dim, latent_dim, heads=cross_heads,
                                                           dim_head=cross_dim_head),
                                                 context_dim=latent_dim)


    def forward(self, latents, embedding, coords_pred, coords_encoded, feature_grid) :
        features_sampled = bilinear_sampler(feature_grid, coords_pred)

        query = torch.cat([coords_encoded, features_sampled, embedding], dim=-1)

        embedding = self.decoder_cross_attn_refine(query, context=latents)

        return embedding