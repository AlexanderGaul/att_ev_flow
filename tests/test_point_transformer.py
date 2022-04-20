import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pointtransformerpytorch.point_transformer_pytorch.point_transformer_pytorch import PointTransformerLayer


attn = PointTransformerLayer(
    dim = 128,
    pos_mlp_hidden_dim = 64,
    attn_mlp_hidden_mult = 4
)

feats = torch.randn(1, 16, 128)
pos = torch.randn(1, 16, 3)
mask = torch.ones(1, 16).bool()

out = attn(feats, pos, mask = mask)


print(out.shape)