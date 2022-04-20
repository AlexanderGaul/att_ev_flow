import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from model import EventTransformer
from perceiverpytorch.perceiver_pytorch.perceiver_io import  *

from utils import get_grid_coordinates

import matplotlib.pyplot as plt

from pathlib import Path
import json

dir = Path("/storage/user/gaul/gaul/thesis/output/runs/week_13_edge/eventsraw_transformer/")
checkpoint = torch.load(dir / "checkpoint100")

config = json.load(open(dir / "config.json"))
print(config['model'])
config['model']['ff_weight_scale'] = config['model'].pop('ff_encoding_sigma')

model = EventTransformer(ff_init_normal=False, **config['model'])

print(model.W_coords.data.mean())
print(model.W_coords.data.std())
print(model.W_coords.data.abs().max())

model.load_state_dict(checkpoint['model'])

print(model.W_coords.data.mean())
print(model.W_coords.data.std())
print(model.W_coords.data.abs().max())

coords = torch.tensor(get_grid_coordinates((64, 64)), dtype=torch.float32)

event_center = torch.tensor([[32, 32, 0, 1]])
events_grid = torch.cat([coords, torch.zeros((64 * 64, 1)), torch.ones((64 * 64, 1))], dim=1)

center_en = model.encode_events(event_center.unsqueeze(0), (64, 64), 100)
grid_en = model.encode_events(events_grid.unsqueeze(0), (64, 64), 100)



def apply_cross_attention(model, q_data, k_data) :
    pre_norm = model.model.cross_attend_layers[0][0]
    q_data_norm = pre_norm.norm(q_data)
    k_data_norm = pre_norm.norm(k_data)
    if pre_norm.norm_context is not None :
        k_data_norm = pre_norm.norm_context(k_data)

    attention_fn = pre_norm.fn
    q = attention_fn.to_q(q_data_norm)
    k, _ = attention_fn.to_kv(k_data_norm).chunk(2, dim=-1)
    q, k = map(lambda t : rearrange(t, 'b n (h d) -> (b h) n d', h=attention_fn.heads), (q, k))
    sim = einsum('b i d, b j d -> b i j', q, k) * attention_fn.scale

    return sim

sim = apply_cross_attention(model, center_en, grid_en).detach().numpy()

plt.imshow(sim.reshape(64, 64))
plt.show()

print("EOF")