import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model import EventTransformer

from utils import get_grid_coordinates

import matplotlib.pyplot as plt
"""
from math import pi

from perceiver_pytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
"""

model = EventTransformer(pos_bands=8)

res = torch.tensor([640, 480]) / 4

# TODO: create grid
grid = get_grid_coordinates((640 // 4, 480 // 4), (0, 0))

grid_grid = grid.reshape((480 // 4, 640 // 4, 2))

x = torch.tensor([[640 - 160, 240]]) // 4

freq = torch.tensor([1, 1])


def custom_encoding(x, min_freq=1./100., num_bands=4, cat_orig=True) :
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    freqs = torch.arange(num_bands) / (num_bands * 2)
    scales = torch.pow(min_freq, freqs)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    if cat_orig:
        x = torch.cat((x, orig_x), dim=-1)
    return x


def gaussian_encoding(x, w, sigma=1) :
    w = w * sigma
    xp = 2 * np.pi * x @ w.t()
    return torch.cat([torch.cos(xp), torch.sin(xp)], axis=-1)


"""def fourier_encode(x, max_freq, num_bands=4, cat_orig=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    if cat_orig:
        x = torch.cat((x, orig_x), dim=-1)
    return x"""

sigma = 128
w =  ((torch.randn((64, 2)))* sigma)

grid_en = gaussian_encoding(torch.tensor(grid) , w,
                            sigma=1)
x_en = gaussian_encoding(x, w,
                         sigma=1)

scores = grid_en.mm(x_en.t())

im = scores.reshape((int(res[1].item()),
                     int(res[0].item()))).numpy()

plt.figure(figsize=(32, 24))
plt.imshow(im)
plt.show()

"""
def sinusoidal_encoding(x, D) :
    freqs = 1 / torch.pow(10000, torch.linspace(0, 1, D)[:-1])
    sins = torch.sin(x.reshape(-1, 1) * freqs.reshape(1, -1))
    coss = torch.cos(x.reshape(-1, 1) * freqs.reshape(1, -1))
    return torch.cat([sins, coss], dim=-1)


D = 8
grid_en_x = sinusoidal_encoding(torch.tensor(grid[:, 0] / res[0] * 2 - 1), D)
grid_en_y = sinusoidal_encoding(torch.tensor(grid[:, 1] / res[1] * 2 - 1), D)

grid_en = torch.cat([grid_en_x, grid_en_y], dim=-1)

x_en_x = sinusoidal_encoding(x[:, 0] / res[0] * 2 - 1, D)
x_en_y = sinusoidal_encoding(x[:, 1] / res[1] * 2 - 1, D)

x_en = torch.cat([x_en_x, x_en_y], dim=-1)

scores = grid_en[:, :].float().mm(x_en[:, :].t())

im = scores.reshape((480 // 4, 640 // 4)).numpy()

plt.figure(figsize=(32, 24))
plt.imshow(im)
plt.show()
"""

"""
model.pos_bands = 4
grid_en_x = custom_encoding(torch.tensor(grid[:, 0]), num_bands=4, cat_orig=False) #
grid_en_y = custom_encoding(torch.tensor(grid[:, 1]), num_bands=4, cat_orig=False)
grid_en = torch.cat([grid_en_x, grid_en_y], dim=1)
grid_en = model.encode_positions(grid, res, cat_orig=False)
x_en_x = custom_encoding(x[:, 0], num_bands=4, cat_orig=False) #
x_en_y = custom_encoding(x[:, 1], num_bands=4, cat_orig=False)
x_en = torch.cat([x_en_x, x_en_y], dim=1)
x_en = model.encode_positions(x, res, cat_orig=False)

scores = grid_en[:, :].float().mm(x_en[:, :].t())

im = scores.reshape((480 // 2, 640 // 2)).numpy()

plt.figure(figsize=(32, 24))
plt.imshow(im)
plt.show()
"""