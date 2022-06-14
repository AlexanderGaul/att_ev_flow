import torch

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from utils import gaussian_encoding, sinusoidal_encoding
from ERAFT.model.utils import coords_grid

class PositionEncoder(torch.nn.Module) :
    def __init__(self, num_bands, encoding_type='perceiver') :
        super().__init__()
        self.encoding_type = encoding_type
        self.pos_bands = num_bands

        if (self.encoding_type == 'random_fourier_features' or
                self.encoding_type == 'learned_fourier_features'):
            self.ff_weight_scale = 1.
            ff_init_normal = True
            requires_grad = self.encoding_type == 'learned_fourier_features'
            if ff_init_normal:
                self.W_coords = torch.nn.Parameter(torch.randn((self.pos_bands * 2, 2)) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                """
                self.W_t = torch.nn.Parameter(torch.randn((self.t_bands, 1)) * self.ff_weight_scale,
                                              requires_grad=requires_grad)
                """
            else:
                self.W_coords = torch.nn.Parameter((torch.rand((self.pos_bands * 2, 2)) * 2 - 1) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                """
                self.W_t = torch.nn.Parameter((torch.rand((self.t_bands, 1)) * 2 - 1) * self.ff_weight_scale,
                                              requires_grad=requires_grad)
                """

    def forward(self, *args, **kwargs) :
        return self.encode_positions(*args, **kwargs)


    def encode_positions(self, locs, res, return_normalized=True, cat_normalized=False):
        res = torch.tensor(res, dtype=torch.float32)
        res = res.type_as(locs)
        locs_norm = locs / res.reshape(-1, *((1,) * (locs.ndim-2)), 2) * 2 - 1
        """
        if self.res_fixed :
            res_fixed = torch.tensor(self.res_fixed, device=locs.device, dtype=torch.float32)
            if len(res.shape) > 1 :
                res_fixed = res_fixed.repeat(res.shape[0], 1)
            res = res_fixed
        """

        if self.encoding_type == 'perceiver':
            if locs.ndim == 2:
                assert res.shape == torch.Size([2])
                x_en = fourier_encode(locs_norm[..., 0], res[0], num_bands=self.pos_bands, cat_orig=False)
                y_en = fourier_encode(locs_norm[..., 1], res[1], num_bands=self.pos_bands, cat_orig=False)
                locs_en = torch.cat([x_en, y_en], dim=-1)
            else:
                if res.ndim == 1:
                    assert locs.shape[0] == 1
                    res = res.unsqueeze(0)
                assert res.shape[1] == 2
                x_en = torch.cat([fourier_encode(locs_norm[[i], ..., 0], res[i, 0],
                                                 num_bands=self.pos_bands, cat_orig=False)
                                  for i in range(locs_norm.shape[0])], dim=0)
                y_en = torch.cat([fourier_encode(locs_norm[[i], ..., 1], res[i, 1],
                                                 num_bands=self.pos_bands, cat_orig=False)
                                  for i in range(locs_norm.shape[0])], dim=0)
                locs_en = torch.cat([x_en, y_en], dim=-1)

        elif self.encoding_type == 'random_fourier_features' or self.encoding_type == 'learned_fourier_features':
            locs_en = gaussian_encoding(locs_norm,
                                        self.W_coords, sigma=1.)

        elif self.encoding_type == 'transformer':
            x_en = sinusoidal_encoding(locs[..., [0]], self.pos_bands)
            y_en = sinusoidal_encoding(locs[..., [1]], self.pos_bands)
            locs_en = torch.cat([x_en, y_en], dim=-1)

        if return_normalized:
            return locs_norm, locs_en
        elif cat_normalized:
            return torch.cat([locs_norm, locs_en], dim=-1)
        else:
            return locs_en


    def coords_arrays(self, N, H, W, device) :
        coords = coords_grid(N, H, W).to(device)
        coords = coords.permute(0, 2, 3, 1)
        coords = coords.reshape(N, -1, 2)

        coords_norm, coords_pe = self.encode_positions(coords,
                                                       res=[(W, H) for _ in range(N)],
                                                       return_normalized=True)

        return coords, coords_norm, coords_pe
