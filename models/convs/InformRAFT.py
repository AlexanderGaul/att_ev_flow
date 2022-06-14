import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ERAFT.model.update import BasicUpdateBlock
from ERAFT.model.extractor import BasicEncoder
from ERAFT.model.corr import CorrBlock
from ERAFT.model.utils import coords_grid, upflow8
from argparse import Namespace
from ERAFT.utils.image_utils import ImagePadder

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO, PreNorm, Attention

from utils import gaussian_encoding, sinusoidal_encoding

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(small=False,
                     dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args


class InformERAFT(nn.Module):
    def __init__(self, config, n_first_channels,
                 corr_levels=4,
                 perceiver_params=None,
                 init_flow=True):
        # args:
        super(InformERAFT, self).__init__()
        args = get_args()
        self.args = args
        self.image_padder = ImagePadder(min_size=32)
        self.subtype = config['subtype'].lower()

        assert (self.subtype == 'standard' or self.subtype == 'warm_start')

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = corr_levels
        args.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                 n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=0,
                                 n_first_channels=n_first_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim if init_flow else cdim*2)

        self.pos_bands = 32

        dim = 256 * 2 + self.pos_bands * 4 + 2
        qdim = dim
        self.perceiver_params = dict(
            dim=dim,  # dimension of sequence to be encoded
            queries_dim=qdim,  # dimension of decoder queries
            logits_dim=2 if init_flow else cdim,  # dimension of final logits
            depth=4,  # depth of net
            num_latents=128,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=256,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=4,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff=False
        )
        if perceiver_params is None:
            perceiver_params = {}

        self.perceiver_params.update(perceiver_params)

        self.perceiver = PerceiverIO(**self.perceiver_params)

        self.encoding_type = 'perceiver'

        self.init_flow = init_flow


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=4, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """
        # Pad Image (for flawless up&downsampling)
        image1 = self.image_padder.pad(image1)
        image2 = self.image_padder.pad(image2)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius,
                            num_levels=self.args.corr_levels)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if (self.subtype == 'standard' or self.subtype == 'warm_start'):
                cnet = self.cnet(image2)
            else:
                raise Exception
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(image1)

        N, T, H, W = image1.shape
        Hdown, Wdown = H // 8, W // 8

        farray1 = fmap1.reshape(N, -1, Hdown * Wdown).transpose(1, 2)
        farray2 = fmap2.reshape(N, -1, Hdown * Wdown).transpose(1, 2)
        coords, coords_norm, coords_pe = self.coords_arrays(N, Hdown, Wdown, image1.device)
        array = torch.cat([coords_norm, coords_pe, farray1, farray2], dim=-1)
        p_flow = torch.stack(self.perceiver(array, queries=array)).reshape(N, Hdown, Wdown, -1)
        p_flow = p_flow.permute(0, 3, 1, 2)
        if self.init_flow :
            flow_init = p_flow
        else :
            inp = torch.cat([inp, torch.relu(p_flow)], dim=1)

        if flow_init is not None:
            coords1 = coords1 + flow_init
            flow_predictions = [upflow8(flow_init)]
        else :
            flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(self.image_padder.unpad(flow_up))

        return coords1 - coords0, flow_predictions

    def encode_positions(self, locs, res, return_normalized=True, cat_normalized=False):
        res = torch.tensor(res, dtype=torch.float32)
        res = res.type_as(locs)
        locs_norm = locs / res.unsqueeze(-2) * 2 - 1
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
