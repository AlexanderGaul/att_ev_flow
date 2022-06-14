import torch

from models.convs.conv2d_modules import Upsampler, UpsamplerV2, BasicEncoder
from models.convs.conv3d_modules import BasicEncoder3D

from models.convs.update import BasicUpdateBlock

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO, PreNorm, Attention

from ERAFT.model.utils import coords_grid, bilinear_sampler

from utils import gaussian_encoding, sinusoidal_encoding

import torch.nn.functional as F

from models.position_encoding import PositionEncoder
from models.attn.iterative_perceiver_io import PerceiverIO as IterativePerceiverIO

class ConvTransformer(torch.nn.Module) :
    def __init__(self,
                 t_dims,
                 perceiver_params,
                 norm_fn='batch',
                 iterative_updates=0,
                 gru_updates=False,
                 return_iterations=False,
                 query_src_and_trgt_positions=False,
                 query_gru_state=False,
                 input_2_frames=False,
                 encode_current_and_prev_volume=False,
                 query_current_and_prev_volume=False,
                 no_query_coords_updates=False,
                 encoding_type='perceiver',
                 iterative_perceiver=False,
                 skip_perceiver=False,
                 feed_flow_into_up_mask=False,
                 detach_up_mask_feed=False,
                 fdim=128,
                 full_transformer=False) :
        super().__init__()

        # TODO: change dimensions
        self.fdim = fdim
        self.hdim = 128
        self.cdim = 128
        self.outdim_perceiver = 128
        self.pos_bands = 32

        self.num_conv_scale_layers = 3


        dim = self.fdim + self.pos_bands * 4 + 2
        qdim = dim
        if iterative_updates :
            if encode_current_and_prev_volume and not iterative_perceiver :
                dim += self.fdim
            qdim += self.fdim
            if query_src_and_trgt_positions :
                qdim += self.pos_bands * 4 + 2
            if query_gru_state :
                qdim += self.hdim + self.cdim
        elif input_2_frames :
            if encode_current_and_prev_volume :
                dim += self.fdim
            if query_current_and_prev_volume :
                qdim += self.fdim
        else :
            if encode_current_and_prev_volume and not iterative_perceiver:
                dim += self.fdim
            if query_current_and_prev_volume :
                qdim += self.fdim

        self.perceiver_params = dict(
            dim=dim,  # dimension of sequence to be encoded
            queries_dim=qdim,  # dimension of decoder queries
            logits_dim=self.outdim_perceiver if (iterative_updates or gru_updates) else 2,  # dimension of final logits
            depth=4,  # depth of net
            num_latents=256,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=256 if not full_transformer else dim,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=4,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff=False,
            transformer_encoder=full_transformer
        )


        self.fnet = BasicEncoder(n_first_channels=t_dims,
                                 output_dim=self.fdim,
                                 norm_fn=norm_fn,
                                 num_layers=self.num_conv_scale_layers)

        self.cnet = BasicEncoder(n_first_channels=t_dims,
                                 output_dim=self.hdim+self.cdim,
                                 norm_fn=norm_fn,
                                 num_layers=self.num_conv_scale_layers)

        if not gru_updates :
            in_dim = self.cdim+self.hdim
            if feed_flow_into_up_mask :
                in_dim += 2
            self.mask = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, 256, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 8 ** 2 * 9, 1, padding=0))

        self.perceiver_params.update(perceiver_params)

        self.iterative_perceiver = iterative_perceiver
        if not iterative_perceiver :
            self.perceiver = PerceiverIO(**self.perceiver_params)
        else :
            self.perceiver = IterativePerceiverIO(**self.perceiver_params)

        self.gru_updates = gru_updates
        if gru_updates:
            assert iterative_updates
            self.update_block = BasicUpdateBlock()

        self.iterative_updates = iterative_updates

        self.res_fixed = False

        self.return_iterations = return_iterations

        self.query_src_and_trgt_positions=query_src_and_trgt_positions
        self.query_gru_state = query_gru_state
        self.encode_current_and_prev_volume = encode_current_and_prev_volume
        self.query_current_and_prev_volume = query_current_and_prev_volume

        self.no_query_coords_updates = no_query_coords_updates

        self.pos_encoder = PositionEncoder(num_bands=self.pos_bands, encoding_type=encoding_type)

        self.skip_perceiver = skip_perceiver
        if skip_perceiver :
            self.conv_flow = torch.nn.Conv2d(self.fdim,
                                             self.outdim_perceiver if (iterative_updates or gru_updates) else 2, 1)

        self.feed_flow_into_up_mask = feed_flow_into_up_mask
        self.detach_up_mask_feed = detach_up_mask_feed


    def coords_arrays(self, N, H, W, device) :
        coords = coords_grid(N, H, W).to(device)
        coords = coords.permute(0, 2, 3, 1)
        coords = coords.reshape(N, -1, 2)

        coords_norm, coords_pe = self.pos_encoder.encode_positions(coords,
                                                       res=[(W, H) for _ in range(N)],
                                                       return_normalized=True)

        return coords, coords_norm, coords_pe


    def forward(self, volume, volume_prev=None) :
        N, T, H, W = volume.shape

        fmap = self.fnet(volume)

        if volume_prev is not None :
            fmap_prev = self.fnet(volume_prev)

        Hdown, Wdown = fmap.shape[-2], fmap.shape[-1]


        coords, coords_norm, coords_pe = self.coords_arrays(N, Hdown, Wdown, volume.device)

        farray = fmap.reshape(N, -1, Hdown * Wdown).transpose(1, 2)

        if volume_prev is not None :
            farray_prev = fmap_prev.reshape(N, -1, Hdown*Wdown).transpose(1, 2)

        array = torch.cat([coords_norm, coords_pe, farray], dim=-1)
        query = array
        if not self.iterative_updates :
            if volume_prev is not None :
                # farray_prev = torch.cat([coords_norm, coords_pe, farray_prev])
                if self.encode_current_and_prev_volume and self.query_current_and_prev_volume :
                    array = torch.cat([array, farray_prev], dim=-1)
                    query = array
                else :
                    raise NotImplementedError()
            if not self.skip_perceiver :
                p_out = torch.stack(self.perceiver(array, queries=query)).reshape(N, Hdown, Wdown, -1)
                pred_grid = p_out.reshape(N, Hdown, Wdown, 2).permute(0, 3, 1, 2)
            else :
                pred_grid = self.conv_flow(fmap)

            cmap = self.cnet(volume)
            if self.feed_flow_into_up_mask :
                cmap = torch.cat([cmap, pred_grid if not self.detach_up_mask_feed else pred_grid.detach()], dim=-3)
            mask = self.mask(cmap)
            # TODO: reshape prediction
            pred = upsample_flow(pred_grid, mask, H // Hdown)
            return pred

        else :
            if self.gru_updates :
                cmap = self.cnet(volume)
                state, cmap = torch.split(cmap, [self.hdim, self.cdim], dim=1)
                state = torch.tanh(state)
                cmap = torch.relu(cmap)

            if not self.skip_perceiver :
                if not self.iterative_perceiver :
                    if volume_prev is not None and self.encode_current_and_prev_volume :
                        array = torch.cat([array, farray_prev], dim=-1)
                    latents = self.perceiver(array, queries=None)
                else :
                    assert self.encode_current_and_prev_volume
                    array_prev = torch.cat([coords_norm, coords_pe, farray_prev], dim=-1)
                    latents = self.perceiver(array, array_prev, queries=None)

            coords_i = coords
            coords_0_grid = coords.reshape(N, Hdown, Wdown, 2).permute(0, 3, 1, 2)
            flow = torch.zeros(coords_0_grid.shape, device=coords_0_grid.device)

            flow_preds = []

            for i in range(self.iterative_updates) :
                # Encode positions need to be arrays
                coords_i = coords_i.detach()
                coords_i_pe = torch.cat(self.pos_encoder.encode_positions(coords_i,
                                                                  res=[(Wdown, Hdown) for _ in range(N)],
                                                                  return_normalized=True),
                                            dim=-1)

                target_fmap = bilinear_sampler(fmap, coords_i.reshape(N, Hdown, Wdown, 2))
                target_farray = target_fmap.reshape(N, -1, Hdown * Wdown).transpose(-2, -1)
                if not self.query_src_and_trgt_positions :
                    query_i = torch.cat([coords_i_pe, target_farray, farray if volume_prev is None else farray_prev], axis=-1)
                else :
                    query_i = torch.cat([coords_norm, coords_pe, farray if volume_prev is None else farray_prev, coords_i_pe, target_farray], axis=-1)

                if self.query_gru_state :
                    cmap_flat = cmap.reshape(N, -1, Hdown*Wdown).transpose(-1, -2)
                    state_flat  = state.reshape(N, -1, Hdown*Wdown).transpose(-1, -2)
                    query_i = torch.cat([query_i, cmap_flat, state_flat], dim=-1)

                if not self.skip_perceiver :
                    corr = self.perceiver.decoder_cross_attn(query_i, context=latents)
                    corr = self.perceiver.to_logits(corr)
                    corr = corr.reshape(N, Hdown, Wdown, -1).permute(0, 3, 1, 2)
                else :
                    corr = self.conv_flow(fmap)

                state, up_mask, delta_flow = self.update_block(state, cmap, corr, flow)

                flow = flow + delta_flow
                if self.no_query_coords_updates :
                    coords_i += delta_flow.reshape(N, 2, Hdown*Wdown).permute(0, 2, 1)
                pred_up = upsample_flow(flow, up_mask, H // Hdown)

                flow_preds.append(pred_up)

                pass

            if not self.return_iterations :
                return flow_preds[-1]
            else :
                return flow_preds[-1], flow_preds
            # TODO: return multiple predictions


        # TODO perceiver and upsample



def upsample_flow(flow, mask, scale):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, scale, scale, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(scale * flow, [3,3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, scale*H, scale*W)


# TODO:
# allow for multiple flow predictions
# gru