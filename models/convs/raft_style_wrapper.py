
import torch

from models.convs.conv2d_modules import Upsampler, UpsamplerV2, BasicEncoder
from models.convs.conv3d_modules import BasicEncoder3D

from models.convs.update import BasicUpdateBlock

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO, PreNorm, Attention

from ERAFT.model.utils import coords_grid, bilinear_sampler

from utils import gaussian_encoding, sinusoidal_encoding

import torch.nn.functional as F


class WrappedTransformer(torch.nn.Module) :

    def __init__(self, t_dims=4, perceiver_params=None,
                 volume_encoding='conv2d',
                 encoding_type='perceiver',
                 ff_weight_scale=1.,
                 ff_init_normal=True,
                 norm_fn='batch',
                 norm_fn_up=None,
                 split_time=False,
                 num_conv_scale_layers=3,
                 upscale_version='1',
                 iterative_temporal_queries=False,
                 iterative_queries=False,
                 num_iterative_queries=1,
                 iterative_queries_mode='separate_with_embedding',
                 convex_upsample=False,
                 convex_upsample_encode=False,
                 perceiver_res_fixed=None,
                 hdim_encoder=128,
                 hdim_decoder=64,
                 fixed_pe=False
                 ) :
        super().__init__()

        if norm_fn_up is None :
            norm_fn_up = norm_fn

        # TODO: set hidden dimensions
        self.hdim_encoder = hdim_encoder
        self.cdim = 0
        self.hdim_decoder = hdim_decoder

        self.pos_bands = 32
        self.t_bands = 32
        self.t_dims = t_dims
        self.scale = 2**num_conv_scale_layers

        self.res_fixed = perceiver_res_fixed

        self.split_time = split_time
        # TOD: allow for one and 0 scaling layers
        self.num_conv_scale_layers = num_conv_scale_layers

        self.iterative_temporal_queries = iterative_temporal_queries
        if self.iterative_temporal_queries : assert volume_encoding == 'conv3d' and self.split_time

        self.iterative_queries = iterative_queries
        self.iterative_queries_mode = iterative_queries_mode

        self.num_iterative_queries = num_iterative_queries

        self.gru_updates = False

        self.convex_upsample = convex_upsample
        self.convex_upsample_encode = convex_upsample_encode

        self.fixed_pe = fixed_pe

        if perceiver_params is None :
            perceiver_params = {}

        if volume_encoding == 'conv2d' :
            dim = self.hdim_encoder+self.pos_bands*4+2
            qdim = dim
            if self.iterative_queries and self.iterative_queries_mode == 'recurrent':
                qdim += self.hdim_encoder
        elif volume_encoding == 'conv3d' :
            if self.split_time :
                dim = self.hdim_encoder + self.pos_bands * 4 + 2 + self.t_bands * 2 + 1
                qdim = self.hdim_encoder + self.pos_bands * 4 + 2 + self.t_bands * 2 + 1
                if self.iterative_temporal_queries :
                    qdim += self.hdim_encoder
            else :
                dim = self.hdim_encoder * t_dims + self.pos_bands * 4 + 2
                qdim = dim

        self.perceiver_params = dict(
            dim=dim,  # dimension of sequence to be encoded
            queries_dim=qdim,  # dimension of decoder queries
            logits_dim=self.hdim_decoder if not self.iterative_temporal_queries
                             and not self.iterative_queries
                             and not self.convex_upsample else 2,  # dimension of final logits
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

        self.perceiver_params.update(perceiver_params)

        self.perceiver = PerceiverIO(**self.perceiver_params)

        if self.iterative_queries :
            iterative_query_dim = self.pos_bands*4 + 2 + 2*self.hdim_encoder + qdim # pe_dim, features_dim, features_dim, query_out_dim
            self.decoder_cross_attn_refine = PreNorm(iterative_query_dim,
                                                     Attention(iterative_query_dim,
                                                               self.perceiver_params['latent_dim'],
                                                               heads=self.perceiver_params['cross_heads'],
                                                               dim_head=self.perceiver_params['cross_dim_head']),
                                                     context_dim=self.perceiver_params['latent_dim'])
            self.refine_to_logits = torch.nn.Linear(iterative_query_dim, self.perceiver_params['logits_dim'], bias=True)


        self.volume_encoding = volume_encoding
        if self.volume_encoding == 'conv2d' :
            self.volume_encoder = BasicEncoder(n_first_channels=t_dims,
                                               output_dim=self.hdim_encoder,
                                               norm_fn=norm_fn,
                                               num_layers=self.num_conv_scale_layers)
            if convex_upsample :
                self.volume_encoder_2 = BasicEncoder(n_first_channels=t_dims,
                                                     output_dim=self.hdim_encoder if not self.gru_updates else
                                                        self.hdim_encoder + self.cdim,
                                                     norm_fn=norm_fn,
                                                     num_layers=self.num_conv_scale_layers)
        elif self.volume_encoding == 'conv3d' :
            self.volume_encoder = BasicEncoder3D(n_first_channels=1,
                                                 output_dim=self.hdim_encoder,
                                                 norm_fn=norm_fn,
                                                 num_layers=self.num_conv_scale_layers)
            if convex_upsample :
                self.volume_encoder_2 = BasicEncoder3D(n_first_channels=1,
                                                       output_dim=self.hdim_encoder if not self.gru_updates else
                                                       self.hdim_encoder + self.cdim,
                                                       norm_fn=norm_fn,
                                                       num_layers=self.num_conv_scale_layers)
        if not self.convex_upsample :
            if upscale_version == '1' :
                self.upsampler = Upsampler(self.hdim_decoder, 2,
                                           norm_fn=norm_fn_up, num_layers=self.num_conv_scale_layers)
            elif upscale_version == '2' :
                self.upsampler = UpsamplerV2(self.hdim_decoder, 2,
                                             norm_fn=norm_fn_up, num_layers=self.num_conv_scale_layers)

        if self.convex_upsample :
            self.mask = torch.nn.Sequential(
                torch.nn.Conv2d(self.hdim_encoder, 256, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, self.scale**2 * 9, 1, padding=0))


        if self.gru_updates :
            self.update_block = BasicUpdateBlock(input_dim=self.hdim_encoder)
            self.cdim = self.hdim_encoder


        self.encoding_type = encoding_type
        if (self.encoding_type == 'random_fourier_features' or
                self.encoding_type == 'learned_fourier_features'):
            self.ff_weight_scale = ff_weight_scale
            requires_grad = self.encoding_type == 'learned_fourier_features'
            if ff_init_normal:
                self.W_coords = torch.nn.Parameter(torch.randn((self.pos_bands * 2, 2)) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                self.W_t = torch.nn.Parameter(torch.randn((self.t_bands, 1)) * self.ff_weight_scale,
                                              requires_grad=requires_grad)
            else:
                self.W_coords = torch.nn.Parameter((torch.rand((self.pos_bands * 2, 2)) * 2 - 1) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                self.W_t = torch.nn.Parameter((torch.rand((self.t_bands, 1)) * 2 - 1) * self.ff_weight_scale,
                                              requires_grad=requires_grad)





    def forward(self, event_volume) :
        # event volume is batched
        N, T, H, W = event_volume.shape
        # TODO: encode or extract patches
        if self.volume_encoding == 'conv2d' :
            volume_en = self.volume_encoder(event_volume)
        elif self.volume_encoding == 'conv3d' :
            volume_en = self.volume_encoder(event_volume.unsqueeze(1))

        Hdown, Wdown = volume_en.shape[-2], volume_en.shape[-1]

        if self.split_time :
            assert self.volume_encoding == 'conv3d'
            coords, ts = self.volume_encode_all_dims(volume_en)

            pos = torch.cat([coords, ts], dim = 1)
            _, P, _, _, _ = pos.shape

            # TODO: missing T
            pos_array = pos.reshape(N, P, T*(Hdown*Wdown)).transpose(1, 2)
            array = volume_en.reshape(N, -1, T*(Hdown*Wdown)).transpose(1, 2)
            array = torch.cat([pos_array, array], dim = -1)

            pos_0_array = pos[:, :, 0].reshape(N, P, (Hdown*Wdown)).transpose(-1, -2)
            vol_array_0 = volume_en[:, :, 0].reshape(N, -1, (Hdown*Wdown)).transpose(-1, -2)
            query = torch.cat([pos_0_array, vol_array_0], dim=-1)

            if not self.iterative_temporal_queries :
                pred = torch.stack(self.perceiver(array, queries=query)).reshape(N, Hdown, Wdown, 64)

            else :
                latents = self.perceiver(array, queries=None)
                # return latents from perceiver
                # call query cross attention manually
                coords = coords_grid(N, Hdown, Wdown).to(event_volume.device)
                coords = coords.permute(0, 2, 3, 1)
                coords = coords.reshape(N, -1, 2) # shape B x (HW) x 2

                # TODO: do we repeat coordinates or do we store flows

                flows_T = torch.zeros((N, T-1, Hdown*Wdown, 2), device=event_volume.device)

                feat_t0 = vol_array_0
                # TODO: repeat T times
                for i in range(self.num_iterative_queries) :
                    for t in range(1, T) :
                        coords_t = coords + flows_T[:, t-1]
                        coords_t_pe = torch.cat(self.encode_positions(coords_t,
                                                res=[(Wdown, Hdown) for _ in range(N)],
                                                return_normalized=True),
                                                dim=-1)
                        t_pe = torch.cat(self.encode_time(torch.tensor([t+1], device=event_volume.device).reshape(1, 1),
                                                [T],
                                                return_normalized = True),
                                         axis=-1)
                        t_pe = t_pe.unsqueeze(0)
                        t_pe = t_pe.repeat(N, Hdown*Wdown, 1) # need shape N x HW x X

                        coords_t_grid = coords_t.reshape(N, Hdown, Wdown, 2)
                        feat_t = bilinear_sampler(volume_en[:, :, t, ...], coords_t_grid)
                        feat_t = feat_t.reshape(N, -1, Hdown*Wdown).transpose(-2, -1)
                        # TODO: reshape

                        query = torch.cat([coords_t_pe, t_pe, feat_t0, feat_t], dim=-1)

                        latents_t = self.perceiver.decoder_cross_attn(query, context=latents)
                        if self.perceiver.decoder_ff is not None :
                            latents_t = self.perceiver.decoder_ff(latents_t)
                        flow = self.perceiver.to_logits(latents_t)
                        # TODO billinear interpolate featuers
                        # query and decode position
                        # update the next position in the coords volume
                        flows_T[:, t-1:] += flow.unsqueeze(1)

                # TODO: how to upsample flow
                pred = flows_T[:, -1]
                pred = pred.transpose(-2, -1)
                pred = pred.reshape(N, 2, Hdown, Wdown)

                if self.res_fixed :
                    pred *= Wdown / self.res_fixed[0]

                if not self.convex_upsample :
                    # TODO: upsample prediction bilinear?
                    pred_up = H / Hdown * torch.nn.functional.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
                    return pred_up

            if self.convex_upsample :
                if not self.convex_upsample_encode :
                    volume_en_context = self.volume_encoder(event_volume.unsqueeze(1))
                else :
                    volume_en_context = self.volume_encoder_2(event_volume.unsqueeze(1))
                mask = self.mask(volume_en_context[:, :, 0])

                pred_up = upsample_flow(pred, mask, H // Hdown)
                return pred_up

        else :
            #volume_en = volume_en.reshape(N, -1, H // 8, W // 8)

            # shape is B x 128 x H//8 x W//8
            coords = coords_grid(N, Hdown, Wdown).to(event_volume.device)
            coords = coords.permute(0, 2, 3, 1)
            coords = coords.reshape(N, -1, 2)

            coords_norm, coords_en = self.encode_positions(coords,
                                                           res=[(Wdown, Hdown) for _ in range(N)],
                                                           return_normalized=True)

            array_features = volume_en.reshape(N, -1, Hdown*Wdown).transpose(1, 2)
            if not self.fixed_pe :
                array = torch.cat([coords, coords_en, array_features], dim=-1)
            else :
                array = torch.cat([coords_norm, coords_en, array_features], dim=-1)

            if not self.iterative_queries :
                pred = torch.stack(self.perceiver(array, queries=array)).reshape(N, Hdown, Wdown, -1)
            else :
                latents = self.perceiver(array, queries=None)
                if self.iterative_queries_mode == 'recurrent' :
                    array = torch.cat([array, array_features], dim=-1)
                embedding_0 = self.perceiver.decoder_cross_attn(array, context=latents)
                flow_0 = self.perceiver.to_logits(embedding_0)

                if self.gru_updates :
                    pass
                    cnet = self.volume_encoder_2(event_volume)
                    net, inp = torch.split(cnet, [self.hdim_encoder, self.cdim], dim=1)
                    net = torch.tanh(net)  # net is hidden state
                    inp = torch.relu(inp)


                for i in range(self.num_iterative_queries) :
                    coords_i = coords + flow_0
                    coords_t_pe = torch.cat(self.encode_positions(coords_i,
                                                                  res=[(Wdown, Hdown) for _ in range(N)],
                                                                  return_normalized=True),
                                            dim=-1)
                    features_sampled = bilinear_sampler(volume_en, coords_i.reshape(N, Hdown, Wdown, 2))
                    features_sampled = features_sampled.reshape(N, -1, Hdown*Wdown).transpose(-2, -1)


                    if self.iterative_queries_mode == 'separate_with_embedding' : # TODO: lukas mode
                        iter_query = torch.cat([coords_t_pe, features_sampled, array_features, embedding_0], dim=-1)
                        # current positions, current featuere, old feature, old embedding

                        iter_embedding = self.decoder_cross_attn_refine(iter_query, context=latents)
                        flow_i = self.refine_to_logits(iter_embedding)
                    elif self.iterative_queries_mode == 'separate_without_embedding' : # do not re use embedding
                        iter_query = torch.cat([coords_t_pe, features_sampled, array_features], dim=-1)
                        # current positions, current featuere, old feature, old embedding

                        iter_embedding = self.decoder_cross_attn_refine(iter_query, context=latents)
                        flow_i = self.refine_to_logits(iter_embedding)

                    elif self.iterative_queries_mode == 'recurrent' :
                        iter_query = torch.cat([coords_t_pe, features_sampled, array_features], dim=-1)
                        iter_embedding = self.perceiver.decoder_cross_attn(iter_query, context=latents)
                        if self.perceiver.decoder_ff is not None :
                            iter_embedding = self.perceiver.decoder_ff(iter_embedding)
                        flow_i = self.perceiver.to_logits(iter_embedding)

                    if self.gru_updates :
                        # flow_i is correlation embedding
                        # TODO reshape flow0
                        corr = flow_i.transpose(-1, -2).reshape(N, -1, Hdown, Wdown) # TODO: reshape
                        flow = flow_0.transpose(-1, -2).reshape(N, -1, Hdown, Wdown)
                        net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

                    flow_0 += flow_i

                pred = flow_0
                pred = pred.reshape(N, Hdown, Wdown, 2)

                if self.res_fixed :
                    pred *= Wdown / self.res_fixed[0]

                if not self.convex_upsample :
                    raise NotImplementedError("bilinear interpolation sucks anyway")


            if self.convex_upsample :
                if self.gru_updates :
                    pred_up = upsample_flow(pred.permute(0, 3, 1, 2), up_mask, H // Hdown)
                    return pred_up
                if self.volume_encoding == 'conv3d' :
                    if not self.convex_upsample_encode :
                        volume_en_context = self.volume_encoder(event_volume.unsqueeze(1))
                    else :
                        volume_en_context = self.volume_encoder_2(event_volume.unsqueeze(1))
                    mask = self.mask(volume_en_context[:, :, 0])

                    pred_up = upsample_flow(pred.permute(0, 3, 1, 2), mask, H // Hdown)
                    return pred_up
                else :
                    if not self.convex_upsample_encode:
                        volume_en_context = self.volume_encoder(event_volume)
                    else :
                        volume_en_context = self.volume_encoder_2(event_volume)

                    mask = self.mask(volume_en_context)

                    pred_up = upsample_flow(pred.permute(0, 3, 1, 2), mask, H // Hdown)
                    return pred_up



        pred = pred.permute(0, 3, 1, 2)

        pred_upsample = self.upsampler(pred)

        return pred_upsample


    def volume_encode_all_dims(self, event_volume) :
        N, D, T, H, W = event_volume.shape

        coords = coords_grid(N, H, W).to(event_volume.device)
        coords = coords.permute(0, 2, 3, 1)
        coords = coords.reshape(N, -1, 2)

        coords_norm, coords_en = self.encode_positions(coords,
                                                       res=[(W, H) for _ in range(N)],
                                                       return_normalized=True)
        ts = torch.arange(0, T, device=event_volume.device).repeat(N, 1).unsqueeze(-1)

        ts_norm, ts_en = self.encode_time(ts, torch.ones(N) * T)

        coords = torch.cat([coords_norm, coords_en], dim=-1)
        assert coords.shape == (N, H*W, 2 + 4*self.pos_bands)
        coords = coords.transpose(-1, -2)
        coords = coords.reshape(N, -1, 1, H, W)
        coords = coords.repeat(1, 1, T, 1, 1)

        ts = torch.cat([ts_norm, ts_en], dim=-1)
        assert ts.shape == (N, T, 1 + 2 * self.t_bands)
        ts = ts.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        ts = ts.repeat(1, 1, 1, H, W)

        #assert coords.shape == event_volume.shape
        #assert ts.shape == event_volume.shape

        return coords, ts



    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def encode_positions(self, locs, res, return_normalized=True, cat_normalized=False):
        res = torch.tensor(res, dtype=torch.float32)
        res = res.type_as(locs)
        locs_norm = locs / res.unsqueeze(-2) * 2 - 1

        if self.res_fixed :
            res_fixed = torch.tensor(self.res_fixed, device=locs.device, dtype=torch.float32)
            if len(res.shape) > 1 :
                res_fixed = res_fixed.repeat(res.shape[0], 1)
            res = res_fixed

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


    def encode_time(self, ts, dts, return_normalized=True) :
        if ts.shape[-1] == 0 :
            if return_normalized :
                return ts, ts
            else :
                return ts
        assert ts.shape[-1] == 1, "Require timestamps as column vectors"
        dts = torch.tensor(dts, device=ts.device, dtype=torch.float32)
        ts_norm = ts / dts[(Ellipsis, *((None,) * (len(ts.shape)- 1)))]

        if self.encoding_type == 'perceiver' :
            if ts.ndim == 2 :
                ts_en = fourier_encode(ts_norm[:, 0], self.t_dims, num_bands=self.t_bands,
                                       cat_orig=False)
            else :
                ts_en = torch.cat([fourier_encode(ts_norm[[i], ..., 0],
                                                  self.t_dims, num_bands=self.t_bands,
                                                  cat_orig=False)
                                   for i in range(ts.shape[0])],
                                dim=0)

        elif self.encoding_type == 'random_fourier_features' or self.encoding_type == 'learned_fourier_features' :
            ts_en = gaussian_encoding(ts_norm, self.W_t, sigma=1.)

        elif self.encoding_type == 'transformer' :
            ts_en = sinusoidal_encoding(ts, self.t_bands)

        if return_normalized :
            return ts_norm, ts_en
        else :
            return ts_en

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