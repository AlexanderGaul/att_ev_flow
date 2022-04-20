
import torch

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO

from utils import gaussian_encoding, sinusoidal_encoding


class PositionEncoding(torch.nn.Module) :
    def __init__(self, d_in, num_bands) :
        super().__init__()
        self.W = torch.nn.Linear(d_in, num_bands)

    def forward(self, X:torch.tensor) :
        pass


class EventTransformer(torch.nn.Module) :

    # Ticket 005 - week 08 - WAITING FOR DEPLOYMENT
    # TODO: enable input modification into model when scaling/cropping/subsampling
    # [x] supply a resolution as model parameter that is being forced during evaluation
    #    - names res, fixed_res, res_fixed
    # [-] needs to scale all inputs accordingly, should we also re-scale the outputs?
    #    - use fixed res if set in model, scale the inputs and gt in training
    
    def __init__(self, pos_bands=64, time_bands=8, depth=16, perceiver_params=None,
                 res_fixed=None,
                 final_layer_init_range=5, output_scale=1.,
                 encoding_type='perceiver',
                 ff_weight_scale=1.,
                 ff_init_normal=True,
                 input_format=None,
                 t_bins=100,
                 transformer_encoder=False,
                 input_as_query=False) :
        super().__init__()
        if perceiver_params is None : perceiver_params = dict()

        self.pos_bands = pos_bands
        self.t_bands = time_bands

        self.input_format = input_format
        if self.input_format is None :
            self.input_format = {'xy' : [0, 1],
                               't' : [2],
                               'p' : [3],
                               'raw' : []}
        self.t_bins = t_bins

        self.transformer_encoder = transformer_encoder

        self.input_as_query=input_as_query

        # TODO: how to store input format
        # what if key does not exist in dict,
        # assume all have to exist?
        # check if functions work with empty arrays?
        dim = (2 * self.pos_bands * len(self.input_format['xy']) +
              len(self.input_format['xy']) +
              2 * self.t_bands * len(self.input_format['t']) +
              len(self.input_format['t']) +
              len(self.input_format['p']) +
              len(self.input_format['raw']))

        self.perceiver_params = dict(
            dim = dim,                    # dimension of sequence to be encoded
            queries_dim = 4 * self.pos_bands + 2 if not self.input_as_query else dim,            # dimension of decoder queries
            logits_dim = 2,             # dimension of final logits
            depth = depth,                   # depth of net
            num_latents = 128,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 256 if not transformer_encoder else dim,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 4,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False,    # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff = True,
            transformer_encoder = transformer_encoder
        )
        self.perceiver_params.update(perceiver_params)

        self.model = PerceiverIO(**self.perceiver_params)

        self.res_fixed = res_fixed

        torch.nn.init.uniform_(self.model.to_logits.weight, -final_layer_init_range, final_layer_init_range)
        torch.nn.init.uniform_(self.model.to_logits.bias, -0.5, 0.5)

        if self.model.no_query_return != "latents" :
            torch.nn.init.uniform_(self.model.latents_to_logits.weight, -final_layer_init_range, final_layer_init_range)
            torch.nn.init.uniform_(self.model.latents_to_logits.bias, -0.5, 0.5)

        self.output_scale = output_scale

        self.encoding_type = encoding_type
        if (self.encoding_type == 'random_fourier_features' or
                self.encoding_type == 'learned_fourier_features') :
            self.ff_weight_scale = ff_weight_scale
            requires_grad = self.encoding_type == 'learned_fourier_features'
            if ff_init_normal :
                self.W_coords = torch.nn.Parameter(torch.randn((self.pos_bands * 2, 2)) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                self.W_t = torch.nn.Parameter(torch.randn((self.t_bands, 1)) * self.ff_weight_scale,
                                              requires_grad=requires_grad)
            else :
                self.W_coords = torch.nn.Parameter((torch.rand((self.pos_bands * 2, 2)) * 2 - 1) * self.ff_weight_scale,
                                                   requires_grad=requires_grad)
                self.W_t = torch.nn.Parameter((torch.rand((self.t_bands, 1)) * 2 - 1) * self.ff_weight_scale,
                                              requires_grad=requires_grad)


        #torch.nn.init.zeros_(self.model.to_logits.weight)
        #torch.nn.init.zeros_(self.model.to_logits.bias)

        #print("Event Transformer initialized with zeros and scale 0.01")

        print(self.perceiver_params)

    # TODO: check for batches
    def encode_positions(self, locs, res, return_normalized=True, cat_normalized=False) :
        res = self.res_fixed if self.res_fixed else res
        res = torch.tensor(res, device=locs.device, dtype=torch.float32)
        locs_norm = locs / res.unsqueeze(-2) * 2 - 1

        if self.encoding_type == 'perceiver' :
            if locs.ndim == 2 :
                assert res.shape == torch.Size([2])
                x_en = fourier_encode(locs_norm[..., 0], res[0], num_bands=self.pos_bands, cat_orig=False)
                y_en = fourier_encode(locs_norm[..., 1], res[1], num_bands=self.pos_bands, cat_orig=False)
                locs_en = torch.cat([x_en, y_en], dim=-1)
            else :
                if res.ndim == 1 :
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

        elif self.encoding_type == 'random_fourier_features' or self.encoding_type == 'learned_fourier_features' :
            locs_en = gaussian_encoding(locs_norm,
                                        self.W_coords, sigma=1.)

        elif self.encoding_type == 'transformer' :
            x_en = sinusoidal_encoding(locs[..., [0]], self.pos_bands)
            y_en = sinusoidal_encoding(locs[..., [1]], self.pos_bands)
            locs_en = torch.cat([x_en, y_en], dim=-1)

        if return_normalized :
            return locs_norm, locs_en
        elif cat_normalized :
            return torch.cat([locs_norm, locs_en], dim=-1)
        else :
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
                ts_en = fourier_encode(ts_norm[:, 0], self.t_bins, num_bands=self.t_bands,
                                       cat_orig=False)
            else :
                ts_en = torch.cat([fourier_encode(ts_norm[[i], ..., 0],
                                                  self.t_bins, num_bands=self.t_bands,
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


    def encode_events(self, events, res, dt):
        coords, coords_en = self.encode_positions(events[..., self.input_format['xy']], res, True)
        ts, ts_en = self.encode_time(events[..., self.input_format['t']], dt, True)
        pol_data = events[..., self.input_format['p']]
        raw_data = events[..., self.input_format['raw']]
        assert coords.shape[-1] +  ts.shape[-1] + pol_data.shape[-1] + raw_data.shape[-1] == events.shape[-1]
        return torch.cat([coords, ts, pol_data, raw_data,
                          coords_en, ts_en],
                          dim=-1)

    def encode_event_input(self, event_data, res, dt) :
        if type(event_data) is not list :
            event_input = self.encode_events(event_data, res, dt)
        else :
            event_input = [self.encode_events(event_frame, res[i], dt[i])
                           for i, event_frame in enumerate(event_data)]
        return event_input

    def encode_query_input(self, query_locs, res) :
        if query_locs is None :
            query = None
        elif type(query_locs) is list :
            query = [self.encode_positions(q, res[i], False, True) for i, q in enumerate(query_locs)]
        elif query_locs.ndim > 2 :
            query = torch.stack([self.encode_positions(q, res[i], False, True) for i, q in enumerate(query_locs)])
        else :
            query = self.encode_positions(query_locs, res, False, True)
        return query

    
    def forward(self, event_data, query_locs, res, dt, mask=None, tbins=None) :
        event_input = self.encode_event_input(event_data, res, dt)
        if query_locs is None and self.input_as_query :
            query = event_input
        else :
            query = self.encode_query_input(query_locs, res)


        pred = self.model(event_input, mask=mask, queries=query)

        scale = self.output_scale
        # scale = torch.tensor([640, 480], device=next(self.parameters()).device)

        if type(pred) is list :
            return [p * scale for p in pred]
        else :
            return pred * scale


        return pred