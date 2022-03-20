
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
    
    def __init__(self, pos_bands=64, time_bands=8, depth=16, perceiver_params=dict(),
                 res_fixed=None,
                 final_layer_init_range=5, output_scale=1.,
                 encoding_type='perceiver',
                 input_format={'xy' : [0, 1],
                               't' : [2],
                               'raw' : [3]},
                 t_bins=100) :
        super().__init__()

        self.pos_bands = pos_bands
        self.t_bands = time_bands

        self.input_format = input_format
        self.t_bins = t_bins

        # TODO: how to store input format
        # what if key does not exist in dict,
        # assume all have to exist?
        # check if functions work with empty arrays?
        dim = (2 * self.pos_bands * len(self.input_format['xy']) +
              len(self.input_format['xy']) +
              2 * self.t_bands * len(self.input_format['t']) +
              len(self.input_format['t']) +
              len(self.input_format['raw']))

        self.perceiver_params = dict(
            dim = dim,                    # dimension of sequence to be encoded
            queries_dim = 4 * self.pos_bands + 2 ,            # dimension of decoder queries
            logits_dim = 2,             # dimension of final logits
            depth = depth,                   # depth of net
            num_latents = 128,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 256,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 4,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False,    # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff = True
        )
        self.perceiver_params.update(perceiver_params)

        self.model = PerceiverIO(**self.perceiver_params)

        self.res_fixed = res_fixed

        torch.nn.init.uniform_(self.model.to_logits.weight, -final_layer_init_range, final_layer_init_range)
        torch.nn.init.uniform_(self.model.to_logits.bias, -0.5, 0.5)

        self.output_scale = output_scale

        self.encoding_type = encoding_type
        if encoding_type == 'random_fourier_features' :
            self.W_coords = torch.nn.Parameter(torch.randn((self.pos_bands * 2, 2)),
                                               requires_grad=False)
            self.W_t = torch.nn.Parameter(torch.randn((self.t_bands, 1)),
                                          requires_grad=False)

        #torch.nn.init.zeros_(self.model.to_logits.weight)
        #torch.nn.init.zeros_(self.model.to_logits.bias)

        #print("Event Transformer initialized with zeros and scale 0.01")

        print(self.perceiver_params)

    # TODO: check for batches
    def encode_positions(self, locs, res, cat_orig=True) :
        res = self.res_fixed if self.res_fixed else res
        if self.encoding_type == 'perceiver' :
            x_en = fourier_encode(locs[..., 0] / (res[0]-1) * 2. - 1., res[0]-1,
                                  num_bands=self.pos_bands, cat_orig=cat_orig)
            y_en = fourier_encode(locs[..., 1] / (res[1]-1) * 2. - 1., res[1]-1,
                                  num_bands=self.pos_bands, cat_orig=cat_orig)
            return torch.cat([x_en, y_en], dim=-1)
        elif self.encoding_type == 'random_fourier_features' :
            locs_en = gaussian_encoding(locs /
                                        torch.tensor(res,
                                                     device=next(self.parameters()).device) * 2 - 1,
                                        self.W_coords)
            if cat_orig :
                return torch.cat([locs_en, locs], axis=-1)
            else :
                return locs_en
        elif self.encoding_type == 'paper_equation' :
            x_en = sinusoidal_encoding(locs[..., [0]], self.pos_bands)
            y_en = sinusoidal_encoding(locs[..., [1]], self.pos_bands)
            if cat_orig :
                locs_en = torch.cat([x_en, y_en,
                                  locs[..., [0]] / res[0] * 2 - 1,
                                  locs[..., [1]] / res[1] * 2 - 1], axis=-1)
                return locs_en
            else :
                return torch.cat([x_en, y_en], axis=-1)


    def encode_time(self, ts, dt) :
        if len(ts.reshape(-1)) == 0 :
            return ts
        if self.encoding_type == 'perceiver' :
            ts_en = torch.cat([#ts,
                                fourier_encode(ts[..., 0] / dt * 2. - 1., self.t_bins, num_bands=self.t_bands)],
                                dim=1)
            return ts_en
        elif self.encoding_type == 'random_fourier_features' :
            ts_en = gaussian_encoding(ts / dt * 2. - 1., self.W_t, )
            return torch.cat([ts_en, ts], axis=-1)
        # TODO: add 'paper_equation' encoding for time


    def encode_events(self, events, res, dt):
        return torch.cat([events[..., self.input_format['raw']],
                          self.encode_time(events[..., self.input_format['t']], dt),
                          self.encode_positions(events[..., self.input_format['xy']], res)],
                          dim=-1)
    
    def forward(self, event_data, query_locs, res, dt, tbins=None) :
        if type(event_data) is not list and event_data.ndim == 2 :
            event_data = [event_data]
        event_input = [self.encode_events(event_frame, res[i], dt[i])
                       for i, event_frame in enumerate(event_data)]
        if type(event_data) is not list :
            event_input = torch.stack(event_input)

        if type(query_locs) is list :
            query = [self.encode_positions(q, res[i]) for i, q in enumerate(query_locs)]
        elif query_locs.ndim > 2 :
            query = torch.stack([self.encode_positions(q, res[i]) for i, q in enumerate(query_locs)])
        else :
            query = self.encode_positions(query_locs, res[0])

        pred = self.model(event_input, queries = query)

        # TODO change tensor construction
        #scale = torch.FloatTensor([self.res[0], self.res[1]]).to(next(self.parameters()).device)
        #scale = 40.
        scale = self.output_scale
        # scale = torch.tensor([640, 480], device=next(self.parameters()).device)

        if type(pred) is list :
            return [p * scale for p in pred]
        else :
            return pred * scale


        return pred