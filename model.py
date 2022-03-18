
import torch

from perceiverpytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiverpytorch.perceiver_pytorch.perceiver_io import PerceiverIO

from utils import gaussian_encoding


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
        self.t_bins = 100

        # TODO: how to store input format
        # what if key does not exist in dict,
        # assume all have to exist?
        # check if functions work with empty arrays?

        self.perceiver_params = dict(
            dim = 2 * self.pos_bands * len(self.input_format['xy']) +
                  len(self.input_format['xy']) +
                  2 * self.t_bands * len(self.input_format['t']) +
                  len(self.input_format['t']) +
                  len(self.input_format['raw']),                    # dimension of sequence to be encoded
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
            x_en = fourier_encode(locs[:, 0] / (res[0]-1) * 2. - 1., res[0]-1,
                                  num_bands=self.pos_bands, cat_orig=cat_orig)
            y_en = fourier_encode(locs[:, 1] / (res[1]-1) * 2. - 1., res[1]-1,
                                  num_bands=self.pos_bands, cat_orig=cat_orig)
            return torch.cat([x_en, y_en], dim=1)
        elif self.encoding_type == 'random_fourier_features' :
            locs_en = gaussian_encoding(locs / torch.tensor(res, device=next(self.parameters()).device) * 2 - 1, self.W_coords)
            if cat_orig :
                return torch.cat([locs_en, locs], axis=-1)
            else :
                return locs_en


    def encode_time(self, ts, dt) :
        if self.encoding_type == 'perceiver' :
            ts_en = torch.cat([#ts,
                                fourier_encode(ts[:, 0] / dt * 2. - 1., self.t_bins, num_bands=self.t_bands)],
                                dim=1)
            return ts_en
        elif self.encoding_type == 'random_fourier_features' :
            ts_en = gaussian_encoding(ts.reshape(-1, 1) / dt * 2. - 1., self.W_t, )
            return torch.cat([ts_en, ts.reshape(-1, 1)], axis=-1)


    def encode_events(self, events, res, dt, tbins):
        return torch.cat([events[:, self.input_format['raw']],
                          self.encode_time(events[:, self.input_format['t']], dt, tbins),
                          self.encode_positions(events[:, self.input_format['xy']], res)],
                          dim=1)
    
    def forward(self, event_data, query_locs, res, dt, tbins=None) :
        if type(event_data) is not list :
            if event_data.ndim > 2 :
                event_data = event_data.squeeze(0)
            event_data = [event_data]

        event_input = [self.encode_events(event_frame, res[i], dt[i])
                       for i, event_frame in enumerate(event_data)]

        # TODO: want to keep the possiblity of having batched queries
        # TODO: identical query in case of querying all locations
        #if query_locs.ndim > 2 :
        #    query_locs = query_locs.squeeze(0)

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