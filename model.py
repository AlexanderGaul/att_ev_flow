
import torch

from perceiver_pytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiver_pytorch.perceiver_pytorch.perceiver_io import PerceiverIO



class EventTransformer(torch.nn.Module) :
    
    def __init__(self, res=(346, 260), pos_bands=64, time_bands=8, depth=16, perceiver_params=dict()) :
        super().__init__()

        self.pos_bands = pos_bands
        self.t_bands = time_bands

        self.perceiver_params = dict(
            dim = 4 * self.pos_bands + 2 * self.t_bands + 4,                    # dimension of sequence to be encoded
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

        torch.nn.init.uniform_(self.model.to_logits.weight, -5, 5)
        torch.nn.init.uniform_(self.model.to_logits.bias, -0.5, 0.5)

        print(self.perceiver_params)

    # TODO: check for batches
    def encode_positions(self, locs, res) :
        x_en = fourier_encode(locs[:, 0] / res[0] * 2. - 1., res[0], num_bands=self.pos_bands)
        y_en = fourier_encode(locs[:, 1] / res[1] * 2. - 1., res[1], num_bands=self.pos_bands)

        return torch.cat([x_en, y_en], dim=1)

    def encode_time(self, ts, dt, tbins) :
         ts_en = torch.cat([#ts,
                            fourier_encode(ts[:, 0] / dt * 2. - 1., tbins, num_bands=self.t_bands)],
                            dim=1)

         return ts_en

    def encode_events(self, events, res, dt, tbins):
        return torch.cat([events[:, [3]],
                          self.encode_time(events[:, [2]], dt, tbins),
                          self.encode_positions(events[:, :2], res)],
                          dim=1)
    
    def forward(self, event_data, query_locs, res, dt, tbins) :
        if type(event_data) is not list :
            if event_data.ndim > 2 :
                event_data = event_data.squeeze(0)
            event_data = [event_data]

        event_input = [self.encode_events(event_frame, res[i], dt[i], tbins[i])
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
        scale = 1.
        # scale = torch.tensor([640, 480], device=next(self.parameters()).device)

        if type(pred) is list :
            return [p * scale for p in pred]
        else :
            return pred * scale


        return pred