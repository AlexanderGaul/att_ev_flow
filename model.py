
import torch

from perceiver_pytorch.perceiver_pytorch.perceiver_pytorch import fourier_encode
from perceiver_pytorch.perceiver_pytorch.perceiver_io import PerceiverIO



class EventTransformer(torch.nn.Module) :
    
    def __init__(self, res=(346, 260)) :
        super().__init__()
        
        self.res = res
        
        self.model = PerceiverIO(
            dim = 293,                    # dimension of sequence to be encoded
            queries_dim = 258,            # dimension of decoder queries
            logits_dim = 2,             # dimension of final logits
            depth = 8,                   # depth of net
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            weight_tie_layers = False,    # whether to weight tie layers (optional, as indicated in the diagram)
            decoder_ff = True)
    
    
    def encode_positions(self, locs) :
        return torch.cat([fourier_encode(locs[:, 0], self.res[0], num_bands=64),
                          fourier_encode(locs[:, 1], self.res[1], num_bands=64)],
                         dim=1)
    
    def encode_time(self, ts) :
        return torch.cat([ts, 
                          fourier_encode(ts[:, 0] * 10, 100., num_bands=16)],
                         dim=1)
    
    def forward(self, event_data, query_locs) :
        event_input = torch.cat([event_data[:, [3]],
                                 self.encode_time(event_data[:, [2]]),
                                 self.encode_positions(event_data[:, :2])],
                                dim=1)
        
        query = self.encode_positions(query_locs)
        
        return self.model(event_input.unsqueeze(0), queries = query.unsqueeze(0)).squeeze(0)