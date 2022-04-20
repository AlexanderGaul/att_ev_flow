import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import EventTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EventTransformer(pos_bands=64,
                         time_bands=64,
                         encoding_type="learned_fourier_features",
                         transformer_encoder=True,
                         perceiver_params={
                             "depth" : 8,
                             "cross_dim_head" : 16,
                             "latent_dim_head" : 16,
                             "latent_heads" : 4,
                             "cross_heads" : 4,
                             "cross_layers" : 1,
                             "no_query_return" : "logits_all"
                         })
model.to(device)
model.train()

res = (64, 64)
dt = 100

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
batch_size = 2
for num_events in [5000, 7500, 10000, 15000] :
    print(num_events)
    optimizer.zero_grad()
    events = [torch.randn((num_events, 4), device=device) for _ in range(batch_size)]
    queries = torch.randn((64 * 64, 2), device=device)

    pred = model(events, None,
                 [res for _ in range(batch_size)],
                 [dt for _ in range(batch_size)])
    loss = torch.cat([torch.nn.L1Loss()(pred[i], torch.zeros(pred[i].shape, device=device)).reshape(-1)
            for i in range(batch_size)]).sum()
    loss.backward()
    optimizer.step()


