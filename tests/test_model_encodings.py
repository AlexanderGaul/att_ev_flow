import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model import EventTransformer


#model_perceiver_encoding = EventTransformer(encoding_type='perceiver', pos_bands=16, time_bands=4)
model_rff_encoding = EventTransformer(encoding_type='random_fourier_features', pos_bands=16, time_bands=4)
model_lff_encoding = EventTransformer(encoding_type='learned_fourier_features', pos_bands=16, time_bands=4)
model_transformer_encoding = EventTransformer(encoding_type='transformer', pos_bands=16, time_bands=4)

model_types = [#model_perceiver_encoding,
               model_rff_encoding,
               model_lff_encoding,
               model_transformer_encoding]


ex1 = torch.randint(0, 64, (1, 20, 1))
ey1 = torch.randint(0, 32, (1, 20, 1))
t1 = torch.randint(0, 100, (1, 20, 1))
p1 = torch.randint(0, 2, (1, 20, 1))

e1 = torch.cat([ex1, ey1, t1, p1], dim = -1)


ex2 = torch.randint(0, 32, (1, 20, 1))
ey2 = torch.randint(0, 16, (1, 20, 1))
t2 = torch.randint(0, 50, (1, 20, 1))
p2 = torch.randint(0, 2, (1, 20, 1))

e2 = torch.cat([ex2, ey2, t2, p2], dim=-1)

e_batch1 = torch.cat([e1, e2], dim=0)


ex12 = torch.randint(0, 48, (1, 20, 1))
ey12 = torch.randint(0, 24, (1, 20, 1))
t12 = torch.randint(0, 25, (1, 20, 1))
p12 = torch.randint(0, 2, (1, 20, 1))

e12 = torch.cat([ex12, ey12, t12, p12], dim = -1)


ex22 = torch.randint(0, 32, (1, 20, 1))
ey22 = torch.randint(0, 16, (1, 20, 1))
t22 = torch.randint(0, 75, (1, 20, 1))
p22 = torch.randint(0, 2, (1, 20, 1))

e22 = torch.cat([ex22, ey22, t22, p22], dim=-1)

e_batch2 = torch.cat([e12, e22], dim=0)


res1 = [(64, 32), (32, 16)]
res2 = [(48, 24),(32, 16)]
dt1 = [100, 50]
dt2 = [25, 75]


for model in model_types :
    print(model.encoding_type)
    print("batch list")
    en_batch_list = model.encode_event_input([e_batch1, e_batch2], [res1, res2],  [dt1, dt2])
    print("batch")
    en_batch = model.encode_event_input(e_batch1, res1, dt1)
    print("list flat")
    en_list_flat = torch.stack(model.encode_event_input([b for b in e_batch1], res1,  dt1), dim=0)
    print("list batch dimension")
    en_list = torch.cat(model.encode_event_input([b.unsqueeze(0) for b in e_batch1], res1,  dt1), dim=0)

    print((en_batch_list[0] - en_batch).abs().max())
    print((en_list - en_batch).abs().max())
    print((en_list_flat - en_list).abs().max())

    assert (en_batch_list[0] == en_batch).all()
    assert (en_batch == en_list_flat).all()
    assert (en_list_flat == en_list).all()
