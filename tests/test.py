import numpy as np
import torch

from dsec import DSEC

from model import EventTransformer

from utils import nested_to_device
from data.utils import collate_dict_list

d = DSEC(bin_type='sum', num_bins=10)

model = EventTransformer(res=None,
                         pos_bands=8,
                         time_bands=8,
                         depth=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

s = collate_dict_list([d[0], d[1]])
s = nested_to_device(s, device)


s['events'][0].requires_grad = True
s['events'][1].requires_grad = True
s['coords'][0].requires_grad = True
s['coords'][1].requires_grad = True

res = model.forward(s['events'], s['coords'], s['res'], s['dt'], s['tbins'])

loss = res[0].sum()
loss.backward()

assert (s['events'][1].grad == 0).all()



print("EOF")
