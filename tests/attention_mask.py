import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from perceiverpytorch.perceiver_pytorch.perceiver_io import Attention

att = Attention(16)


q = torch.randn((3, 5, 16))

mask = torch.ones((3, 5), dtype=bool)
mask[0, 2:] = False
mask[1, 3:] = False


out = att(q, mask=mask)

