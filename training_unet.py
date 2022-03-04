import torch
from ERAFT.model.utils import bilinear_sampler

from training_eraft import eval_volume_custom

def forward_unet(model, sample, dataset, **kwargs) :
    return model(torch.stack(sample['event_volume_new']))

def eval_unet(pred_dense, sample, lfunc, **kwargs) :
    return eval_volume_custom(pred_dense, sample, lfunc)