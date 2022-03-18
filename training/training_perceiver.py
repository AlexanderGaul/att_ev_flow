import numpy as np
import torch

from utils import collate_dict_list, get_grid_coordinates

from plot import create_event_picture


# Ticket 005 - week 08 - WAITING FOR DEPLOYMENT
# TODO: enable input modification into model when scaling/cropping/subsampling
# [x] scale inputs to meet model resolution
# [x] scale predictions to be usable with ground truth later on
# [x] could have different resolutions for different samples in batch
# [~] need to work out resolution conventions
#    - could use y, x for actual images and x, y for listed coordinates
#    - could use a diffrent name for those listed coordinates
#    - coord_bounds, xy_bounds,


class EventPerceiverTrainer :
    def __init__(self, dataset, full_query=False, predict_targets=False) :
        self.full_query = full_query
        self.predict_targets = predict_targets
        self.dataset = dataset
    
    def sample_to_device(self, sample) :
        pass

    def forward(self, model, sample) :
        return forward_perceiver(model, sample, self.dataset,
                                 self.full_query, self.predict_targets)
    
    def evaluate(self, sample, out, lfunc) :
        eval_perceiver_out(out, sample lfunc, self.full_query)



def forward_perceiver(model, sample, dataset, full_query=False, predict_targets=False, **kwargs) :
    events_torch = sample['events']

    device = next(model.parameters()).device

    if full_query :
        # TODO: the dataset can generate and distort these
        # this is not correct probably
        # could link dataset and compute here on the fly
        # otherwise have to tell the dataset when to compute these
        query_torch = torch.stack(len(events_torch) *
                                  [torch.tensor(
                                      get_grid_coordinates(
                                          dataset.res, (0, 0)),
                                      dtype=torch.float32, device=device)])
    else:
        query_torch = sample['coords']

    if model.res_fixed :
        scales = torch.tensor([[model.res_fixed[0] / res[0],
                                model.res_fixed[1] / res[1]]
                               for res in sample['res']])
        for i, e in enumerate(events_torch) :
            e[:, :2] *= scales[i]
        for i, q in enumerate(query_torch) :
            q *= scales[i]

    forward_begin = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_begin.record()
    pred = model(events_torch, query_torch,
                 sample['res'], sample['dt'], sample['tbins'])
    forward_end.record()

    if predict_targets:
        for i in range(len(pred)):
            pred[i] = pred[i] - query_torch[i]

    if model.res_fixed :
        for i, p in enumerate(pred) :
            p /= scales[i]

    return (pred, query_torch) #, (forward_begin, forward_end)


def eval_perceiver_out(out,
                       sample, lfunc, full_query=False, **kwargs) :
    pred, query_torch = out

    if full_query:
        pred = [predi[sample['coords_grid_idx'][i], :]
                for i, predi in enumerate(pred)]

    flows_torch = sample['flows']

    loss = torch.cat([lfunc(pred[i],
                            flows_torch[i]).reshape(1)
                      for i in range(len(pred))]).nansum()

    return (loss,
            [predi.detach() for predi in pred],
            [f.detach() for f in flows_torch],
            sample['coords'])
