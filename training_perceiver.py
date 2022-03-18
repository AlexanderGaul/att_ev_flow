import numpy as np
import torch

from utils import collate_dict_list, get_grid_coordinates

from plot import create_event_picture


def forward_perceiver(model, sample, dataset, full_query, predict_targets, **kwargs) :
    events_torch = sample['events']

    device = next(model.parameters()).device

    if full_query:
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
