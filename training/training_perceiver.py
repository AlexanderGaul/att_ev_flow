import numpy as np
import torch


from model import EventTransformer
from training.training_interface import AbstractTrainer
from training.training_report import compute_statistics, paint_pictures
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


class EventPerceiverTrainer(AbstractTrainer) :
    def __init__(self, lfunc,
                 full_query=False,
                 predict_targets=False,
                 predict_length=False) :
        self.lfunc = lfunc
        self.full_query = full_query
        assert not self.full_query, "needs to be re-implemented"
        self.predict_targets = predict_targets
        self.predict_length = predict_length

    def batch_size(self, sample) :
        if type(sample['events']) is not list :
            return None
        else :
            return len(sample['events'])
    
    def sample_to_device(self, sample, device) :
        sample['events'] = [eventi.to(device) for eventi in sample['events']]
        sample['flows'] = [flowi.to(device) for flowi in sample['flows']]
        sample['coords'] = [coordi.to(device) for coordi in sample['coords']]
        return sample

    def forward(self, model:EventTransformer, sample) :
        if model.res_fixed:
            scales = torch.tensor([[model.res_fixed[0] / res[0],
                                    model.res_fixed[1] / res[1]]
                                   for res in sample['res']])
            for i, e in enumerate(sample['events']):
                e[:, :2] *= scales[i]
            for i, q in enumerate(sample['events']):
                q *= scales[i]

        forward_begin = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_begin.record()
        pred = model(sample['events'], sample['coords'],
                     sample['res'], sample['dt'], sample['tbins'])
        forward_end.record()

        if self.predict_targets :
            for i in range(len(pred)):
                pred[i] = pred[i] - sample['coords'][i]
        if model.res_fixed:
            for i, p in enumerate(pred):
                p /= scales[i]

        if self.predict_length :
            for i in range(len(pred)) :
                pred[i] = torch.nn.functional.normalize(pred[i])

        return {'pred' : pred}
    
    def evaluate(self, sample, out) :
        if self.predict_length :
            flows_norm = [torch.nn.functional.normaliez(flowi)
                          for flowi in sample['flows']]
            loss = torch.cat([self.lfunc(out['pred'][i],
                                         flows_norm[i]).reshape(-1)
                              for i in range(len(out['pred']))]).nansum()
        else :
            loss = torch.cat([self.lfunc(out['pred'][i],
                                         sample['flows'][i]).reshape(-1)
                              for i in range(len(out['pred']))]).nansum()
        return {'loss' : loss}

    def statistics(self, sample, out, eval, fraction, report) :
        return compute_statistics(out['pred'], sample['flows'],
                                  report, fraction)

    def visualize(selfs, visualize_frames, sample, out, eval, report=dict()) :
        if not visualize_frames or len(visualize_frames) == 0 :
            return report
        for i, id in enumerate(sample['frame_id']) :
            if list(id[:2]) in visualize_frames :
                report = paint_pictures(
                    create_event_picture(sample['events'][i].detach().cpu().numpy(),
                                         res=np.flip(sample['res'][i]) if 'res' in sample
                                         else sample['events'][i].shape[1:3]),
                    sample['coords'][i].detach().cpu().numpy(),
                    out['pred'][i].detach().cpu().numpy(),
                    sample['flows'][i].detach().cpu().numpy(),
                    str(sample['frame_id'][i]), report)

        return report



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
