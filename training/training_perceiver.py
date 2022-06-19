import numpy as np
import torch


from model import EventTransformer
from training.training_interface import AbstractTrainer
from training.training_report import compute_statistics, paint_pictures_evaluation_arrrays, cat_images
from utils import get_grid_coordinates
from data.utils import collate_dict_list, tensor_or_tensorlist_to_device

from plot import create_event_picture_format, put_text


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
                 predict_length=False,
                 classify_LR=False,
                 predict_uni=False,
                 no_query=False) :
        self.lfunc = lfunc
        self.full_query = full_query
        assert not self.full_query, "needs to be re-implemented"
        self.predict_targets = predict_targets
        self.predict_length = predict_length
        assert not predict_length
        self.classify_LR = classify_LR
        self.predict_uni = predict_uni
        self.no_query = self.classify_LR or self.predict_uni or no_query

    def batch_size(self, sample) :
        return len(sample['event_array'])
    
    def sample_to_device(self, sample, device) :
        tensor_keys = ['event_array', 'flow_array', 'coords']
        if 'coords_mask' in sample :
            tensor_keys.append('coords_mask')
        if 'event_array_pad_mask' in sample :
            tensor_keys.append('event_array_pad_mask')
        return tensor_or_tensorlist_to_device(sample, device, tensor_keys)

    def forward(self, model:EventTransformer, sample) :

        if model.res_fixed:
            scales = torch.tensor([[model.res_fixed[0] / res[0],
                                    model.res_fixed[1] / res[1]]
                                   for res in sample['res']],
                                  device=sample['event_array'].device)
            for i, e in enumerate(sample['event_array']):
                e[:, model.input_format['xy']] *= scales[i]
            for i, q in enumerate(sample['coords']):
                q *= scales[i]


        forward_begin = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_begin.record()
        pred = model(sample['event_array'], sample['coords'] if (not self.no_query) else None,
                     sample['res'], sample['dt'],
                     mask=sample['event_array_pad_mask'] if 'event_array_pad_mask' in sample else None) # , sample['tbins'])
        forward_end.record()

        if self.predict_targets :
            for i in range(len(pred)):
                pred[i] = pred[i] - sample['coords'][i]

        if model.res_fixed and not self.classify_LR :
            for i, p in enumerate(pred):
                p /= scales[i]

        if self.predict_length :
            for i in range(len(pred)) :
                pred[i] = torch.nn.functional.normalize(pred[i])

        if model.res_fixed:
            for i, e in enumerate(sample['event_array']):
                e[:, model.input_format['xy']] /= scales[i]
            for i, q in enumerate(sample['coords']):
                q /= scales[i]

        # TODO: refcactor this hack so that each function takes the model as input as well
        return {'pred' : pred, 'format' : model.input_format,
                'torch_cuda_events' : (forward_begin, forward_end)}


    def evaluate(self, sample, out) :
        if self.classify_LR or self.predict_uni or self.predict_length :
            loss_items = self.evaluate_var(sample, out)
        elif 'coords_mask' in sample :
            loss_items = torch.cat([self.lfunc(out['pred'][i][sample['coords_mask'][i]],
                                               sample['flow_array'][i]).reshape(-1)
                                    for i in range(len(out['pred']))])
        else :
            loss_items = torch.cat([self.lfunc(out['pred'][i],
                                         sample['flow_array'][i]).reshape(-1)
                              for i in range(len(out['pred']))])
        loss = loss_items.nanmean()
        if torch.isnan(loss) :
            print("Loss is NAN")
            print(sample['frame_id'])
            loss = torch.zeros(loss.shape, requires_grad=True, device=loss.device)
        return {'loss' : loss}


    def statistics(self, sample, out, eval, return_separate=False) :
        pred = out['pred']
        if 'coords_mask' in sample :
            pred = [pred[i][sample['coords_mask'][i]] for i in range(len(pred))]
        return compute_statistics(pred, sample['flow_array'], return_separate)

    def visualize(self, visualize_frame_ids, sample, out, eval) :
        report = dict()
        if not visualize_frame_ids or len(visualize_frame_ids) == 0 :
            return report
        for i, id in enumerate(sample['frame_id']) :
            if list(id[:2]) in visualize_frame_ids :
                report.update(self.visualize_item(i, sample, out, eval,
                                                  write_label_on_img=True, concat=True))
        return report

    # TODO: also need output can not decide from outside what indices mean
    def visualize_item(self, idx, sample, out, eval,
                       write_label_on_img=False, concat=False) :
        event_array = sample['event_array'][idx].detach().cpu().numpy()
        report = paint_pictures_evaluation_arrrays(
            create_event_picture_format(event_array, out['format'],
                                        res=np.flip(sample['res'][idx]) if 'res' in sample
                                        else sample['events'][idx].shape[1:3]),
            sample['coords'][idx].detach().cpu().numpy(),
            out['pred'][idx].detach().cpu().numpy(),
            sample['flow_array'][idx].detach().cpu().numpy(),
            str(sample['frame_id'][idx]), report=dict(),
            pred_mask=sample['coords_mask'][idx].detach().cpu().numpy() if 'coords_mask' in sample else None,
            flow_frame=sample['flow_frame'][idx].numpy(), flow_frame_valid=sample['flow_frame_valid'][idx].numpy())
        if write_label_on_img :
            if not concat :
                for k in report.keys() :
                    report[k] = put_text(report[k], [sample['path'][idx], k])
            else :
                for k in report.keys():
                    report[k] = put_text(report[k], [k])
        if concat :
            im_cat = cat_images(list(report.values()))
            report = {str(sample['frame_id'][idx]) : im_cat}
        return report


    def evaluate_var(self, sample, out) :
        if self.classify_LR :
            scores = out['pred']
            probs = torch.softmax(scores, dim=-1)
            device = device=probs[0].device
            x_mean = [flowi[:, 0].mean() for flowi in sample['flow_array']]
            labels = [torch.tensor([0], device=device) if xi < 0 else
                      torch.tensor([1], device=device) for xi in x_mean]
            loss_items = torch.cat([torch.nn.CrossEntropyLoss()(scores[i].reshape(1, -1),
                                                          labels[i]).reshape(-1)
                              for i in range(len(probs))])
            out['pred'] = [torch.tensor([probs[i, ..., 1] * 2 - 1, 0.], device=device).repeat(len(sample['flow_array'][i]), 1)
                           for i in range(len(probs))]
        elif self.predict_uni :
            flow_mean = [flowi.mean(dim=-2, keepdim=True) for flowi in sample['flow_array']]
            loss_items = torch.cat([self.lfunc(out['pred'][i],
                                         flow_mean[i]).reshape(-1)
                              for i in range(len(out['pred']))])
            out['pred'] = [out['pred'][i].repeat(len(sample['flow_array'][i]), 1)
                           for i in range(len(out['pred']))]
        elif self.predict_length :
            flows_norm = [torch.nn.functional.normaliez(flowi)
                          for flowi in sample['flow_array']]
            loss_items = torch.cat([self.lfunc(out['pred'][i],
                                         flows_norm[i]).reshape(-1)
                              for i in range(len(out['pred']))])
        return loss_items






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
