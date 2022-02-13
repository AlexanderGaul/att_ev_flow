import numpy as np

import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader

import time

from utils import default, collate_dict_list, nested_to_device
from plot import *



def compute_statistics(pred, flow, report, fraction) :
    for i in range(len(pred)):
        if len(pred[i]) == 0:
            continue
        pred_norm = pred[i].norm(dim=1)
        flow_norm = flow[i].norm(dim=1)
        report['stats']['flow_length']['pred'] += \
            pred_norm.nanmean().item() / fraction
        report['stats']['flow_length']['gt'] += \
            flow_norm.nanmean().item() / fraction
        report['stats']['error_metrics']['cosine'] += \
            (1 - ((pred[i] * flow[i]).sum(dim=1) /
                  (pred_norm * flow_norm)).abs()).nanmean().item() / fraction
        report['stats']['error_metrics']['length'] += \
            (pred_norm - flow_norm).abs().nanmean().item() / fraction
        report['stats']['error_metrics']['l2-loss'] += \
            (pred[i] - flow[i]).norm(dim=1).nanmean().item() / fraction
    return report


def paint_pictures(E_im, coords, pred, flows, label, report) :
    im = get_np_plot_flow(
        E_im,
        coords,
        pred,
        flow_res=E_im.shape[:2])
    report['ims'][label] = im

    im_gt = get_np_plot_flow(
        E_im,
        coords,
        flows,
        flow_res=E_im.shape[:2])
    report['ims'][label + "/gt"] = im_gt

    im_color = plot_flow_color(coords, pred, res=np.flip(E_im.shape[:2]))
    report['ims'][label + "/color"] = im_color

    im_error = plot_flow_error(coords, pred, flows,
                               res=np.flip(E_im.shape[:2]))
    report['ims'][label + "/error"] = im_error

    return report


def visualize(vis_frames, sample, coords, pred, flows, report) :
    # TODO: make sample into batch before calling this
    if len(vis_frames) == 0 :
        return report
    is_batched = hasattr(sample['frame_id'][0], '__len__')

    if 'events' in sample.keys() :
        events = sample['events']
    else :
        events = sample['event_volume_new']

    assert is_batched
    if not is_batched :
        if sample['frame_id'][:2] in vis_frames :
            report = paint_pictures(
                create_event_picture(events.detach().cpu().numpy(),
                                     res=np.flip(sample['res']) if 'res' in sample
                                    else events.shape[1:3]),
                coords[0].detach().cpu().numpy(), pred[0].detach().cpu().numpy(), flows[0].detach().cpu().numpy(),
                str(sample['frame_id'][0]) + "_" + str(sample['frame_id'][1]),
                report)
        return report

    for i, id in enumerate(sample['frame_id']) :
        if tuple(id[:2]) in vis_frames :
            report = paint_pictures(
                create_event_picture(events[i].detach().cpu().numpy(),
                                     res=np.flip(sample['res'][0]) if 'res' in sample
                                     else events[i].shape[1:3]),
                coords[i].detach().cpu().numpy(), pred[i].detach().cpu().numpy(), flows[i].detach().cpu().numpy(),
                str(sample['frame_id'][i][0]) + "_" + str(sample['frame_id'][i][1]),
                report)

    return report


def process_epoch(epoch, model, LFunc, dataset, device,
                  forward_model_fun, eval_model_out_fun,
                  dataloader:DataLoader=None, optimizer=None, vis_frames=[],
                  **kwargs) :
    data = default(dataloader, dataset)
    report = {'stats' : {
                  'loss' : 0.,
                  'flow_length': {'pred': 0.,
                                  'gt': 0.},
                  'error_metrics' : {'cosine' : 0.,
                                     'length' : 0.,
                                     'l2-loss' : 0.}},
                  'runtime': {'total': 0.,
                              'forward': 0.,
                              'model': 0.,
                              'loss': 0.,
                              'backward': 0.,
                              'stats' : 0.,
                              'images' : 0.},
              'ims' : {}}

    for sample in data :
        time_begin = time.time()
        if optimizer :
            optimizer.zero_grad(set_to_none=True)

        batch_size = dataloader.batch_size if dataloader else 1

        if dataloader is None :
            sample = collate_dict_list([sample])

        sample_device = nested_to_device(sample, device)

        forward_begin = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_begin.record()
        out = forward_model_fun(model, sample_device, dataset, **kwargs)
        forward_end.record()

        loss_begin = torch.cuda.Event(enable_timing=True)
        loss_end = torch.cuda.Event(enable_timing=True)
        loss_begin.record()
        loss, pred, flows, coords = eval_model_out_fun(out, sample_device, LFunc, **kwargs)
        loss_end.record()

        backward_begin = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        backward_begin.record()
        if optimizer :
            loss /= batch_size
            loss.backward()
            optimizer.step()
        backward_end.record()

        stats_begin = time.time()
        report = compute_statistics(pred, flows, report, len(dataset))
        report['runtime']['stats'] = time.time() - stats_begin

        images_begin = time.time()
        report = visualize(vis_frames, sample, coords, pred, flows, report)
        report['runtime']['images'] = time.time() - images_begin

        torch.cuda.synchronize()
        report['runtime']['forward'] += forward_begin.elapsed_time(forward_end)
        #report['runtime']['model'] += fwd_call_ev[0].elapsed_time(fwd_call_ev[1])
        report['runtime']['loss'] += loss_begin.elapsed_time(loss_end)
        report['runtime']['backward'] += backward_begin.elapsed_time(backward_end)
        report['runtime']['total'] += time.time() - time_begin

        report['stats']['loss'] += loss.item() / len(dataset) * batch_size

    return report


def write_stats_to_tensorboard(stats:dict, writer:tensorboard.SummaryWriter,
                               it:int,
                               prefix:str="", suffix:str="") :
    for k in stats.keys() :
        if type(stats[k]) is dict :
            write_stats_to_tensorboard(stats[k], writer, it,
                                       prefix=prefix+"/"+k, suffix=suffix)
        else :
            writer.add_scalar(prefix+"/"+k+"/"+suffix, stats[k], it)