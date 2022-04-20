import numpy as np
import torch
from torch.utils import tensorboard

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
            (1 - (((pred[i] * flow[i]).sum(dim=1) /
                  (pred_norm * flow_norm)) * 0.5 + 0.5)).nanmean().item() / fraction
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

def paint_pictures_evaluation_frames(E_im, pred_frame, flow_frame, label="",
                                     eval_mask=None, flow_frame_valid=None,
                                     include_gt=True, E_im_gt=None) :
    ims = dict()
    if include_gt :
        if E_im_gt is None : E_im_gt = E_im
        im_gt = get_np_plot_flow_grid(E_im_gt, flow_frame, flow_frame_valid)
        ims[label + label + "/events_gt"] = im_gt

    im = get_np_plot_flow_grid(E_im, pred_frame, None)
    ims[label  + "/events_pred"] = im

    if include_gt :
        im_gt_color = flow_frame_color(flow_frame, mask_valid=flow_frame_valid)
        im_gt_color = get_np_plot_flow_grid(im_gt_color, flow_frame, flow_frame_valid)
        ims[label + "/color_gt"] = im_gt_color

    im_color = flow_frame_color(pred_frame)
    im_color = get_np_plot_flow_grid(im_color, pred_frame)
    ims[label + "/color_pred"] = im_color

    if eval_mask is not None and not eval_mask.all() :
        im_color_eval = flow_frame_color(pred_frame, mask_valid=eval_mask)
        im_color_eval = get_np_plot_flow_grid(im_color_eval, pred_frame, eval_mask)
        ims[label + "/color_pred_eval"] = im_color_eval

    im_error = plot_flow_error_abs_frame(pred_frame, flow_frame, eval_mask)
    ims[label + "/abs_error"] = im_error

    return ims


def visualize_sample(data) :
    e_im = create_event_picture(data['event_array'], tuple(reversed(data['res'])))
    im = get_np_plot_flow(e_im, data['coords'], data['flow_array'], flow_res=e_im.shape[:2], freq=data['res'][0] // 10)
    return im

def visualize_sample_volume(data, freq=5) :
    e_im = create_event_frame_picture(data['event_volume'])
    im = get_np_plot_flow(e_im,
                          np.flip(np.stack(data['flow_frame_valid'].nonzero()).T, axis=-1),
                          data['flow_frame'][data['flow_frame_valid'], :], freq=freq, figsize=(e_im.shape[1] / 8,
                                                                                               e_im.shape[0] / 8))
    return im


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
    """    if not is_batched :
        if sample['frame_id'][:2] in vis_frames :
            report = paint_pictures(
                create_event_picture(events.detach().cpu().numpy(),
                                     res=np.flip(sample['res']) if 'res' in sample
                                    else events.shape[1:3]),
                coords[0].detach().cpu().numpy(), pred[0].detach().cpu().numpy(), flows[0].detach().cpu().numpy(),
                str(sample['frame_id'][0]) + "_" + str(sample['frame_id'][1]),
                report)
        return report"""

    for i, id in enumerate(sample['frame_id']) :
        if list(id[:2]) in vis_frames :
            report = paint_pictures(
                create_event_picture(events[i].detach().cpu().numpy(),
                                     res=np.flip(sample['res'][0]) if 'res' in sample
                                     else events[i].shape[1:3]),
                coords[i].detach().cpu().numpy(), pred[i].detach().cpu().numpy(), flows[i].detach().cpu().numpy(),
                str(sample['frame_id'][i]),
                report)

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
def cat_images(images) :
    height = max(map(lambda a : a.shape[0], images))
    width = np.sum(list(map(lambda a: a.shape[1], images)))
    im_cat = np.zeros((height, width, images[0].shape[-1]), dtype=images[0].dtype)
    width_acc = 0
    for im in images :
        im_cat[0:im.shape[0], width_acc:width_acc+im.shape[1]] = im
        width_acc += im.shape[1]
    return im_cat
