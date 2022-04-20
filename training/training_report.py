import numpy as np
import torch
from torch.utils import tensorboard

from plot import *


def compute_statistics(pred, flow, return_separate=False) :
    report = {'flow_length': {'pred': 0., 'gt': 0.},
              'error_metrics': {'angle': 0.,
                                'length': 0.,
                                'l2-loss': 0,
                                'weighted-l1-loss' : 0.}}
    reports = []
    for i in range(len(pred)):
        if return_separate :
            report = {'flow_length': {'pred': 0., 'gt': 0.},
                      'error_metrics': {'angle': 0.,
                                        'length': 0.,
                                        'l2-loss': 0.,
                                        'weighted-l1-loss' : 0.}}

        if len(pred[i]) == 0:
            if return_separate :
                reports.append(report)
            continue
        pred_norm = pred[i].norm(dim=1)
        flow_norm = flow[i].norm(dim=1)
        report['flow_length']['pred'] += pred_norm.nanmean().item()
        report['flow_length']['gt'] += flow_norm.nanmean().item()
        report['error_metrics']['angle'] += \
            (1 - (((pred[i] * flow[i]).sum(dim=1) /
                  (pred_norm * flow_norm)) * 0.5 + 0.5)).nanmean().item()
        report['error_metrics']['length'] += \
            (pred_norm - flow_norm).abs().nanmean().item()
        report['error_metrics']['l2-loss'] += \
            (pred[i] - flow[i]).norm(dim=1).nanmean().item()
        report['error_metrics']['weighted-l1-loss'] += \
            ((pred[i] - flow[i]) / (flow[i].abs() + 1e-6)).abs().nanmean().item()

        if return_separate :
            reports.append(report)
    if not return_separate :
        return report
    else :
        return reports

def rel_errors(pred, flow) :
    errors = []
    for i in range(pred) :
        error =  pred[i]
        errors.append(error)

    return errors


# TODO: actually do not want to introduce im naming here
# TODO: remove report
# TODO: move label outsidee the function
def paint_pictures_evaluation_arrrays(E_im, coords, pred, flows, label, report,
                                      include_gt=True, pred_mask=None,
                                      flow_frame=None, flow_frame_valid=None) :
    if E_im.shape[0] <= 5 :
        figsize = (E_im.shape[1] / 16, E_im.shape[0] / 16)
        freq = min(E_im.shape[:2]) // 5
    else :
        figsize = (E_im.shape[1] / E_im.shape[0] * 5, 5)
        freq = min(E_im.shape[:2]) // 10

    if 'ims' in report :
        ims = report['ims']
    else :
        ims = report

    if pred_mask is not None:
        coords_gt = coords[pred_mask]
    else:
        coords_gt = coords

    if include_gt:
        if flow_frame is None :
            im_gt = get_np_plot_flow(
                E_im,
                coords_gt,
                flows,
                flow_res=E_im.shape[:2],
                freq=freq,
                figsize=figsize)
        else :
            im_gt = get_np_plot_flow_grid(E_im, flow_frame, flow_frame_valid, freq=freq,
                figsize=figsize)
        ims[label + "/events_gt"] = im_gt
    im = get_np_plot_flow(
        E_im,
        coords,
        pred,
        flow_res=E_im.shape[:2],
        freq=freq,
        figsize=figsize)
    ims[label + "/events_pred"] = im

    if include_gt :
        if flow_frame is None :
            im_gt_color = flow_color(coords_gt, flows, res=np.flip(E_im.shape[:2]))
            im_gt_color = get_np_plot_flow(im_gt_color, coords_gt, flows, freq=freq,
                figsize=figsize)
            ims[label + "/color_gt"] = im_gt_color
        else :
            img_gt_color = flow_frame_color(flow_frame, mask_valid=flow_frame_valid)
            im_gt_color = get_np_plot_flow_grid(img_gt_color,
                                                flow_frame, flow_frame_valid, freq=freq,
                figsize=figsize)
            ims[label + "/color_gt"] = im_gt_color


    im_color = flow_color(coords, pred, res=np.flip(E_im.shape[:2]))
    im_color = get_np_plot_flow(im_color, coords, pred, freq=freq,
                figsize=figsize)
    ims[label + "/color_pred"] = im_color
    if pred_mask is not None and not pred_mask.all() :
        im_color_eval = flow_color(coords[pred_mask], pred[pred_mask], res=np.flip(E_im.shape[:2]))
        im_color_eval = get_np_plot_flow(im_color_eval, coords[pred_mask], pred[pred_mask], freq=E_im.shape[0]//5,
                figsize=figsize)
        ims[label + "/color_pred_eval"] = im_color_eval

    if pred_mask is None :
        im_error = plot_flow_error_abs(coords, pred, flows,
                                         res=np.flip(E_im.shape[:2]),
                figsize=figsize)
    else :
        im_error = plot_flow_error_abs(coords[pred_mask], pred[pred_mask], flows,
                                         res=np.flip(E_im.shape[:2]),
                figsize=figsize)
    ims[label + "/abs_error"] = im_error

    if pred_mask is None :
        im_error = plot_flow_error_multi(coords, pred, flows,
                                         res=np.flip(E_im.shape[:2]),
                figsize=figsize)
    else :
        im_error = plot_flow_error_multi(coords[pred_mask], pred[pred_mask], flows,
                                         res=np.flip(E_im.shape[:2]),
                figsize=figsize)
    ims[label + "/multi_error"] = im_error
    if 'ims' in report:
        report['ims'] = ims
        return report
    else :
        return ims


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

    for i, id in enumerate(sample['frame_id']) :
        if list(id[:2]) in vis_frames :
            report = paint_pictures_evaluation_arrrays(
                create_event_picture(events[i].detach().cpu().numpy(),
                                     res=np.flip(sample['res'][0]) if 'res' in sample
                                     else events[i].shape[1:3]),
                coords[i].detach().cpu().numpy(), pred[i].detach().cpu().numpy(), flows[i].detach().cpu().numpy(),
                str(sample['frame_id'][i]),
                report)

    return report

def sum_stats(stats_1, stats_2) :
    res = {}
    for key in stats_1 :
        if type(stats_1[key]) is dict :
            res[key] = sum_stats(stats_1[key], stats_2[key])
        else :
            res[key] = stats_1[key] + stats_2[key]
    return res

def sum_stats_list(stats) :
    res = {}
    for key in stats[0] :
        if type(stats[0][key]) is dict :
            res[key] = sum_stats_list([s[key] for s in stats])
        else :
            res[key] = np.sum([s[key] for s in stats])
    return res

def divide_stats(stats, divisor) :
    res = {}
    for key in stats :
        if type(stats[key]) is dict :
            res[key] = divide_stats(stats[key], divisor)
        else :
            res[key] = stats[key] / divisor
    return res

def write_stats_to_tensorboard(stats:dict, writer:tensorboard.SummaryWriter,
                               it:int,
                               prefix:str="", suffix:str="") :
    for k in stats.keys() :
        if type(stats[k]) is dict :
            write_stats_to_tensorboard(stats[k], writer, it,
                                       prefix=prefix+"/"+k, suffix=suffix)
        else :
            writer.add_scalar(prefix+"/"+k+"/"+suffix, stats[k], it)

def write_ims_to_tensorboard(ims, writer, it, prefix="", suffix="") :
    for im_name in ims :
        writer.add_image(prefix + "/" + im_name + "/" + suffix,
                         ims[im_name].transpose((2, 0, 1)), it)

def write_runtime_stats_to_tensorboard(runtime_stats, writer, it, prefix="", samples=1) :
    runtime_total = runtime_stats['total']
    writer.add_scalar(prefix + "runtime/" + "modelforward_of_batch",
                      runtime_stats['model'] / runtime_stats['batch_total'], it)
    writer.add_scalar(prefix + "runtime/" + "sampletodevice_of_batch",
                      runtime_stats['sample_to_device'] / runtime_stats['batch_total'], it)
    writer.add_scalar(prefix + "runtime/" + "modelforward_of_forward",
                      runtime_stats['model'] / runtime_stats['forward'], it)
    writer.add_scalar(prefix + "runtime/" + "batch_of_loop",
                      runtime_stats['batch_total'] / runtime_total, it)
    writer.add_scalar(prefix + "runtime/" + "forwardevalbackward_of_batch",
                      (runtime_stats['forward'] +
                       runtime_stats['loss'] + runtime_stats['batch_total']) / runtime_stats['batch_total'], it)
    writer.add_scalar(prefix + "runtime/" + "ims_of_batch",
                      (runtime_stats['images']) / runtime_stats['batch_total'], it)
    writer.add_scalar(prefix + "runtime/" + "batch_per_sample",
                      runtime_stats['batch_total'] / samples, it)
    writer.add_scalar(prefix + "runtime/" + "total_per_sample",
                      runtime_stats['total'] / samples, it)


def create_image_grid(image_rows) :
    # how to set this  up
    height = 0
    width = 0
    for row in image_rows :
        height += max(map(lambda a : a.shape[0], row))
        width = max(width, np.sum(list(map(lambda a : a.shape[1], row))))
    im_grid = np.zeros((height, width, image_rows[0][0].shape[-1]), dtype=np.uint8)
    height_acc = 0
    for row in image_rows :
        width_acc = 0
        for im in row :
            im_grid[height_acc:height_acc + im.shape[0],
                width_acc:width_acc + im.shape[1]] = im
            width_acc += im.shape[1]
        height_acc += max(map(lambda a : a.shape[0], row))

    return im_grid


def cat_images(images) :
    height = max(map(lambda a : a.shape[0], images))
    width = np.sum(list(map(lambda a: a.shape[1], images)))
    im_cat = np.zeros((height, width, images[0].shape[-1]), dtype=images[0].dtype)
    width_acc = 0
    for im in images :
        im_cat[0:im.shape[0], width_acc:width_acc+im.shape[1]] = im
        width_acc += im.shape[1]
    return im_cat

def stack_images(images) :
    width = max(map(lambda a: a.shape[1], images))
    height = np.sum(list(map(lambda a: a.shape[0], images)))
    im_cat = np.zeros((height, width, images[0].shape[-1]), dtype=images[0].dtype)
    height_acc = 0
    for im in images:
        im_cat[height_acc:height_acc+im.shape[0], 0:im.shape[1]] = im
        height_acc += im.shape[0]
    return im_cat