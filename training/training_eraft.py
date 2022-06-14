import numpy as np
import torch

from training.training_interface import AbstractTrainer
from ERAFT.model.utils import bilinear_sampler
from training.training_volume import eval_volume_custom, eval_volume_args
from training.training_report import compute_statistics, paint_pictures_evaluation_frames, cat_images

from plot import create_event_frame_picture, put_text

from data.utils import tensor_or_tensorlist_to_device

class ERaftTrainer(AbstractTrainer) :
    def __init__(self, lfunc, eval_iterations=False, iters=None) :
        self.lfunc = lfunc
        self.eval_iterations = eval_iterations
        self.iters = iters

    def batch_size(self, sample):
        return len(sample['event_volume'])

    def sample_to_device(self, sample, device) :
        tensor_keys = ['event_volume', 'event_volume_prev', 'flow_frame', 'flow_frame_eval']
        return tensor_or_tensorlist_to_device(sample, device, tensor_keys)

    def forward(self, model, sample) :
        im1 = sample['event_volume_prev']
        im2 = sample['event_volume']

        if self.iters is None :
            flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None)
        else :
            flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None, iters=self.iters)
        return {'pred' : flow_list[-1],
                'pred_iterations' : flow_list,
                'flow_low_res' : flow_low_res,
                'flow_list' : flow_list}


    def evaluate(self, sample, out) :
        if not self.eval_iterations:
            loss, preds, flows, coords = eval_volume_args(out['pred'],
                                                          sample['flow_frame'],
                                                          sample['flow_frame_eval'],
                                                          self.lfunc)
        else :
            losses = []
            for pred in out['pred_iterations']:
                loss, preds, flows, coords = eval_volume_args(pred,
                                                              sample['flow_frame'],
                                                              sample['flow_frame_eval'],
                                                              self.lfunc)
                losses.append(loss)
            loss = 0.
            for i, l in enumerate(losses):
                loss += 0.8 ** (len(losses) - i - 1) * l
        return {'loss': loss,
                'preds_flat': preds,
                'flows_flat': flows,
                'coords': coords}

    def statistics(self, sample, out, eval, return_separate=False) :
        return compute_statistics(eval['preds_flat'],
                                  eval['flows_flat'],
                                  return_separate)

    def visualize(self, visualize_frames, sample, out, eval) :
        report=dict()
        if not visualize_frames or len(visualize_frames) == 0:
            return report
        for i, id in enumerate(sample['frame_id']) :
            if list(id[:2]) in visualize_frames :
                report.update(self.visualize_item(i, sample, out, eval,
                                                  write_label_on_img=True,
                                                  concat=True))

        return report

    def visualize_item(self, idx, sample, out , eval, write_label_on_img=False, concat=False) :
        double_volume = np.concatenate([
            sample['event_volume_prev'][idx].detach().cpu().numpy(),
            sample['event_volume'][idx].detach().cpu().numpy()
                                        ], axis=0)
        report = paint_pictures_evaluation_frames(
            create_event_frame_picture(double_volume),
            out['pred'][idx].detach().cpu().numpy().transpose(1, 2, 0),
            sample['flow_frame'][idx].detach().cpu().numpy(),
            str(sample['frame_id'][idx]),
            eval_mask=sample['flow_frame_eval'][idx].detach().cpu().numpy(),
            flow_frame_valid=sample['flow_frame_valid'][idx].numpy(),
            include_gt=True, E_im_gt=create_event_frame_picture(sample['event_volume'][idx].detach().cpu().numpy())
        )
        if write_label_on_img:
            if not concat:
                for k in report.keys():
                    report[k] = put_text(report[k], [sample['path'][idx], k])
            else:
                for k in report.keys():
                    report[k] = put_text(report[k], [k])
        if concat:
            im_cat = cat_images(list(report.values()))
            report = {str(sample['frame_id'][idx]): im_cat}
        return report





def forward_eraft(model, sample, dataset, **kwargs) :
    im1 = sample['event_volume_old']
    im2 = sample['event_volume_new']

    flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None)

    return (flow_low_res, flow_list)


def forward_eraft_custom(model, sample, dataset, **kwargs) :
    im1 = torch.stack(sample['event_volume_old'])
    im2 = torch.stack(sample['event_volume_new'])

    flow_low_res, flow_list = model(image1=im1, image2=im2, flow_init=None)

    return (flow_low_res, flow_list)


def eval_eraft_out(out,
                   sample, lfunc, **kwargs) :
    flow_low_res, flow_list = out
    pred_refined = flow_list[-1]

    valid = sample['valid2D']

    pred_eval = [pred_refined[i, :, valid[i, :]].transpose(1, 0) for i in range(len(pred_refined))]

    flow_dev = [sample['flow'][i, sample['valid2D'][i, :], :] for i in range(len(pred_refined))]

    loss = torch.cat([lfunc(pred_eval[i], flow_dev[i]).reshape(1)
                      for i in range(len(pred_refined))]).sum()

    return (loss,
            [predi.detach() for predi in pred_eval],
            [flowi.detach() for flowi in flow_dev],
            [torch.flip(torch.nonzero(sample['valid2D'][i, :]),
                        dims=[1])
             for i in range(len(pred_refined))])


def eval_eraft_custom(out, sample, lfunc, **kwargs) :
    flow_low_res, flow_list = out
    pred_flow = flow_list[-1]

    return eval_volume_custom(pred_flow, sample, lfunc)


