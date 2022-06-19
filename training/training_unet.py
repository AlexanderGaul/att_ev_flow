import torch


from training.training_interface import AbstractTrainer
from training.training_volume import eval_volume_custom, eval_volume_args
from training.training_report import compute_statistics, paint_pictures_evaluation_frames, cat_images

from data.utils import tensor_or_tensorlist_to_device

from plot import create_event_frame_picture, put_text

from data.volume_functional import tile_tensor, untile_weight_tensor

class EventUnetTrainer(AbstractTrainer) :
    def __init__(self, lfunc, eval_iterations=False, tile_res=None, num_tiles_per_dim=None) :
        self.lfunc = lfunc
        self.eval_iterations = eval_iterations
        self.tile_res = tile_res
        self.num_tiles_per_dim = num_tiles_per_dim

    def batch_size(self, sample) :
        return len(sample['event_volume'])

    def sample_to_device(self, sample, device) :
        tensor_keys = ['event_volume', 'flow_frame', 'flow_frame_eval']
        if 'event_volume_prev' in sample : tensor_keys.append('event_volume_prev')
        return tensor_or_tensorlist_to_device(sample, device, tensor_keys)

    def forward(self, model, sample) :
        tile = False
        if self.tile_res is not None and sample['event_volume'].shape[-2] != self.tile_res[0] and sample['event_volume'].shape[-1] != self.tile_res[1] :
            tile = True
            if 'event_volume_prev' in sample:
                input = (tile_tensor(sample['event_volume'],
                                     self.tile_res, self.num_tiles_per_dim),
                         tile_tensor(sample['event_volume_prev'],
                                     self.tile_res, self.num_tiles_per_dim))
            else:
                input = (tile_tensor(sample['event_volume'],
                                     self.tile_res, self.num_tiles_per_dim),)
        else :
            if 'event_volume_prev' in sample :
                input = (sample['event_volume'], sample['event_volume_prev'])
            else :
                input = (sample['event_volume'],)
        if not self.eval_iterations :
            pred = model(*input)
            if tile :
                pred = untile_weight_tensor(pred, sample['event_volume'].shape[-2:], self.num_tiles_per_dim)
            return {'pred' : pred}
        else :
            pred, pred_iterations = model(*input)
            if tile :
                pred = untile_weight_tensor(pred, sample['event_volume'].shape[-2:], self.num_tiles_per_dim)
                pred_iterations = [untile_weight_tensor(p, sample['event_volume'].shape[-2:], self.num_tiles_per_dim)
                                   for p in pred_iterations]
            return {'pred' : pred, 'pred_iterations' : pred_iterations}

    def evaluate(self, sample, out) :
        if not self.eval_iterations :
            loss, preds, flows, coords = eval_volume_args(out['pred'],
                                                      sample['flow_frame'],
                                                      sample['flow_frame_eval'],
                                                      self.lfunc)
        else :
            losses = []
            for pred in out['pred_iterations'] :
                loss, preds, flows, coords = eval_volume_args(pred,
                                                              sample['flow_frame'],
                                                              sample['flow_frame_eval'],
                                                              self.lfunc)
                losses.append(loss)
            loss = 0.
            for i, l in enumerate(losses) :
                loss += 0.8**(len(losses) - i - 1) * l
        return {'loss' : loss,
                'preds_flat' : preds,
                'flows_flat' : flows,
                'coords' : coords}

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
        report = paint_pictures_evaluation_frames(
            create_event_frame_picture(sample['event_volume'][idx].detach().cpu().numpy()),
            out['pred'][idx].detach().cpu().numpy().transpose(1, 2, 0),
            sample['flow_frame'][idx].detach().cpu().numpy(),
            str(sample['frame_id'][idx]),
            eval_mask=sample['flow_frame_eval'][idx].detach().cpu().numpy(),
            flow_frame_valid=sample['flow_frame_valid'][idx].numpy(),
            include_gt=True
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


def forward_unet(model, sample, dataset, **kwargs) :
    return model(torch.stack(sample['event_volume_new']))

def eval_unet(pred_dense, sample, lfunc, **kwargs) :
    return eval_volume_custom(pred_dense, sample, lfunc)