import torch

from training.training_interface import *
from model import EventTransformer
from training.training_report import compute_statistics, paint_pictures_evaluation_arrrays, visualize


class ImagePerceiverTrainer(AbstractTrainer) :
    def __init__(self, lfunc) :
        self.lfunc = lfunc

    def batch_size(self, sample) :
        if len(sample['patch_array'].shape) != 3 :
            return None
        else :
            return sample['patch_array'].shape[0]
    
    # TODO: should we copy the sample create sample only with deviced elements
    def sample_to_device(self, sample, device) :
        assert len(sample['patch_array'].shape) == 3, "assume batched tensors"
        sample['patch_array'] = sample['patch_array'].to(device)
        sample['coords'] = sample['coords'].to(device)
        sample['flow_array'] = sample['flow_array'].to(device)
        if 'coords_mask' in sample :
            sample['coords_mask'] = sample['coords_mask'].to(device)
        return sample
    
    def forward(self, model:EventTransformer, sample) :
        patch_data = sample['patch_array']
        coords = sample['coords']

        out = model(patch_data, coords, sample['res'], sample['dt'])
        return {'pred' : out}
    
    def evaluate(self, sample, out) :
        pred = out['pred']
        flow = sample['flow_array']
        loss = torch.cat([self.lfunc(pred[i],
                                     flow[i]).reshape(1)
                          for i in range(len(pred))]).nanmean()
        return {'loss' : loss}
    
    def statistics(self, sample, out, eval, fraction, report) :
        return compute_statistics(out['pred'], sample['flow_array'], 
                                  report, fraction)
    
    def visualize(self, visualize_frames, sample, out, eval, report=dict()) :
        if not visualize_frames or len(visualize_frames) == 0:
            return report
        for i, id in enumerate(sample['frame_id']) :
            if list(id[:2]) in visualize_frames :
                im = sample['im1'][i]
                im2 = sample['im2'][i]
                # TODO: can blend images
                report = paint_pictures_evaluation_arrrays(0.5 * im + 0.5 * im2,
                                                           sample['coords'][i].detach().cpu().numpy(),
                                                           out['pred'][i].detach().cpu().numpy(),
                                                           sample['flow_array'][i].detach().cpu().numpy(),
                                                           str(sample['frame_id'][i]),
                                                           report)
        return report
