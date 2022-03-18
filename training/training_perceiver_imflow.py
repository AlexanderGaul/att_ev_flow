import torch

from model import EventTransformer
from training.training import compute_statistics, paint_pictures, visualize


class ImagePerceiverTrainer :
    def __init__(self, visualize_frames) :
        self.visualize_frames = visualize_frames
        
    
    # TODO: should we copy the sample create sample only with deviced elements
    def sample_to_device(self, sample, device) :
        assert len(sample['patch_array'].shape) == 3, "assume batched tensors"
        sample['patch_array'] = sample['patch_array'].to(device)
        sample['flow_coords'] = sample['flow_coords'].to(device)
        sample['flow_array'] = sample['flow_array'].to(device)
        return sample
    
    def forward(self, model:EventTransformer, sample) :
        patch_data = sample['patch_array']
        coords = sample['coords']

        out = model(patch_data, coords, sample['res'], sample['dt'])
        return {'pred' : out}
    
    def evaluate(self, sample, out, lfunc) :
        pred = out['pred']
        flow = sample['flow_array']
        loss = torch.cat([lfunc(pred[i],
                                flow[i]).reshape(1)
                          for i in range(len(pred))]).nansum()
        return {'loss' : loss}
    
    def statistics(self, sample, out, eval, fraction) :
        return compute_statistics(out['pred'], sample['flow_array'], 
                                  {}, fraction)
    
    def visualize(self, sample, out, eval) :
        report = {}
        for i, id in enumerate(sample['frame_id']) :
            if list(id[:2]) in self.visualize_frames :
                im = sample['im1'][i]
                report = paint_pictures(im
                               sample['flow_coords'][i].detach().cpu().numpy()
                               out['pred'][i].detach().cpu().numpy(),
                               str(sample['frame_id'][i]),
                               {})
        return report
