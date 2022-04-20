import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import interp_volume_jit, interp_volume_jit_xyfloat

import matplotlib.pyplot as plt

from data.event_datasets import EventFlowDataset

from training.training_report import visualize_sample, visualize_sample_volume

from plot import flow_frame_color, flow_color, get_np_plot_flow, create_event_frame_picture


dataset = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/Test",
                           seqs=5,
                           return_volume=False,
                           normalize_volume=False,
                           t_bins=4,
                           spatial_downsample=2)

dsec = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/DSEC_flow",
                           seqs=1,
                           batch_backward=False,
                           return_volume=True,
                           normalize_volume=False,
                           t_bins=4,
                           spatial_downsample=4,
                           dataset_format='dsec',
                           undistort_events=True)
dsec_dist = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/DSEC_flow",
                           seqs=1,
                           batch_backward=False,
                           return_volume=True,
                           normalize_volume=False,
                           t_bins=4,
                           spatial_downsample=4,
                           dataset_format='dsec',
                           undistort_events=False)


sample = dataset[0]

vol = interp_volume_jit(sample['event_array'], sample['res'], 5, 0, sample['dt'])
vol_xy = interp_volume_jit_xyfloat(sample['event_array'], sample['res'], 5, 0, sample['dt'])

diff = (vol - vol_xy)


sample = dsec[0]
sample_dist = dsec_dist[0]


plt.imshow(visualize_sample_volume({'event_volume': sample['event_volume'],
                                            'flow_frame': sample['flow_frame'],
                                            'flow_frame_valid': sample['flow_frame_valid']}, freq=20))
plt.show()
plt.figure(figsize=(sample['event_volume'].shape[2] /10,
                   sample['event_volume'].shape[1] /10))
plt.imshow(create_event_frame_picture(sample['event_volume']))
plt.show()

sample = dsec_dist[0]

plt.imshow(visualize_sample_volume({'event_volume': sample['event_volume'],
                                            'flow_frame': sample['flow_frame'],
                                            'flow_frame_valid': sample['flow_frame_valid']}, freq=20))
plt.show()

print("EOF")