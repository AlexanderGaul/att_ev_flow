import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

from data.event_datasets import EventFlowDataset

from training.training_report import visualize_sample, visualize_sample_volume

from plot import flow_frame_color, flow_color, get_np_plot_flow

"""
dataset_array = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/Test",
                                 seqs=5)

sample = dataset_array[0]

im_col = flow_color(sample['coords'], sample['flow_array'], sample['res'], max_length=2)
im_vec = get_np_plot_flow(im_col,
                          sample['coords'], sample['flow_array'], sample['res'],
                          freq=5)
im_text = cv.putText(im_vec, "Hello World", (10, 60), cv.FONT_HERSHEY_TRIPLEX,
                     2, (0, 0, 0, 255), 20)
im_text = cv.putText(im_text, "Hello World", (10, 60), cv.FONT_HERSHEY_TRIPLEX,
                     2, (255, 255, 255, 255), 5)

im_text = cv.putText(im_vec, "Hello World", (200, 200), cv.FONT_HERSHEY_DUPLEX,
                     4, (0, 0, 0, 255), 20)
im_text = cv.putText(im_vec, "Hello World", (200, 200), cv.FONT_HERSHEY_DUPLEX,
                     4, (255, 255, 255, 255), 5)

plt.imshow(im_text)
plt.show()
"""

dataset = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/Test",
                           seqs=5,
                           return_volume=True,
                           normalize_volume=False,
                           t_bins=20,
                           spatial_downsample=2)

dataset = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/DSEC_flow",
                           seqs=1,
                           batch_backward=True,
                           return_volume=True,
                           normalize_volume=False,
                           t_bins=20,
                           spatial_downsample=4,
                           dataset_format='dsec')

#mean_events = np.zeros(dataset[0]['event_volume'].shape)
#mean_flow = np.zeros(dataset[0]['flow_frame'].shape)

for i in range(len(dataset) - 1, len(dataset)+1) :
    sample = dataset[i]
    samples = [sample] if type(sample) is not list else sample
    for sample in sample :
        #mean_events += sample['event_volume'] / len(dataset)
        #mean_flow[sample['flow_frame_valid']] += sample['flow_frame'][sample['flow_frame_valid']] / len(dataset)
        #plt.imshow(visualize_sample_volume(sample))
        #plt.show()
        #print(len(sample['event_array']))
        plt.imshow(visualize_sample_volume({'event_volume': sample['event_volume'],
                                            'flow_frame': sample['flow_frame'],
                                            'flow_frame_valid': sample['flow_frame_valid']}, freq=5))
        plt.show()
        plt.imshow(flow_frame_color(sample['flow_frame']))
        plt.show()
        a = 1
"""
plt.imshow(visualize_sample_volume({'event_volume' : mean_events,
                                    'flow_frame' : mean_flow,
                                    'flow_frame_valid' : np.ones(mean_flow.shape[:-1],
                                                          dtype=bool)}))
plt.show()

"""

print("EOF")