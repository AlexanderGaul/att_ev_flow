import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.event_datasets import EventFlowDataset

import numpy as np

dataset = EventFlowDataset(dir="/storage/user/gaul/gaul/thesis/data/HomographiesPatchOffset/")

print(dataset.__dict__)

max_events = 0
mean_events = 0
num_events = np.zeros(len(dataset))
for i, data in enumerate(dataset) :
    #print(data['event_array'].shape[0])
    if data['event_array'].shape[0] > max_events :
        max_events  = data['event_array'].shape[0]
    mean_events  += data['event_array'].shape[0] / len(dataset)
    num_events[i] = data['event_array'].shape[0]
print("--------")
print(max_events)
print(mean_events)
print("EOF")