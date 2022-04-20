import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from data.event_datasets import EventFlowDataset


d_sep = EventFlowDataset("/storage/user/gaul/gaul/thesis/data/HomographiesForegroundTest48",
                         batch_backward=True)

d_comb = EventFlowDataset("/storage/user/gaul/gaul/thesis/data/HomographiesForegroundTestCombination48",
                          batch_backward=True)


dummy = d_comb.seqs[0].event_flow_data.event_stream.get_events(1000001, 3000000)


assert len(d_sep) == len(d_comb)

for i in range(0, len(d_sep)) :

    sample_sep = d_sep[i]
    sample_comb = d_comb[i]


    for j in range(len(sample_sep)) :
        for k in sample_sep[j].keys() :
            print(k)
            if k == 'frame_id' or k == 'ts' or k == 'path' : continue
            if type(sample_sep[j][k]) is np.ndarray :
                assert (sample_sep[j][k] == sample_comb[j][k]).all()
                print("asserted")
            else :
                assert sample_sep[j][k] == sample_comb[j][k]
                print("asserted")

    print(len(sample_sep[0]['event_array']))
    print(len(sample_sep[1]['event_array']))

    # TODO: do asserts automatically for all members


    print(i)
