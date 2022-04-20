import os
from pathlib import Path

import numpy as np

from dsec import read_flows_mask


class FlowFrameSequence :
    def __init__(self, flow_dir, ts_path) :
        self.dir = flow_dir
        self.file_names = sorted(os.listdir(self.dir))
        with open(ts_path, 'r') as f_ppek :
            self.ts = np.loadtxt(ts_path, dtype=np.uint64,
                                 delimiter=',' if ',' in f_ppek.readline() else None)

    def __len__(self) :
        return len(self.file_names)

    def __getitem__(self, idx) :
        flows, mask = read_flows_mask(self.dir / self.file_names[idx])

        return {'flow_frame' : flows,
                'flow_frame_valid' : mask,
                'ts' : tuple(self.ts[idx, :]),
                'dt' : (max(self.ts[idx]) - min(self.ts[idx])) / 1000,
                'res' : tuple(reversed(flows.shape[:2])),
                'path' : str(os.path.basename(self.dir)) + "/" + self.file_names[idx].split('.')[0]}

    def get_ts(self, idx) :
        return self.ts[idx, :]


class FlowFrameUniBackward :
    def __init__(self) : pass
    def __call__(self, data) :
        data['flow_frame'] *= -1
        return data


class FlowFrame2MaskedArray :
    def __init__(self) : pass
    def __call__(self, data) :
        flow_data = data
        flow_array = flow_data['flow_frame'][flow_data['flow_frame_eval'], :]
        coords_all = np.stack(tuple(reversed(np.where(np.ones(flow_data['flow_frame_eval'].shape))))).transpose()
        coords_mask = flow_data['flow_frame_eval'].reshape(-1)
        res = tuple(reversed(flow_data['flow_frame'].shape[:2]))
        #del flow_data['flow_frame']
        del flow_data['flow_frame_eval']
        return {'flow_array': flow_array, 'coords': coords_all, 'coords_mask' : coords_mask,
                'res' : res,
                **flow_data}

def flow_at_coords(coords, flow_frame, flow_mask):
    flows = flow_frame[coords[:, 1].astype(int), coords[:, 0].astype(int)]
    mask = flow_mask[coords[:, 1].astype(int), coords[:, 0].astype(int)]
    return flows, mask

# What can we alread crop here
# coords is possibly always query input
#
class FlowFrame2FlowArrayAtAllCoordinates:
    def __init__(self, input='event_array'):
        self.input = input
    def __call__(self, data):
        coords = data[self.input][:, :2]
        flows, mask = flow_at_coords(coords, data['flow_frame'], data.pop('flow_frame_eval'))

        data['flow_array'] = flows[mask]
        data['coords'] = coords
        data['coords_mask'] = mask

        return data



def flow_frame_2_array(flow_frame, flow_mask) :
    flow_array = flow_frame[flow_mask, :]
    flow_coords = np.stack(tuple(reversed(np.where(flow_mask)))).transpose()
    return flow_array, flow_coords

class FlowFrame2Array :
    def __init__(self) : pass
    def __call__(self, data) :
        flow_data = data
        res = tuple(reversed(flow_data['flow_frame'].shape[:2]))
        flow_array, flow_coords = flow_frame_2_array(data['flow_frame'],
                                                     data.pop('flow_frame_eval'))
        return {'flow_array': flow_array, 'coords': flow_coords,
                'res': res,
                **flow_data}


class FlowCoordsSequence(FlowFrameSequence) :
    def __init__(self, flow_dir, ts_path):
        super().__init__(flow_dir, ts_path)
        self.prep = FlowFrame2Array()

    def __getitem__(self, idx) :
        flow_data = super()[idx]
        return self.prep(flow_data)


