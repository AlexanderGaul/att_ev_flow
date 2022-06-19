import os
from pathlib import Path

import numpy as np

from dsec import read_flows_mask


class EvalFlowSequence :
    def __init__(self, ts_path) :
        ts_file = np.genfromtxt(ts_path, delimiter=',', dtype=np.uint64)
        self.ts = ts_file[:, :2]
        self.idxs = ts_file[:, 2]
        self.seq_name = os.path.basename(ts_path).split('.')[0]

    def __len__(self) :
        return len(self.ts)

    def __getitem__(self, idx) :
        return {'flow_frame': np.zeros((480, 6480, 2)),
                'flow_frame_valid': np.zeros((480, 640)),
                'ts': tuple(self.get_ts(idx)),
                'dt': (max(self.get_ts(idx)) - min(self.get_ts(idx))) / 1000,  # should be in ms
                'res': (640, 480),
                'path': self.seq_name,
                'seq_name': self.seq_name,
                'out_file_name': str(self.idxs[idx]).zfill(6)
                }

    def get_ts(self, idx) :
        return self.ts[idx, :]


class FlowFrameSequence :
    def __init__(self, flow_dir, ts_path, ts_to_us=1, skip_flow_freq=1) :
        self.dir = flow_dir
        self.file_names = sorted(os.listdir(self.dir))
        self.ts_to_us = ts_to_us
        with open(ts_path, 'r') as f_peek :
            line = f_peek.readline()
            delimiter = ',' if ',' in line else None
            if '#' in line :
                line = f_peek.readline()
                dtype = np.uint64 if not '.' in line else np.float64
            else :
                dtype = np.uint64 if not '.' in line else np.float64
            self.ts = np.loadtxt(ts_path, dtype=dtype,
                                 delimiter=delimiter)[:, :2]
        if len(self.ts) > 2000 :
            self.skip_flow_freq = skip_flow_freq
        else :
            self.skip_flow_freq = 1

    def __len__(self) :
        return len(self.file_names) // self.skip_flow_freq

    def __getitem__(self, idx) :
        idx_raw = idx
        if self.skip_flow_freq > 1 :
            idx *= self.skip_flow_freq
        flows, mask = read_flows_mask(self.dir / self.file_names[idx])

        return {'flow_frame' : flows,
                'flow_frame_valid' : mask,
                'ts' : tuple(self.get_ts(idx_raw)),
                'dt' : (max(self.get_ts(idx_raw)) - min(self.get_ts(idx_raw))) / 1000, # should be in ms
                'res' : tuple(reversed(flows.shape[:2])),
                'path' : str(os.path.basename(self.dir)) + "/" + self.file_names[idx].split('.')[0]}


    def getitems(self, idx, num_items) :
        idx_raw = idx
        if self.skip_flow_freq > 1:
            idx *= self.skip_flow_freq
        flow_list = []
        mask_list = []
        for i in range(idx, idx+num_items) :
            flows, mask = read_flows_mask(self.dir / self.file_names[idx])
            flow_list.append(flows)
            mask_list.append(mask)

        flows, mask = read_flows_mask(self.dir / self.file_names[idx])

        return {'flow_frame': np.stack(flow_list),
                'flow_frame_valid': np.stack(mask_list),
                'ts': tuple(self.get_ts(idx_raw)),
                'dt': (max(self.get_ts(idx_raw)) - min(self.get_ts(idx_raw))) / 1000,  # should be in ms
                'res': tuple(reversed(flows.shape[:2])),
                'path': str(os.path.basename(self.dir)) + "/" + self.file_names[idx].split('.')[0]}

    def get_ts(self, idx) :
        if self.skip_flow_freq > 1 :
            idx *= self.skip_flow_freq
        if idx > len(self.ts) :
            print("fail")
        if len(self.ts.shape) == 2 :
            return self.ts[idx, :] * self.ts_to_us
        else :
            return self.ts[idx:idx+2] * self.ts_to_us



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


class FlowFrameCrop :
    def __init__(self, offset, res) :
        self.offset = offset
        self.res = res
    def __call__(self, data):
        for key in ['flow_frame', 'flow_frame_valid', 'flow_frame_eval'] :
            data[key] = data[key][self.offset[0]:self.offset[0]+self.res[0],
                                  self.offset[1]:self.offset[1]+self.res[1]]
        return data
