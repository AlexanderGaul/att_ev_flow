import os

import numpy as np

from dsec import read_flows_mask


class FlowFrameSequence :
    def __init__(self, flow_dir, ts_path) :
        self.dir = flow_dir
        self.file_names = sorted(os.listdir(self.dir))
        self.ts = np.loadtxt(ts_path)

    def __len__(self) :
        return len(self.file_names)

    # [] TODO : should we not rather return a mask
    def __getitem__(self, idx) :
        flows, mask = read_flows_mask(self.dir / self.file_names[idx])

        return {'flow_frame' : flows,
                'mask' : mask, # TODO: change name to 'flow_mask'
                'ts' : self.ts[idx, :],
                'dt' : self.ts[idx, 1] - self.ts[idx, 0]}


class FlowFrame2MaskedArrayPrep :
    def __init__(self) : pass
    def __call__(self, data) :
        flow_data = data
        flow_array = flow_data['flow_frame'][flow_data['mask'], :]
        coords_all = np.stack(tuple(reversed(np.where(np.ones(flow_data['mask'].shape))))).transpose()
        coords_mask = flow_data['mask'].reshape(-1)
        del flow_data['flow_frame']
        del flow_data['mask']
        return {'flow_array': flow_array, 'coords': coords_all, 'coords_mask' : coords_mask,
                **flow_data}


class FlowFrame2ArrayPrep :
    def __init__(self) : pass
    def __call__(self, data) :
        flow_data = data
        flow_array = flow_data['flow_frame'][flow_data['mask'], :]
        flow_coords = np.stack(tuple(reversed(np.where(flow_data['mask'])))).transpose()
        del flow_data['flow_frame']
        del flow_data['mask']
        return {'flow': flow_array, 'coords': flow_coords,
                **flow_data}


class FlowCoordsSequence(FlowFrameSequence) :
    def __init__(self, flow_dir, ts_path):
        super().__init__(flow_dir, ts_path)
        self.prep = FlowFrame2ArrayPrep()

    def __getitem__(self, idx) :
        flow_data = super()[idx]
        return self.prep(flow_data)