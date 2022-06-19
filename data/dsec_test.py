import os
from pathlib import Path

import numpy as np
import torch

import h5py

from torch.utils.data import Dataset

from data.event_datasets import EventStream
from data.preprocessing.event_modules import EventData2EventArray, EventArrayUndistortMap
from data.volume_modules import EventArray2Volume

class DSECTestSequence(Dataset) :

    def __init__(self, seq_name, event_dir, ts_path, *args, **kwargs) :
        super().__init__()

        self.seq_name = seq_name
        self.event_stream = EventStream(Path(event_dir) / "events.h5")
        self.rectify_map = h5py.File(Path(event_dir) / "rectify_map.h5", locking=False)['rectify_map'][:]

        ts_file = np.genfromtxt(ts_path, delimiter=',', dtype=np.uint64)
        self.ts = ts_file[:, :2]
        self.idxs = ts_file[:, 2]

        self.preprocessing = [EventData2EventArray(),
                              EventArrayUndistortMap(self.rectify_map),
                              EventArray2Volume(num_bins=kwargs['t_bins'], xyfloat=True)]


    def __len__(self) :
        return len(self.idxs)

    def __getitem__(self, idx) :
        ts = self.ts[idx]
        data = {'event_data' : self.event_stream.get_events(*ts),
                'ts' : tuple(ts),
                'dt' : (ts[1] - ts[0]) / 1000,
                'seq_name' : self.seq_name,
                'out_file_name' : str(self.idxs[idx]).zfill(6),
                'res' : (640, 480)}

        for fn in self.preprocessing :
            fn(data)

        return data





class DSECFlowTest(Dataset) :

    def __init__(self, dir, *args, **kwargs) :
        self.dir = Path(dir)

        dir_content = sorted(os.listdir(self.dir))

        self.ts_files = list(filter(lambda f : '.csv' in f, dir_content))
        self.seq_names = [f_name.split('.')[0] for f_name in self.ts_files]

        self.seqs = [DSECTestSequence(sname,
                                      self.dir / sname / "events/left/",
                                      self.dir / (sname + ".csv"),
                                      *args, **kwargs)
                     for sname in self.seq_names]

        self.dataset = torch.utils.data.ConcatDataset(self.seqs)

    def __len__(self) :
        return len(self.dataset)

    def __getitem__(self, idx) :
        return self.dataset[idx]