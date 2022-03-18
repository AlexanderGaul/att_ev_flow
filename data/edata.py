import math
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

import h5py
import imageio

import os
from pathlib import Path

from data.basic_datasets import *
from dsec import read_flows_mask
from events import interp_volume

from event_slicer import EventSlicer

from utils import get_grid_coordinates

from typing import Dict, List

# Ticket 006 - week 08 - Fri 04.03.
# TODO: refactor abstract parts of the DSEC dataset into super class
# TODO: move "Event Slicer" out of dataset for easy use
# [] should we create a separate branch in git?

# [] Class to read events that conntains the file
#    - mainly a wrapper class?


# Ticket 007 - week 08 - Fri 04.03.
# TODO: implement event slicer class with inspiration from ERaft
# STATS: expected time 1 hour,  actual time
# [] can wwe reuse from loader_dsec with minor adjustments?
# [] copy past basic functionality
# [] do not require offset in the event file
# [] allow to include upper bound
# [] allow to exclude lower bound


# [] Class to access ground truth flow? -> maybe not
#    - could swap this out if different format for timestamps is used?
#    - could inherit from torch dataset -> but is not a "dataset"
#    - are there actually different methods for format dataset -> no, not really
#    - for synthetic dataset we can actually construct backward events
# [] How to fuse backward events with corresponsing forward events?
#    - child class should know structure
# should the base class be for a sequence?
# needs t obe dataset but what should we return that can actually be used
# could contain said ground truth class that provides number of ground truth flow labels
# could provide interface to allow for augmentation of said ground truth flow
# could be a dataset but semantically it is not
# don't have to create a base class might as well be any class and owned by dataset
# implementations
#

# [] keep order for augmentation flexible ?
# [] downsample first, then cropping/flipping, then binning
# [] can we reuse binning for

# event directory, might contain h5 file or numpy arrays
# preprocessing, formatting
# have object that gives flow
# might return timestamps or file indices


# [] how to combine backward and forward
# [] keep backward and forward separate or in same class
# [] can have either virtual backward or actual backward
# [] backward events might also be separate

# TODO: could choose frames
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
                'mask' : mask,
                'ts' : self.ts[idx, :],
                'dt' : self.ts[idx, 1] - self.ts[idx, 0]}

class FlowCoordsSequence :
    def __init__(self, flow_dir, ts_path):
        super().__init__(flow_dir, ts_path)

    def __getitem__(self, idx) :
        flow_data = super()[idx]
        flow_array = flow_data['flow_frame'][flow_data['mask'], :]
        flow_coords = np.stack(tuple(reversed(np.where(flow_data['mask'])))).transpose()
        del flow_data['flow_frame']
        del flow_data['mask']
        return {'flow' : flow_array, 'coords' : flow_coords,
                **flow_data}



# [] add previous frame inside or outside
# this is just a useless wrapper
class EventSequence :
    def __init__(self, slicer, include_endpoint=True) :
        self.slicer = slicer
        self.include_endpoint = include_endpoint

    # TODO which time unit
    # Better not to put this into a __getitem__
    def get_events(self, t_begin, t_end) :
        events = self.slicer.get_events(int(t_begin), int(t_end), self.include_endpoint)
        events['p'] = events['p'].astype(np.int8)
        events['p'][events['p'] == 0] = -1
        #events['t'] -= t_begin

        # should we construct array already here? maybe not
        # but would have to rewrite functions to handle this stupid dict?
        return events





class BasicArraySequence(Dataset) :
    # TODO: change to folders
    def __init__(self, flow_sequence, event_sequence) :
        self.flow_sequence = flow_sequence
        self.event_sequence = event_sequence

    def __len__(self) :
        return len(self.flow_sequence)

    def __getitem__(self, idx) :
        flow_data = self.flow_sequence[idx]

        flow_array = flow_data['flow_frame'][flow_data['mask'], :]
        flow_coords = np.stack(tuple(reversed(np.where(flow_data['mask'])))).transpose()

        events = self.event_sequence.get_events(*flow_data['ts'])
        event_array = np.concatenate([events['x'][:, np.newaxis],
                                      events['y'][:, np.newaxis],
                                      events['t'][:, np.newaxis] - flow_data['ts'][0],
                                      events['p'][:, np.newaxis]
                                      ], axis=1)
        event_array[:, 2] /= 1000.

        return {'events' : event_array,
                'flows' : flow_array,
                'coords' : flow_coords,
                'res' : tuple(reversed(flow_data['flow_frame'].shape[:2])),
                'dt' : (flow_data['ts'][1] - flow_data['ts'][0]) / 1000.,
                'frame_id' : idx}


class BasicVolumeSequence(Dataset) :
    def __init__(self, flow_sequence, event_sequence, num_bins) :
        self.flow_sequence = flow_sequence
        self.event_sequence = event_sequence
        self.num_bins = int(num_bins)

    def __len__(self) :
        return len(self.flow_sequence)

    def __getitem__(self, idx) :
        flow_data = self.flow_sequence[idx]

        events = self.event_sequence.get_events(*flow_data['ts'])
        event_array = np.concatenate([events['x'][:, np.newaxis],
                                      events['y'][:, np.newaxis],
                                      events['t'][:, np.newaxis] - flow_data['ts'][0],
                                      events['p'][:, np.newaxis]
                                      ], axis=1)
        event_array[:, 2] /= 1000.

        res = tuple(reversed(flow_data['flow_frame'].shape[:2]))
        dt = (flow_data['ts'][1] - flow_data['ts'][0]) / 1000.

        event_volume = interp_volume(event_array, res,
                                     self.num_bins, 0., dt)

        return {'event_volume_new' : event_volume,
                'flow_frame' : flow_data['flow_frame'].transpose(2, 0, 1),
                'flow_mask' : flow_data['mask'],
                'res': res,
                'dt': dt,
                'frame_id': idx}


class BasicDataset(Dataset) :
    def __init__(self, dir:Path, flow_suffix, ts_suffix, event_suffix,
                 ESeqClass, FSeqClass, SeqClass,
                 seqs=None,
                 seq_params=dict()) :
        self.dir = Path(dir)
        seq_names = sorted(os.listdir(self.dir))
        if seqs is not None and hasattr(seqs, '__len__') :
            seq_names = [seq_names[i] for i in seqs]
        elif seqs is not None:
            if seqs > 0 :
                seq_names = seq_names[:seqs]
            elif seqs < 0 :
                seq_names = seq_names[seqs:]

        self.seqs = []
        for sq in seq_names :
            slicer = EventSlicer(h5py.File(self.dir / sq / event_suffix))
            e_seq = ESeqClass(slicer)
            f_seq = FSeqClass(self.dir / sq / flow_suffix,
                              self.dir / sq / ts_suffix)
            self.seqs.append(SeqClass(f_seq, e_seq, **seq_params))


        self.dataset = torch.utils.data.ConcatDataset(self.seqs)

    def __len__(self) :
        return len(self.dataset)

    def __getitem__(self, idx) :
        # TODO check if frame id requires 3-tuple
        item = self.dataset[idx]
        seq_idx = np.searchsorted(self.dataset.cumulative_sizes, idx)

        item['frame_id'] = (seq_idx, item['frame_id'])
        item['tbins'] = 100

        return item


class HomographyDataset(BasicDataset) :
    def __init__(self, dir="/storage/user/gaul/gaul/thesis/data/Homographies_01",
                 seqs=None) :

        super().__init__(dir,
                         flow_suffix="flow/forward/",
                         ts_suffix="flow/forward_timestamps.txt",
                         event_suffix="events/events.h5",
                         ESeqClass=EventSequence,
                         FSeqClass=FlowFrameSequence,
                         SeqClass=BasicArraySequence,
                         seqs=seqs)


class HomographyDatasetVolume(BasicDataset) :
    def __init__(self, dir="/storage/user/gaul/gaul/thesis/data/Homographies_01",
                 seqs=None, num_bins=15) :
        super().__init__(dir,
                         flow_suffix="flow/forward/",
                         ts_suffix="flow/forward_timestamps.txt",
                         event_suffix="events/events.h5",
                         ESeqClass=EventSequence,
                         FSeqClass=FlowFrameSequence,
                         SeqClass=BasicVolumeSequence,
                         seqs=seqs,
                         seq_params={'num_bins' : num_bins})
        self.num_bins = num_bins
