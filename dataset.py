import numpy as np

import torch

from scipy.interpolate import RegularGridInterpolator

from PIL import Image

import h5py

class MVSEC(torch.utils.data.Dataset) :
    def __init__(self, path, frames, dt_plus, dt_minus=0) :
        self.frames = frames
        
        
        self.data = h5py.File(path + "/indoor_flying1_data.hdf5")
        self.data_gt = h5py.File(path + "/indoor_flying1_gt.hdf5")
        
        self.events = np.array(self.data['davis']['left']['events'])
        
        self.images = []
        self.event_frames = []
        self.coords = []
        self.flows = []

        for i in self.frames : 
            flow_gt = self.data_gt['davis']['left']['flow_dist'][i]
            flow_gt_ts = self.data_gt['davis']['left']['flow_dist_ts'][i]
            
            im_id = np.absolute(self.data['davis']['left']['image_raw_ts'][:] - 
                                flow_gt_ts).argmin()
            
            self.images.append(self.data['davis']['left']['image_raw'][im_id])
            
            event_frame = torch.FloatTensor(
                self.events[
                    np.logical_and(self.events[:, 2] > flow_gt_ts - dt_minus,
                                   self.events[:, 2] < flow_gt_ts + dt_plus), :]
                )
            
            event_coords = event_frame[:, :2].long()
            event_has_gt = (
                torch.FloatTensor(
                    flow_gt[:, event_coords[:, 1], event_coords[:, 0]]).transpose(0, 1) 
                != 0.).any(dim=1)
            
            event_frame = event_frame[event_has_gt, :]
            event_frame[:, 2] -= flow_gt_ts
            event_frame[:, 2] *= 100
            
            self.event_frames.append(event_frame)
            
            coords = torch.FloatTensor(np.unique(event_frame[:, :2], axis=0))
            
            self.coords.append(coords)
            
            coords_int = coords.long()
            flows = torch.FloatTensor(flow_gt[:, 
                                              coords_int[:, 1], 
                                              coords_int[:, 0]]).transpose(0, 1)
            
            self.flows.append(flows) 
            
        
    def __len__(self) :
        return len(self.event_frames)
        
    
    def __getitem__(self, idx) :
        return self.event_frames[idx], self.coords[idx], self.flows[idx]


class TartanAir(torch.utils.data.Dataset) :
    def __init__(self, path, frames, dt_plus, dt_minus=0) :
        def get_image_name(i) :
            return '0' * (6 - len(str(i))) + str(i)
        
        self.frames = frames
        
        self.data = h5py.File(path + "/P001_events/DVS_H5.h5")
        
        #self.data_gt = h5py.File(path + "/indoor_flying1_gt.hdf5")
        
        # TODO: rename data
        self.events = np.array(self.data['events'])
        # self.events[:, 3][self.events[:, 3] == 0] = -1
        
        self.image_paths = [path + "/P001/image_left/" + 
                            get_image_name(i) + "_left.png"
                            for i in frames]
        self.flow_paths = [path + "/P001/flow/" + 
                           get_image_name(i) + "_" + get_image_name(i+1) + 
                           "_flow.npy"
                           for i in frames]
        
        self.images = [np.array(Image.open(img_path)) for img_path in self.image_paths]
        
        self.event_frames = []
        self.coords = []
        self.flows = []
        
        for i, frame in enumerate(self.frames) : 
            flow_gt = np.load(self.flow_paths[i])
            ts = frame * 33333
            
            event_frame = torch.FloatTensor(
                self.events[
                    np.logical_and(self.events[:, 0] > ts - dt_minus * 1000000,
                                   self.events[:, 0] < ts + dt_plus * 1000000), :].astype(dtype=np.int64)
                )
            event_frame = torch.cat([event_frame[:, 1:3], 
                                     event_frame[:, [0]],
                                     event_frame[:, [3]]], dim=1)
            event_coords = event_frame[:, :2].long()
            event_has_gt = (
                torch.FloatTensor(
                    flow_gt[event_coords[:, 1], event_coords[:, 0], :]) != 0.).any(dim=1)
            
            event_frame = event_frame[event_has_gt, :]
            event_frame[:, 2] -= ts
            event_frame[:, 2] *= 0.000001
            event_frame[:, 3][event_frame[:, 3] == 0.] = -1
            
            self.event_frames.append(event_frame)
            
            coords = torch.FloatTensor(np.unique(event_frame[:, :2], axis=0))
            
            self.coords.append(coords)
            
            coords_int = coords.long()
            
            flow_interpolator = RegularGridInterpolator(
                points = (np.arange(0, flow_gt.shape[0], 1), 
                          np.arange(0, flow_gt.shape[1], 1)),
                values = flow_gt)
            
            flows = flow_interpolator(torch.flip(coords, dims=[1]) * flow_gt.shape[0] / 260)
            
            self.flows.append(torch.FloatTensor(flows))
    
    
    def __len__(self) :
        return len(self.event_frames)
        
    
    def __getitem__(self, idx) :
        return self.event_frames[idx], self.coords[idx], self.flows[idx]