import numpy as np

from data.preprocessing.event_functional import *
from events import spatial_downsample

from utils import dist2rect


class FlowAtEvents :
    def __init__(self) : pass
    def __call__(self, data) :
        e_mask = event_mask(data['event_array'], data['res'], data['dt'])
        data['flow_frame_eval'] = data['flow_frame_eval'] & e_mask


class EventMask :
    def __init__(self, area=1) : self.area = area
    def __call__(self, data) : return event_mask(data['event_array'], data['res'], data['dt'], self.area)


class EventArrayContext :
    def __init__(self, patch_size=3, num_events=4, mode='volume', t_relative=True) :
        self.patch_size = patch_size
        self.num_events = num_events
        self.mode = mode
        self.t_relative = t_relative
    def __call__(self, data) :
        if self.mode == 'volume' :
            event_context = event_array_patch_context(data['event_array'],
                                                      data['res'], data['dt'],
                                                      self.num_events,
                                                      self.patch_size,
                                                      self.t_relative)
        elif self.mode == 'time_surface' :
            event_context = event_array_patch_time_surface(data['event_array'],
                                                           data['res'], data['dt'],
                                                           self.patch_size,
                                                           self.t_relative)
        data['event_array'] = np.concatenate([data['event_array'], event_context], axis=1)
        return data


class EventDataCrop :
    def __init__(self, offset, res) :
        self.offset = offset
        self. res = res
    def __call__(self, data) :
        x_valid = data['event_data']['x'] >= self.offset[1] & data['event_data']['x'] < self.offset[1] + self.res
        y_valid = data['event_data']['y'] >= self.offset[0] & data['event_data']['y'] < self.offset[0] + self.res
        valid = x_valid & y_valid

        for key in data['event_data'].keys() :
            data['event_data'][key] = data['event_data'][key][valid]
        data['event_data']['x'] -= int(self.offset[1])
        data['event_data']['y'] -= int(self.offset[0])
        data['res'] = (self.res[1], self.res[0])
        data['offset'] = self.offset
        return data


class EventData2EventArray :
    def __init__(self) : pass
    def __call__(self, data) :
        data['event_array'] = event_data_2_array(data.pop('event_data'),
                                                 min(data['ts']))
        return data


class EventArray2ImagePair :
    def __init__(self) : pass
    def __call__(self, data):
        data['im1'], data['im2'] = event_array_2_image_pair(data['event_array'], data['res'], data['dt'])
        del data['event_array']
        return data


class Dropout :
    def __init__(self, key='event_array', p=0.2) :
        self.key = key
        self.p = p
    def __call__(self, data_list) :
        l = len(data_list[0]['event_array'])
        dropout = np.random.binomial(1, 1 - self.p, l).astype(bool)
        for data in data_list :
            data[self.key] = data[self.key][dropout]
        return data_list

class Noise :
    def __init__(self, l, sort) :
        self.sort = sort
        self.l = l
    def __call__(self, data_list) :
        noise_events = event_noise(self.l,
                                   data_list[0]['dt'],
                                   tuple(reversed(data_list[0]['res'])),
                                   self.sort)
        for data in data_list :
            if self.sort :
                data['event_array'] = merge_sorted(data['event_array'],
                                                   noise_events,
                                                   axis=2)
            else :
                data['event_array'] = np.concatenate([data['event_array'],
                                                      noise_events],
                                                     axis=0)

        return data_list



class ArrayRemoveZeros :
    def __init__(self,
                 key='event_array',
                 raw_dims=[3]) :
        self.key =  key
        self.raw_dims = raw_dims
    def __call__(self, data) :
        data[self.key] = event_array_remove_zeros(data[self.key], self.raw_dims)
        return data


class EventArrayBackward :
    def __init__(self) : pass
    def __call__(self, data) :
        data['event_array'] = event_array_backward(data.pop('event_array'),
                                                   data['dt'])
        return data

"""
class EventDataCat :
    def __call__(self, data_1, data_2) :
        event_data_1 = data_1['event_data']
        event_data_2 = data_2['event_data']
        event_data = {}
        for k in event_data_1.keys() :
            event_data[k] = np.concatenate([event_data_1[k],
                                            event_data_2[k]])
            
        return {**data_1, **data_2, }
"""

class EventArraySpatialDownsample :
    def __init__(self, factor=1) : self.factor = factor
    def __call__(self, data) :
        if self.factor != 1 :
            data['event_array'] = spatial_downsample(data.pop('event_array'),
                                                     self.factor)
            data['res'] = (data['res'][0] // self.factor,
                           data['res'][1] // self.factor)
        return data


class EventArrayUndistortMap :
    def __init__(self, rectify_map) : self.rectify_map = rectify_map
    def __call__(self, data) :
        xy = data['event_array'][:, :2].astype(int)
        data['event_array'][:, :2] = self.rectify_map[xy[:, 1], xy[:, 0]]
        return data

class EventArrayUndistortIntrinsics :
    def __init__(self, intrinsics, scale=1) : self.intrinsics = intrinsics; self.scale = scale
    def __call__(self, data) :
        if self.scale == 1 :
            data['event_array'][:, :2] = dist2rect(data['event_array'][:, :2], *self.intrinsics)
        else :
            if len(data['event_array']) > 0 :
                xys = data['event_array'][:, :2]
                xys = xys * self.scale + self.scale / 2
                xys = dist2rect(xys, *self.intrinsics)
                xys = xys - self.scale / 2
                xys = xys / self.scale
                data['event_array'][:, :2] = xys
        return data