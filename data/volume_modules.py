import numpy as np
import time
from data.preprocessing.event_functional import event_array_time_surface_sequence
from data.volume_functional import *


class EventArray2Volume :
    def __init__(self, num_bins,
                 normalize=True,
                 unit_scale=False,
                 one_hot_polarity=False,
                 xyfloat=False) :
        self.num_bins = num_bins
        self.normalize = normalize
        self.unit_scale = unit_scale; assert not (unit_scale and normalize)
        self.one_hot_polarity = one_hot_polarity
        assert not (normalize and one_hot_polarity)
        self.xyfloat = xyfloat

    def __call__(self, data) :
        event_array = data.pop('event_array')
        if self.one_hot_polarity :
            event_array_pos = event_array[event_array[:, 3] > 0., :]
            event_array_neg = event_array[event_array[:, 3] < 0., :]
        volumes = []
        bins = self.num_bins if type(self.num_bins) is list else [self.num_bins]
        for t_bins in bins :
            if not self.one_hot_polarity :
                volume = event_array_2_volume(event_array,
                                              data['res'],
                                              data['dt'],
                                              t_bins,
                                              self.normalize,
                                              self.xyfloat)
                if self.unit_scale :
                    abs_max = np.abs(volume).max()
                    if abs_max > 0 :
                        volume /= abs_max
                volumes.append(volume)
            else :
                event_volume_pos = event_array_2_volume(event_array_pos,
                                                            data['res'],
                                                            data['dt'],
                                                            t_bins // 2,
                                                            self.normalize,
                                                        self.xyfloat)
                event_volume_neg = event_array_2_volume(event_array_neg,
                                                        data['res'],
                                                        data['dt'],
                                                        t_bins // 2,
                                                        self.normalize,
                                                        self.xyfloat)
                volume = np.zeros((self.num_bins, *event_volume_pos.shape[-2:]))
                volume[0::2] = event_volume_pos
                volume[1::2] = event_volume_neg
                volumes.append(volume)
                if self.unit_scale : raise NotImplementedError("")
        data['event_volume'] = np.concatenate(volumes, axis=0)
        return data


class EventVolumeCrop :
    def __init__(self, offset, res) :
        self.offset = offset
        self.res = res
    def __call__(self, data) :
        data['event_volume']  = data['event_volume'][...,
                                self.offset[0]:self.offset[0]+self.res[0],
                                self.offset[1]:self.offset[1]+self.res[1]]
        data['res'] = tuple(reversed(self.res))
        return data


class EventArray2TimeSurfaceVolume :
    def __init__(self, num_bins, crop_ts=False) :
        self.num_bins = num_bins
        self.crop_ts = crop_ts

    def __call__(self, data) :
        event_array = data.pop('event_array')
        res = data['res']
        dt = data['dt']
        last = event_array_time_surface_sequence(event_array, res, dt,
                                                 self.num_bins, False,
                                                 self.crop_ts)
        first = event_array_time_surface_sequence(event_array, res, dt,
                                                  self.num_bins, True,
                                                  self.crop_ts)
        ts_volume = np.concatenate([first, last], axis=-1)
        data['event_volume'] = ts_volume.reshape(res[1], res[0], self.num_bins*4)
        raise NotImplementedError()


class EventVolumeFlat :
    def ___init__(self) : pass
    def __call__(self, data) :
        data['event_array'] = event_volume_flatten_space(data.pop('event_volume'))
        return data


class EventVolume2Array :
    def __init__(self) : pass
    def __call__(self, data) :
        data['event_array'] = event_volume_flatten_spacetime(data.pop('event_volume'),
                                                             data['dt'])
        return data


class EventVolume2ImagePair :
    def __init__(self) : pass
    def __call__(self, data) :
        data['im1'],  data['im2'] = event_volume_2_image_pair(data['event_volume'])
        del data['event_volume']
        return data


class EventVolumeBackward :
    def __init__(self) : pass
    def __call__(self, data) :
        data['event_volume'] = event_volume_backward(data['event_volume'])
        return data


class Volume2PatchArray :
    def __init__(self,
                 patch_size=3,
                 stride=1,
                 padding=1,
                 input_name='volume',
                 output_name='patch_array',
                 format='torch') :
        self.patch_size=patch_size
        self.stride = stride
        self.padding = padding
        self.input_name = input_name
        self.output_name = output_name
        self.format = format
    def __call__(self, data) :
        volume = data.pop(self.input_name)
        coords, patch_array = volume_2_patch_array(volume,
                                                   self.patch_size,
                                                   self.stride,
                                                   self.padding,
                                                   self.format)
        data[self.output_name] = np.concatenate([coords, patch_array], axis=1)


        if self.format == 'torch' :
            res = tuple(reversed(volume.shape[1:]))
        else :
            res = tuple(reversed(volume.shape[:2]))
        return {**data,
                'res' : res}


class EventVolume2PatchArrayFlatTime :
    def __init__(self,
                 patch_size=3, stride=1,
                 input_name='event_volume', output_name='event_array',
                 format='torch') :
        self.patch_size = patch_size; self.stride = stride
        self.input_name = input_name; self.output_name = output_name
        self.format = format
    def __call__(self, data) :
        data[self.output_name] = event_volume_2_patch_array_flatten_time(
            data.pop(self.input_name),
            self.patch_size, self.stride, data['dt'], self.format)
        return data


class Volume23DPatchArray :
    def __init__(self,
                 patch_size=3, stride=1, padding=1,
                 patch_size_3=None, stride_3=None, padding_3=0,
                 input_name='event_volume', output_name='event_array',
                 format='torch',
                 stack_t_dim=False, cat_ts=True):
        self.patch_size = patch_size; self.stride = stride; self.padding = padding
        self.patch_size_3 = patch_size_3; self.stride_3 = stride_3; self.padding_3 = padding_3
        self.input_name = input_name; self.output_name = output_name
        self.format = format
        self.stack_t_dim = stack_t_dim
        self.cat_ts = cat_ts

    def __call__(self, data) :
        volume = data.pop(self.input_name)
        coords, ts, patch_array = volume_2_3d_patch_array(volume, self.patch_size, self.stride, self.padding,
                                                          self.patch_size_3, self.stride_3, self.padding_3,
                                                          dt=data['dt'], format=self.format,
                                                          stack_dim_3=self.stack_t_dim)
        if self.cat_ts :
            data[self.output_name] = np.concatenate([coords,
                                                     ts[:, None],
                                                     patch_array], axis=-1)
        else :
            data[self.output_name] = np.concatenate([coords,
                                                     patch_array], axis=-1)
        return data