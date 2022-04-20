import numpy as np
import time
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
                 input_name='volume',
                 output_name='patch_array',
                 format='torch') :
        self.patch_size=patch_size
        self.stride = stride
        self.input_name = input_name
        self.output_name = output_name
        self.format = format
    def __call__(self, data) :
        volume = data.pop(self.input_name)
        coords, patch_array = volume_2_patch_array(volume,
                                                   self.patch_size,
                                                   self.stride,
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
                 patch_size=3, stride=1,
                 input_name='event_volume', output_name='event_array',
                 format='torch'):
        self.patch_size = patch_size; self.stride = stride
        self.input_name = input_name; self.output_name = output_name
        self.format = format

    def __call__(self, data) :
        volume = data.pop(self.input_name)
        coords, ts, patch_array = volume_2_3d_patch_array(volume, self.patch_size, self.stride,
                                                          data['dt'], self.format)
        data[self.output_name] = np.concatenate([coords,
                                                 ts[:, None],
                                                 patch_array], axis=1)
        return data