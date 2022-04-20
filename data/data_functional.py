import numpy as np


def crop_flow_frame(data, crop, offset) :
    pass

class CropFlowFrameEventData :
    def __init__(self, crop=(64, 64), offset='center') :
        self.crop = crop
        self.offset = offset
    def __call__(self, data) :
        if self.offset == 'center' :
            offset = ((data['res'][0] - self.crop[0]) // 2,
                      (data['res'][1] - self.crop[1]) // 2)
        elif self.offset == 'random' :
            offset = (np.random.randint(0, data['res'][0] - self.crop[0]),
                      np.random.randint(0, data['res'][1] - self.crop[1]))
        else :
            offset = self.offset
        crop_high = (offset[0] + self.crop[0],
                     offset[1] + self.crop[1])
        data['flow_frame'] = data['flow_frame'][offset[1] : crop_high[1],
                                                offset[0] : crop_high[0]]
        data['flow_frame_valid'] = data['flow_frame_valid'][offset[1] : crop_high[1],
                                              offset[0] : crop_high[0]]
        x_in_crop = (data['event_data']['x'] >= offset[0]) & (data['event_data']['x'] < crop_high[0])
        y_in_crop = (data['event_data']['y'] >= offset[1]) & (data['event_data']['y'] < crop_high[1])
        in_crop = x_in_crop & y_in_crop
        for k in data['event_data'].keys() :
            data['event_data'][k] = data['event_data'][k][in_crop]
        data['event_data']['x'] -= offset[0]
        data['event_data']['y'] -= offset[1]
        data['res'] = self.crop
        return data


class CropTimeFlowFrameEventData :
    def __init__(self, dt) :
        self.dt = dt

    def __call__(self, data) :
        dt_full = data['dt']
        scale = self.dt / dt_full
        data['flow_frame'] *= scale

        # TODO: could make this searchsorted index
        t_in_crop = ((data['event_data']['t'] - data['ts'][0]) / 1000) < self.dt
        for k in data['event_data'] :
            data['event_data'][k] = data['event_data'][k][t_in_crop]

        data['dt'] = self.dt
        return data
