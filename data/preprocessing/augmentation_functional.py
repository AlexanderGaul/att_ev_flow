import numpy as np


def event_array_flip_horizontal(event_array, res) :
    event_array[:, 0] = -event_array[:, 0] + res[0] - 1
    return event_array

def event_array_flip_vertical(event_array, res) :
    event_array[:, 1] = -event_array[:, 1] + res[1] - 1
    return event_array

def frame_flip_horizontal(frame) :
    return np.flip(frame, axis=1).copy()

def frame_flip_vertical(frame) :
    return np.flip(frame, axis=0).copy()


class EventArrayFlipHorizontal :
    def __call__(self, data) :
        data['event_array'] = event_array_flip_horizontal(data['event_array'],
                                                          data['res'])
        return data

class EventArrayFlipVertical :
    def __call__(self, data) :
        data['event_array'] = event_array_flip_vertical(data['event_array'],
                                                          data['res'])
        return data

class FlowFrameFlipHorizontal :
    def __call__(self, data) :
        data['flow_frame'] = frame_flip_horizontal(data['flow_frame'])
        data['flow_frame'][..., 0] *= -1
        data['flow_frame_valid'] = frame_flip_horizontal(data['flow_frame_valid'])
        return data

class FlowFrameFlipVertical :
    def __call__(self, data) :
        data['flow_frame'] = frame_flip_vertical(data['flow_frame'])
        data['flow_frame'][..., 1] *= -1
        data['flow_frame_valid'] = frame_flip_vertical(data['flow_frame_valid'])
        return data