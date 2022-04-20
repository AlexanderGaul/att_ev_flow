import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2 as cv
import math

import h5py
import hdf5plugin

from event_slicer import EventSlicer

from plot import create_event_picture

import matplotlib.pyplot as plt

file_path = "/storage/user/gaul/gaul/thesis/data/Test/00000/events/events.h5"

t_begin_s = 0.01
t_end_s = 1.

fps = 100

frame_time = 1 / fps

eventremain_time = 0.

event_file = h5py.File(file_path)

event_slicer = EventSlicer(event_file)

res = (48, 48)


ts = np.linspace(t_begin_s, t_end_s, math.ceil(fps * (t_end_s - t_begin_s) + 1))

vid = cv.VideoWriter("/storage/user/gaul/gaul/thesis/data/Test/00000/event_vid.avi",
                     0, fps, (res[1], res[0]))

for i in range(len(ts) - 1) :

    events = event_slicer.get_events((ts[i] - eventremain_time) * 1e6 + event_slicer.t_offset,
                                     ts[i+1] * 1e6 + event_slicer.t_offset)

    event_array = np.concatenate([events['x'][:, None].astype(float),
                                  events['y'][:, None].astype(float),
                                  events['t'][:, None].astype(float) / 1e3,
                                  events['p'][:, None].astype(float) * 2 - 1], axis=1)
    im = create_event_picture(event_array, res)
    im = (im * 255).astype(np.uint8)
    imt = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    #plt.imshow(im)
    #plt.show()
    vid.write(im)

vid.release()

