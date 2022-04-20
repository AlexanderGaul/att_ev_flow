import numpy as np

import h5py

import matplotlib.pyplot as plt

from dsec import DSEC

from events import *

from plot import create_event_picture, plot_flow

import time

def write_downsampling(d:DSEC, patch_size, seq_idxs, name) :
    assert name != 'left'
    assert name != 'right'

    for seq_idx in seq_idxs :
        pass
        ev_downsample = downsample(d.event_files[seq_idx], patch_size)
        
        


def downsample(ef : h5py.File, patch_res) :
    res = (480, 640)
    frame = np.zeros((res[0] / patch_res, res[1] / patch_res))

    num_events = ef['events'].shape[0]

    events = np.zeros([num_events // patch_res**2, ])
    idx = 0

    for i in range(num_events) :
        x = ef['events']['x'][i]
        y = ef['events']['y'][i]
        p = ef['events']['p'][i]

        frame[y, x] += p * (1 / patch_res ** 2)
        if frame[y, x] >= 1 :
            frame[y, x] -= 1
            events[idx, :] = (x, y, ef['events']['t'][i], 1)
        elif frame[y, x] <= -1 :
            frame[y, x] += 1
            events[idx, :] = (x, y, ef['events']['t'][i], -1)
        idx += 1
        if idx == len(events) :
            events_new = np.zeros([2 * len(events), 4])
            events_new[:len(events), :] = events
            events = events_new

    return events[:idx, :]

def main() :
    patch_res = 4
    res = (480, 640)
    frame = np.zeros((res[0] // patch_res, res[1] // patch_res))

    d = DSEC()
    ev = d[0]['events']
    co = d[0]['coords']
    fl = d[0]['flows']

    num_events = len(ev)

    t_begin = time.time()
    ev_downsampled = spatial_downsample(ev, patch_res)
    print(time.time() - t_begin)

    t_begin = time.time()
    ev_downsampled = spatial_downsample(ev, patch_res)
    print(time.time() - t_begin)




    t_begin = time.time()
    co_down, fl_down = downsample_flow_jit(co, fl, patch_res)
    print(time.time() - t_begin)

    t_begin = time.time()
    co_down_jit, fl_down_jit = downsample_flow_jit(co, fl, patch_res)
    print(time.time() - t_begin)

    t_begin = time.time()
    co_down, fl_down = downsample_flow(co, fl, patch_res)
    print(time.time() - t_begin)

    print(num_events)
    print(len(ev_downsampled))

    e_img = create_event_picture(ev_downsampled, frame.shape)
    plot_flow(e_img, co_down, fl_down, freq=1)
    plt.show()

if __name__ == "__main__" :
    main()