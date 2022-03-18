import numpy as np

import esim_py

import h5py

from pathlib import Path


# TODO: create h5py dataset

# Ticket 002 - week 08 - WAITING FOR DEPLOYMENT
# TODO: package function into usable interface
# [x] should we do this in an object-oriented way?


class EventGenerator :
    def __init__(self, thres_pos=0.2, thres_neg=0.2) :
        self.esim = esim_py.EventSimulator(
            thres_pos, # contrast thesholds for positive
            thres_neg, # and negative events
            0, # minimum waiting period (in sec) before a pixel can trigger a new event
            1e-8, # epsilon that is used to numerical stability within the logarithm
            True # wether or not to use log intensity
            )

    def generate_events(self, im_path, ts_path) :
        events_from_images = self.esim.generateFromFolder(
            str(im_path),
            # absolute path to folder that stores images in numbered order
            str(ts_path)
            # absolute path to timestamps file containing one timestamp (in secs) for each
        )
        return events_from_images

    @staticmethod
    def write_events(events_from_images, event_file_path) :
        num_events = len(events_from_images)
        events_from_images[:, 3][events_from_images[:, 3] < 1] = 0
        # Ticket 001 - week 08 - Thu 03 March - WAITING FOR TESTING
        # [] test by re executing dataset
        # TODO: Write generated events to h5py file format
        # [x] figure out data format of DSEC event files
        # xy: '<u2', t: '<u4', p: '|u1'
        # [x] check if time in us : True
        # [x] create dataset file
        ef_out = h5py.File(Path(event_file_path), 'a')
        ef_out.clear()

        # [x] create event subgroups/datasets
        event_grp = ef_out.create_group('/events')
        event_grp.create_dataset('p', shape=(num_events,),
                                 dtype='|u1')
        event_grp.create_dataset('t', shape=(num_events,),
                                 dtype='<u4')
        event_grp.create_dataset('x', shape=(num_events,),
                                 dtype='<u2')
        event_grp.create_dataset('y', shape=(num_events,),
                                 dtype='<u2')


        # [x] write event data to datastes
        # [x] check if first column is actually x: True
        # [x] time of generated events: is in s
        event_grp['x'][:] = events_from_images[:, 0].astype(int)
        event_grp['y'][:] = events_from_images[:, 1].astype(int)
        event_grp['t'][:] = (events_from_images[:, 2] * 1e6).astype(int)
        event_grp['p'][:] = events_from_images[:, 3].astype(int)

        # [x] create ms to index dataset
        last_ms = np.ceil(events_from_images[-1, 2] * 1e3)
        ef_out.create_dataset('ms_to_idx',
                              shape=(last_ms+1,),
                              dtype='<u8')
        # [x] compute ms to idx
        # [x] how to handle boundaries?
        # t[ms_to_idx[ms] - 1] < ms*1000 <= t[ms_to_idx[ms]]
        # [~] shouldn't we want an inclusive boundary on the right?? -> Ticket 003
        # Note: DSEC example code actually computes tight window after conservative
        # but does not include the boundary itself
        ms_to_idx = np.searchsorted(events_from_images[:, 2] * 1e3,
                                    np.arange(0, last_ms+1))
        ef_out['ms_to_idx'][:] = ms_to_idx
        # [x] adapt dsec to handle event files without 't_offset'

        # [x] close event file
        ef_out.flush()
        ef_out.close()

# Ticket 004 - week 08
# TODO: move execution to different file
# [] need to figure out class structure for dataset generation
"""
im_path = "/storage/user/gaul/gaul/thesis/output/synth/ims"
ts_path = "/storage/user/gaul/gaul/thesis/output/synth/timestamps.txt"
out_path = "/storage/user/gaul/gaul/thesis/output/synth/"

writer = EventGenerator()
writer.write_events(im_path, ts_path, out_path)
"""