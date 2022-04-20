import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math

import h5py
import imageio
import shutil

import argparse
from pathlib import Path

from utils import get_grid_coordinates


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--start_seq", type=int, required=True)
parser.add_argument("--end_seq", type=int, required=True)
parser.add_argument("--group", action='store_true')
parser.add_argument("--group_size", type=int)

def main() :
    pass
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    start_seq = args.start_seq
    end_seq = args.end_seq

    if args.group :
        for i in range(int(math.ceil(end_seq - start_seq) / args.group_size)) :
            out_folder = '0' * (5 - len(str(i))) + str(i)
            combine_sequences(output_dir / out_folder, input_dir,
                              start_seq + args.group_size * i,
                              min(start_seq + args.group_size * (i+1), end_seq))

    else :
        combine_sequences(input_dir, output_dir, start_seq, end_seq)


def combine_sequences(output_dir, input_dir, start_seq, end_seq) :
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(output_dir / "events"): os.makedirs(output_dir / "events")
    if not os.path.exists(output_dir / "flow"): os.makedirs(output_dir / "flow")
    if not os.path.exists(output_dir / "flow/backward"): os.makedirs(output_dir / "flow/backward")
    if not os.path.exists(output_dir / "flow/forward"): os.makedirs(output_dir / "flow/forward")



    input_dir_seq_names = sorted(os.listdir(input_dir))

    seq_names = input_dir_seq_names[start_seq:end_seq]


    # TODO: load resolution
    ffnames = sorted(sorted(os.listdir(input_dir / input_dir_seq_names[0]  /"flow/forward/")))
    res = imageio.imread(input_dir / input_dir_seq_names[0]  / "flow/forward/" / ffnames[0]).shape[:2]


    count_events = 0
    count_ms = 0
    t_current_ms = 0
    duration_last_ms = 0

    for seq_idx, seq_name in enumerate(seq_names) :
        event_file = h5py.File(input_dir / seq_name / "events/events.h5")
        count_events += len(event_file['events/x'])
        count_ms += len(event_file['ms_to_idx'])

        t_current_ms += len(event_file['ms_to_idx'])

        if (seq_idx < len(seq_names) - 1) :

            t_next_second_ms = (t_current_ms // 1000 + 1) * 1000
            t_current_ms = t_next_second_ms + 1000
        event_file.close()

    t_total_ms = t_current_ms

    dummy_frame_num_events = (res[0] * res[1]) * 4

    new_events = count_events + (len(seq_names) - 1) * dummy_frame_num_events

    ef_out = h5py.File(output_dir / "events/events.h5", 'a')
    ef_out.clear()
    event_grp = ef_out.create_group('/events')
    event_grp.create_dataset('p', shape=(new_events,),
                             dtype='|u1')
    event_grp.create_dataset('t', shape=(new_events,),
                             dtype='<u4')
    event_grp.create_dataset('x', shape=(new_events,),
                             dtype='<u2')
    event_grp.create_dataset('y', shape=(new_events,),
                             dtype='<u2')

    ef_out.create_dataset('ms_to_idx',
                              shape=(t_total_ms,),
                              dtype='<u8')

    open(output_dir / "events" / "flow_frames_event_counts.txt", "w").close()
    ecf = open(output_dir / "events" / "flow_frames_event_counts.txt", "ab")

    open(output_dir / "flow" / "forward_timestamps.txt", "w").close()
    fftsf = open(output_dir / "flow" / "forward_timestamps.txt", "ab")

    open(output_dir / "flow" / "backward_timestamps.txt", "w").close()
    fbtsf = open(output_dir / "flow" / "backward_timestamps.txt", "ab")
    #fbtsf.truncate(0)

    ffcount = 0
    fbcount = 0
    e_idx = 0
    t_current_ms = 0

    # is the current ms already occupied : NO it is not

    for seq_idx, seq_name in enumerate(seq_names) :
        # TODO: copy events into new file
        ef_in = h5py.File(input_dir / seq_name / "events/events.h5")
        e_count = len(ef_in['events/x'])
        t_add = ef_in['events/t'][-1]
        ef_out['events/x'][e_idx:e_idx + e_count] = ef_in['events/x'][:]
        ef_out['events/y'][e_idx:e_idx + e_count] = ef_in['events/y'][:]
        ef_out['events/t'][e_idx:e_idx + e_count] = ef_in['events/t'][:] + t_current_ms * 1000
        ef_out['events/p'][e_idx:e_idx + e_count] = ef_in['events/p'][:]

        count_ms = len(ef_in['ms_to_idx'])

        # TODO: add at least one second
        # always start sequence at even second


        # TODO: maybe need to add indices before
        ef_out['ms_to_idx'][t_current_ms:t_current_ms+count_ms] = ef_in['ms_to_idx'][:] + e_idx



        np.savetxt(fftsf, np.loadtxt(input_dir / seq_name / "flow/forward_timestamps.txt", dtype=np.int64) + t_current_ms * 1000, fmt='%d')
        np.savetxt(fbtsf, np.loadtxt(input_dir / seq_name / "flow/backward_timestamps.txt", dtype=np.int64) + t_current_ms * 1000, fmt='%d')
        np.savetxt(ecf, np.loadtxt(input_dir / seq_name / "events" / "flow_frames_event_counts.txt", dtype=np.int64), fmt='%d')

        e_idx += e_count
        t_current_ms += count_ms

        if (seq_idx < len(seq_names) - 1) :
            t_next_second_ms = (t_current_ms // 1000 + 1) * 1000
            t_dummy_frame_ms = t_next_second_ms + 500
            t_overnext_second_ms = t_next_second_ms + 1000   # this should be the start of the next sequence

            xy = get_grid_coordinates((res[1]*2, res[0]*2))
            ef_out['events/x'][e_idx:e_idx + dummy_frame_num_events] = xy[:, 0].astype(np.uint16)
            ef_out['events/y'][e_idx:e_idx + dummy_frame_num_events] = xy[:, 1].astype(np.uint16)
            ef_out['events/t'][e_idx:e_idx + dummy_frame_num_events] = t_dummy_frame_ms * 1000
            ef_out['events/p'][e_idx:e_idx + dummy_frame_num_events] = np.ones(dummy_frame_num_events, dtype=np.uint8)

            ef_out['ms_to_idx'][t_current_ms:t_dummy_frame_ms+1] = e_idx

            e_idx += dummy_frame_num_events

            ef_out['ms_to_idx'][t_dummy_frame_ms + 1:t_overnext_second_ms] = e_idx  # does not exist yet?

            t_current_ms = t_overnext_second_ms


        ffnames = sorted(sorted(os.listdir(input_dir / seq_name / "flow/forward/")))
        for fname in ffnames :
            fsplit = fname.split('.')
            digits = len(fsplit[0])
            fidx = int(fsplit[0])
            fname_new = '0' * (digits - len(str(fidx + ffcount))) + str(fidx + ffcount) + "." + fsplit[1]
            shutil.copy(input_dir / seq_name / "flow/forward/" / fname, output_dir / "flow/forward" / fname_new)
        ffcount += len(ffnames)

        fbnames = sorted(sorted(os.listdir(input_dir / seq_name / "flow/backward/")))
        for fname in fbnames:
            fsplit = fname.split('.')
            digits = len(fsplit[0])
            fidx = int(fsplit[0])
            fname_new = '0' * (digits - len(str(fidx + fbcount))) + str(fidx + fbcount) + "." + \
                        fsplit[1]
            shutil.copy(input_dir / seq_name / "flow/backward/" / fname, output_dir / "flow/backward" / fname_new)
        fbcount += len(fbnames)

        # TODO: append event counts
    ecf.close()
    fbtsf.close()
    fftsf.close()

    ef_out.flush()
    ef_out.close()


if __name__ == "__main__" :
    main()
