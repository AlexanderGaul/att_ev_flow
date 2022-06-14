import sys
import os

from data_generation.seq_writer import SequenceWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import argparse

from seq_generation import *
from hom_generation import *

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--num_seqs", type=int, required=True)
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--setting", type=str)
parser.add_argument("--start_seq", type=int, default=0)
parser.add_argument("--end_seq", type=int, default=0)
parser.add_argument("--min_events_per_frame", type=int, default=100)
parser.add_argument("--max_events_per_frame", type=int, default=10000000)


def main() :
    t = time.time()
    args = parser.parse_args()

    num_curves = 5

    dt_rnd_back = 0.
    dt_rnd_fore = 0.
    min_objects = 0
    max_objects = 0

    res_dsec = (1080, 1440)
    crop = (480 // 2, 640 // 2)
    res = (64, 64)

    t_seq = 2.

    params = (48, 0, 0., 0., 0., 0.)

    hom_gen = GenHomSeqIID(GenHomUniOffset(*params))

    seq_writer = SequenceWriter(t_seq,
                                im_fps=500,
                                flow_fps=10,
                                event_file="events/events.h5",
                                flow_dir="flow/forward/",
                                flow_backward_dir="flow/backward/",
                                im_dir="ims/",
                                flow_ts_file="flow/forward_timestamps.txt",
                                flow_backward_ts_file="flow/backward_timestamps.txt",
                                im_ts_file="image_timestamps.txt",
                                event_count_file="events/flow_frames_event_counts.txt",
                                min_events_per_frame=args.min_events_per_frame,
                                max_events_per_frame=args.max_events_per_frame)

    homseq_gen_background = GenHomSplineSeq(hom_gen, t_seq, num_curves, dt_rnd_fore)

    seq_generator = SequenceGenerator(homseq_gen_background,
                                      None,
                                      crop_offset=((res_dsec[0] - crop[0]) // 2,
                                                   (res_dsec[1] - crop[1]) // 2),
                                      crop=crop, res=res,
                                      min_objects=min_objects, max_objects=max_objects,
                                      blur_image=True, random_offset=False)

    for i in range(args.start_seq, max(args.start_seq + args.num_seqs, args.end_seq)):
        name = "0" * (5 - len(str(i))) + str(i)
        print("------------------")
        print("Sequence: " + name)
        seq = seq_generator(i)
        while not seq_writer.write_sequence(seq, Path(args.dir) / name, write_video=True):
            print("repeat generation")
            seq = seq_generator()
        print("Sequence written")

    print("Time: " + str(time.time() - t))


if __name__ == "__main__" :
    main()

