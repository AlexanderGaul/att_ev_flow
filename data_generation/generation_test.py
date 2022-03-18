import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from seq_generation import *
from hom_generation import *


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--num_seqs", type=int, required=True)
parser.add_argument("--type", type=str, required=True)
parser.add_argument("--setting", type=str)
parser.add_argument("--start_seq", type=int, default=0)

def main() :
    args = parser.parse_args()
    # [] figure out folder structure

    # t_only = (40, 0, 0, 0, 0, 0)
    if args.type == "uni" :
        if args.setting == "extreme" :
            params = (40, 10, 0.4, 0.1, 0.005, 0.3)
        if args.setting == "easy" :
            params = (20, 2.5, 0.1, 0.02, 0.001, 0.2)
        if args.setting == "normal" :
            params = (40, 5, 0.2, 0.04, 0.002, 0.3)

        num_curves = 5
        hom_gen = GenHomUniOffset(*params)
        dt_rnd_back = 0.1
        dt_rnd_fore = 0.2
        min_objects = 2; max_objects = 6

        res_dsec = (1080, 1440)
        crop = (480 // 2, 640 // 2)
        res = (240 // 2, 320 // 2)

        t_seq = 2.

    elif args.type == "leftright" :
        hom_gen = GenHomLeftRight(100)
        num_curves = 2
        dt_rnd_back = 0.
        dt_rnd_fore = 0.
        min_objects, max_objects = 0, 0

        res_dsec = (1080, 1440)
        crop = (240 , 320)
        res = (240 // 2, 320 // 2)

        t_seq = 1.

    seq_writer = SequenceWriter(t_seq,
                                im_fps=500,
                                flow_fps=10,
                                event_file="events/events.h5",
                                flow_dir="flow/forward/",
                                im_dir="ims/",
                                flow_ts_file="flow/forward_timestamps.txt",
                                im_ts_file="image_timestamps.txt")

    homseq_gen_foreground = GenHomSeqSteps(hom_gen, t_seq,
                                           num_curves=num_curves,
                                           dt_rnd=dt_rnd_fore)
    homseq_gen_background = GenHomSeqSteps(hom_gen, t_seq,
                                           num_curves=num_curves,
                                           dt_rnd=dt_rnd_back)

    seq_generator = SequenceGenerator(homseq_gen_background,
                                      homseq_gen_foreground,
                                      crop_offset=((res_dsec[0] - crop[0]) // 2,
                                                   (res_dsec[1] - crop[1]) // 2),
                                      crop=crop, res=res,
                                      min_objects=min_objects, max_objects=max_objects)

    for i in range(args.start_seq, args.num_seqs) :
        name = "0" * (4 - len(str(i))) + str(i)
        print(name)
        seq = seq_generator(i)
        seq_writer.write_sequence(seq, Path(args.dir) / name, write_video=True)


if __name__ == "__main__" :
    main()

