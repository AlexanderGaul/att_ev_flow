import sys
import os

from data_generation.seq_writer import SequenceWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import argparse

from multiprocessing.pool import ThreadPool

from functools import partial


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
parser.add_argument("--multiprocessing", type=int, default=1)

def main() :
    t = time.time()
    args = parser.parse_args()
    # [] figure out folder structure

    # t_only = (40, 0, 0, 0, 0, 0)
    if args.type == "uni" :
        num_curves = 5

        dt_rnd_back = 0.1
        dt_rnd_fore = 0.2
        min_objects = 2;
        max_objects = 6

        res_dsec = (1080, 1440)
        crop = (480 // 2, 640 // 2)
        res = (240 // 2, 320 // 2)

        t_seq = 2.

        if args.setting == "extreme" :
            params = (40, 10, 0.4, 0.1, 0.005, 0.3)
        if args.setting == "easy" :
            params = (20, 2.5, 0.1, 0.02, 0.001, 0.2)
        if args.setting == "normal" :
            params = (40, 5, 0.2, 0.04, 0.002, 0.3)
        if args.setting == "no_foreground" :
            num_curves = 5
            params = (4, 4, 0.15, 0.015, 0.0015, 0.15)
            #params = (10, 4, 0.3, 0.03, 0.003, 0.3)
            min_objects = 0
            max_objects = 0
            dt_rnd_back = 0.
            dt_rnd_fore = 0.
            blur_image = True
            crop = (48*4, 48*4)
            res = (48, 48)
            t_seq = 1
        if args.setting == "one_foreground" :
            num_curves = 5
            params = (25, 25, 0.2, 0.1, 0.0015, 0.2)
            #params = (0, 0, 0, 0.15, 0.002, 0.2)
            #params = (10, 4, 0.3, 0.03, 0.003, 0.3)
            params_obj = (30, 30, 0.5, 0.1, 0.002, 0.4)
            min_objects = 1
            max_objects = 1
            dt_rnd_back = 0.
            dt_rnd_fore = 0.
            blur_image = True
            crop = (48*4, 48*4)
            res = (48, 48)
            t_seq = 1
        if args.setting == "one_foreground_highres" :
            num_curves = 5
            params = (25, 25, 0.2, 0.1, 0.0015, 0.2)
            #params = (0, 0, 0, 0.15, 0.002, 0.2)
            #params = (10, 4, 0.3, 0.03, 0.003, 0.3)
            params_obj = (30, 30, 0.5, 0.1, 0.002, 0.4)
            min_objects = 1
            max_objects = 1
            dt_rnd_back = 0.
            dt_rnd_fore = 0.
            blur_image = True
            crop = (48*4, 48*4)
            res = (48*2, 48*2)
            t_seq = 1
        if args.setting == "more_foreground_highres" :
            num_curves = 5
            params = (40, 40, 0.2, 0.1, 0.0015, 0.2)
            #params = (0, 0, 0, 0.15, 0.002, 0.2)
            #params = (10, 4, 0.3, 0.03, 0.003, 0.3)
            params_obj = (50, 50, 0.5, 0.1, 0.002, 0.4)
            min_objects = 1
            max_objects = 5
            dt_rnd_back = 0.
            dt_rnd_fore = 0.
            blur_image = True
            crop = (240, 320)
            res = (120, 160)
            t_seq = 1
        if args.setting == "more_foreground_fullres" :
            num_curves = 5
            params = (40, 40, 0.2, 0.05, 0.0005, 0.2)
            #params = (0, 0, 0, 0.15, 0.002, 0.2)
            #params = (10, 4, 0.3, 0.03, 0.003, 0.3)
            params_obj = (50, 50, 0.5, 0.1, 0.002, 0.4)
            min_objects = 5
            max_objects = 10
            dt_rnd_back = 0.
            dt_rnd_fore = 0.
            blur_image = True
            crop = (480, 640)
            res = (480, 640)
            t_seq = 1

        hom_gen = GenHomSeqIID(GenHomUniOffset(*params))
        hom_gen_obj = GenHomSeqIID(GenHomUniOffset(*params_obj))

    elif args.type == "offset" :
        if args.setting == "patch" :
           params = (20, 10, 0., 0., 0., 0.)
        num_curves = 10
        hom_gen = GenHomSeqIID(GenHomUniOffset(*params))
        hom_gen_obj = hom_gen
        dt_rnd_back = 0.
        dt_rnd_fore = 0.2
        min_objects = 0; max_objects = 0

        res_dsec = (1080, 1440)
        crop = (128, 128)
        res = (64, 64)

        t_seq = 2.

    elif args.type == "leftright" :
        if args.setting == "var_length" :
            hom_gen = GenHomSeqLeftRightVarLength(length_min=5,
                                                  length_max=10)
            hom_gen_obj = hom_gen
            num_curves = 10

        else :
            hom_gen = GenHomSeqIID(GenHomLeftRight(100))
            hom_gen_obj = hom_gen
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
                                flow_backward_dir="flow/backward/",
                                im_dir="ims/",
                                flow_ts_file="flow/forward_timestamps.txt",
                                flow_backward_ts_file="flow/backward_timestamps.txt",
                                im_ts_file="image_timestamps.txt",
                                event_count_file="events/flow_frames_event_counts.txt",
                                min_events_per_frame=args.min_events_per_frame,
                                max_events_per_frame=args.max_events_per_frame)

    homseq_gen_foreground = GenHomCurveSeqSteps(hom_gen_obj, t_seq,
                                                num_curves=num_curves,
                                                dt_rnd=dt_rnd_fore)
    homseq_gen_background = GenHomCurveSeqSteps(hom_gen, t_seq,
                                                num_curves=num_curves,
                                                dt_rnd=dt_rnd_back)

    seq_generator = SequenceGenerator(homseq_gen_background,
                                      homseq_gen_foreground,
                                      crop_offset=((res_dsec[0] - crop[0]) // 2,
                                                   (res_dsec[1] - crop[1]) // 2),
                                      crop=crop, res=res,
                                      min_objects=min_objects, max_objects=max_objects,
                                      blur_image=blur_image)



    if args.multiprocessing > 1 :
        with ThreadPool(args.multiprocessing) as p :
            p.map(partial(create_sequence,
                          seq_generator=seq_generator,
                          seq_writer=seq_writer,
                          args=args),
                  range(args.start_seq, max(args.start_seq+args.num_seqs, args.end_seq)))
    else :
        for i in range(args.start_seq, max(args.start_seq+args.num_seqs, args.end_seq)) :
            create_sequence(i, seq_generator, seq_writer, args)

    print("Time: " + str(time.time() - t))

def create_sequence(i, seq_generator, seq_writer, args) :
    name = "0" * (5 - len(str(i))) + str(i)
    print("------------------")
    print("Sequence: " + name)
    seq = seq_generator(i)
    while not seq_writer.write_sequence(seq, Path(args.dir_seg) / name, write_video=True):
        print("repeat generation")
        seq = seq_generator()
    print("Sequence written: " + name)

if __name__ == "__main__" :
    main()

