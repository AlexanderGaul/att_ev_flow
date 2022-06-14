import sys
import os

from data_generation.seq_writer import SequenceWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from seq_generation import *
from hom_generation import *


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--num_seqs", type=int, required=True)
parser.add_argument("--start_seq", type=int, default=0)

def main() :
    args = parser.parse_args()

    hom_gen = GenHomSeqLeftRightVarLength(length_min=2,
                                          length_max=18)
    num_curves = 5
    t_seq = 0.5

    seq_writer = SequenceWriter(t_seq,
                                im_fps=500,
                                flow_fps=10,
                                event_file="events/events.h5",
                                flow_dir="flow/forward/",
                                im_dir="ims/",
                                flow_ts_file="flow/forward_timestamps.txt",
                                im_ts_file="image_timestamps.txt")

    homcurveseq_gen = GenHomCurveSeqSteps(hom_gen, t_seq,
                                          num_curves=num_curves,
                                          dt_rnd=0.)

    seq_generator = MovingEdgeGenerator(homcurveseq_gen)

    for i in range(args.start_seq, args.start_seq+args.num_seqs) :
        name = "0" * (4 - len(str(i))) + str(i)
        print(name)
        seq = seq_generator(i)
        seq_writer.write_sequence(seq, Path(args.dir) / name, write_video=True)


if __name__ == "__main__" :
    main()
