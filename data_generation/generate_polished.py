import sys
import os
sys.path = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + sys.path

import math
import time

import argparse

from multiprocessing.pool import ThreadPool

from functools import partial

import matplotlib.pyplot as plt

from plot import flow_frame_color
from data_generation.datasets import *


from seq_generation import *
from hom_generation import *

"""
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
"""

def main() :
    params = (14, 240, 0.1, 0.1, 0.0005, 0.4)
    params_obj = (32, 360, 0.5, 0.05, 0.00025, 0.4)
    params_mid = (0, 20, 0.1, 0.05, 0.0001, 0.1)

    hom_gen = GenHomSeqIID(GenHomTtailedUni(*params, dropout=0.5))
    hom_gen_mid = GenHomSeqIID(GenHomUniOffsetPolished(*params_mid, dropout=0.5))
    hom_gen_obj = GenHomSeqIID(GenHomTtailedUni(*params_obj, dropout=0.5))

    t_seq = 1
    dt_rnd = 0.05
    num_keypoints = 6

    homseq_gen_background = GenHomSplineSeqSteps(hom_gen, t_seq, num_keypoints, dt_rnd, first_is_id=True)
    homseq_gen_foreground = GenHomSplineSeqSteps(hom_gen_obj, t_seq, num_keypoints, dt_rnd, first_is_id=True)
    homseq_gen_midground = GenHomSplineSeqStepsDrift(hom_gen_mid, num_keypoints)

    crop = (480, 640)
    res = (480, 640)

    min_middleground = 6
    max_middleground = 10
    min_foreground = 6
    max_foreground = 10

    bins = np.arange(0, 260, 1)
    histogram = np.zeros(len(bins) - 1)



    seq_generator = SequenceGeneratorPolished(PNGDataset("/storage/user/gaul/gaul/thesis/data/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"),
                                              CatRandomDataset([Cityscapes(), MOTSForeground()], np.array([0.75, 0.25])),
                                              homseq_gen_background,
                                              homseq_gen_midground,
                                              homseq_gen_foreground,
                                              crop, res,
                                              min_middleground, max_middleground,
                                              min_foreground, max_foreground)

    seq = seq_generator(6)
    for t in np.linspace(0, 0.9, 10) :
        im = seq.get_image(t)
        plt.imshow(im)
        plt.show()
        flow = seq.get_flow(t, t+0.1)
        plt.imshow(flow_frame_color(flow))
        plt.show()

        flow_norm = np.linalg.norm(flow.reshape((-1, 2)), axis=1)
        histogram += np.histogram(flow_norm.reshape(-1), bins)[0]

    plt.hist(bins[:-1] + 1, bins, weights=histogram)
    plt.show()
    # TODO: histogram

if __name__ == "__main__":
    main()
