import numpy as np
import torch

from utils import rect2dist

import os
from pathlib import Path

from PIL import Image
import imageio
import h5py
import hdf5plugin
import yaml

def get_event_left_params(cams_yaml) :
    dist = np.array(cams_yaml['intrinsics']['cam0']['distortion_coeffs'])
    k_dist = np.array(cams_yaml['intrinsics']['cam0']['camera_matrix'])
    res = np.array(cams_yaml['intrinsics']['cam0']['resolution'])
    k_rect = np.array(cams_yaml['intrinsics']['camRect0']['camera_matrix'])
    R = np.array(cams_yaml['extrinsics']['R_rect0'])

    K_rect = np.array([[k_rect[0], 0., k_rect[2]],
                       [0., k_rect[1], k_rect[3]],
                       [0., 0., 1.]])

    K_dist = np.array([[k_dist[0], 0., k_dist[2]],
                       [0., k_dist[1], k_dist[3]],
                       [0., 0., 1.]])

    return res, K_rect, R, dist, K_dist


class DSEC(torch.utils.data.Dataset) :
    def __init__(self,
                 dir = "/storage/user/gaul/gaul/thesis/data/DSEC",
                 seqs = -1, frames = -1):

        self.dir = Path(dir)
        self.seqs = [
             "zurich_city_01_a",
             "zurich_city_02_a",
             #"zurich_city_02_c",
             #"zurich_city_02_d",
             #"zurich_city_02_e"
             #"zurich_city_03_a",
             #"zurich_city_05_a",
             #"zurich_city_05_b"
            ]
        if seqs != -1 :
            self.seqs = [self.seqs[i] for i in seqs]

        # TODO: where to store names of

        self.seqs_flow_names = [sorted(
            os.listdir(self.get_flow_dir(seq))) for seq in self.seqs]
        if frames != -1 :
            self.seqs_flow_names = [ [self.seqs_flow_names[i][j] for j in frames[i]] for i, seq in enumerate(self.seqs)]

        self.seq_lens = np.array([len(self.seqs_flow_names[i])
                                  for i in range(len(self.seqs_flow_names))])

        self.seq_len_csum = self.seq_lens.cumsum()

        self.len = self.seq_lens.sum()

        self.idx2seq_map = np.repeat(np.arange(len(self.seqs)),
                                     self.seq_lens)

        #self.event_files = [h5py.File(self.dir / seq / (seq + "_events_left/events.h5")) for seq in self.seqs]
        #self.cam_files = [yaml.safe_load(open(self.dir / seq / (seq + "_calibration/cam_to_cam.yaml"))) for seq in self.seqs]
        #self.flow_ts_files = [np.loadtxt(self.dir / seq / (seq + "_optical_flow_forward_timestamps.txt"), delimiter=',') for seq in self.seqs]

    def get_frame_name(self, idx):
        seq_idx = self.idx2seq_map[idx]
        if seq_idx > 0:
            idx -= self.seq_len_csum[seq_idx - 1]

        return self.seqs[seq_idx] + "_" + self.seqs_flow_names[seq_idx][idx].split('.')[0]


    def get_flow_dir(self, seq):
        return self.dir / seq / (seq + "_optical_flow_forward_event")

    def get_valid_flows(self, path):
        flow_img = imageio.imread(path, format='PNG-FI')
        indxs = np.argwhere(flow_img[:, :, 2] > 0)
        flows_raw = flow_img[indxs[:, 0],
                             indxs[:, 1], :2]

        flows = (np.array(flows_raw, dtype=np.float32) - 2**15) / 128.
        xys = np.flip(indxs, axis=1).astype(np.float32)

        return flows, xys


    def prep_item(self, seq_idx, gt_idx) :
        seq = self.seqs[seq_idx]
        cam_path = self.dir / seq / (seq + "_calibration/cam_to_cam.yaml")
        flow_ts_path = self.dir / seq / (seq + "_optical_flow_forward_timestamps.txt")

        res, K_rect, R, dist, K_dist = get_event_left_params(yaml.safe_load(open(cam_path)))

        flow_path = self.get_flow_dir(seq) / self.seqs_flow_names[seq_idx][gt_idx]

        flows, xys_rect = self.get_valid_flows(flow_path)
        xys_dist = rect2dist(xys_rect, K_rect, R, dist, K_dist).astype(np.float32)

        flow_ts = np.loadtxt(flow_ts_path, delimiter=',')

        event_file = h5py.File(self.dir / seq / (seq + "_events_left/events.h5"), 'r')# self.event_files[seq_idx]

        ts = flow_ts[gt_idx, :] - event_file['t_offset']
        event_begin_idx = event_file['ms_to_idx'][int(ts[0] / 1000)]
        event_end_idx =  event_file['ms_to_idx'][int(ts[1] / 1000)]

        event_slice = np.concatenate(
            [np.array(event_file['events']['x']
                      [event_begin_idx:event_end_idx]).reshape(-1, 1),
             np.array(event_file['events']['y']
                      [event_begin_idx:event_end_idx]).reshape(-1, 1),
             (np.array(event_file['events']['t']
                      [event_begin_idx:event_end_idx]).reshape(-1, 1) - event_file['events']['t'][event_begin_idx]).astype(np.float32) / 1000,
             np.array(event_file['events']['p']
                      [event_begin_idx:event_end_idx]).reshape(-1, 1)], axis=1)

        event_slice[event_slice[:, 3] < 1, 3] = -1

        return event_slice, xys_dist, flows

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq_idx = self.idx2seq_map[idx]
        if seq_idx > 0 :
            idx -= self.seq_len_csum[seq_idx - 1]

        event_slice, xys, flows = self.prep_item(seq_idx, idx)
        return event_slice, xys, flows, idx



# class DSECPrep(torch.utils.Dataset) :
