import math

import numpy as np
import torch

from events import *
from utils import rect2dist, default,  remap_linear

import os
from pathlib import Path

from PIL import Image
import imageio
import h5py
import hdf5plugin
import tables
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

# TODO: move to utils
def get_grid_coordinates(res, offset=(0, 0)) :
    xs, ys = np.meshgrid(np.arange(offset[0], res[0] + offset[0]),
                         np.arange(offset[1], res[1] + offset[1]))
    return np.concatenate([xs.reshape(-1, 1),
                           ys.reshape(-1, 1)], axis=1)

# TODO: add  offset
def get_idx_in_grid(locs, res) :
    return (locs[:, 0] + locs[:, 1] * res[0]).astype(int)


def read_valid_flows(path, res=None, offset=(0, 0),
                     return_crop_indices=False):
    # return_crop_indices was supposed to provide functionnality when ceeping orignial resolution when cropping
    # these could possibly recovered later on
    # TODO: move to some io file
    flow_img = imageio.imread(path, format='PNG-FI')
    if res is None :
        res = np.flip(flow_img.shape[:2])
    indxs_crop = np.argwhere(flow_img[offset[1]:(res[1]+offset[1]),
                             offset[0]:(res[0]+offset[0]), 2] > 0)
    indxs = indxs_crop + np.flip(offset)
    flows_raw = flow_img[indxs[:, 0],
                indxs[:, 1], :2]

    flows = (np.array(flows_raw, dtype=np.float32) - 2**15) / 128.
    xys = np.flip(indxs, axis=1).astype(np.float32)

    if return_crop_indices :
        print("No guarantee for correct implementation")
        return flows, xys, indxs_crop
    else :
        return flows, xys


def read_flows(path) :
    flow_img = imageio.imread(path, format='PNG-FI')
    flows_raw = flow_img[:, :, :2]
    flows = (np.array(flows_raw, dtype=np.float32) - 2**15) / 128.
    return flows

"""
def flip(locs, res, offset, idx) :
    locs[:, idx] -= offset[idx]
    locs[:, idx] *= -1
    locs[:, idx] += res[idx] - 1
    locs[:, idx] += offset[idx]
    return locs
"""

class DSEC(torch.utils.data.Dataset) :
    def __init__(self,
                 dir = "/storage/user/gaul/gaul/thesis/data/DSEC_flow",
                 seqs = -1,
                 frames = -1,
                 dt = 0,
                 full_query = False,
                 crop = None,
                 random_crop_offset = False,
                 fixed_crop_offset = None,
                 random_moving = False,
                 scale_crop = False,
                 crop_keep_full_res = False,
                 random_flip_horizontal = False,
                 random_flip_vertical = False,
                 include_backward = False,
                 num_bins = 0.,
                 bin_type = 'sum',
                 event_set = 'left'):

        self.dir = Path(dir)
        self.dt = dt
        # TODO: actually implement it
        self.full_query = full_query

        self.crop = crop
        self.random_crop_offset = random_crop_offset
        self.fixed_crop_offset = fixed_crop_offset
        self.random_moving = random_moving

        self.scale_crop = scale_crop
        if self.scale_crop : random_moving = False
        self.crop_keep_full_res = crop_keep_full_res

        self.random_flip_horizontal = random_flip_horizontal
        self.random_flip_vertical = random_flip_vertical

        # are indexed after the forward flows
        # frame indices are incremented to cover the same range of events as corresponding forward flows
        self.include_backward = include_backward

        self.num_bins = num_bins
        self.bin_type = bin_type

        self.event_set = event_set

        self.dsec_seq_names = [
            "zurich_city_01_a",
            "zurich_city_02_a",
            "zurich_city_02_c",
            "zurich_city_02_d",
            "zurich_city_02_e",
            "zurich_city_03_a",
            "zurich_city_05_a",
            "zurich_city_05_b",
            "zurich_city_06_a",
            "zurich_city_07_a",
            "zurich_city_08_a",
            "zurich_city_09_a",
            "zurich_city_10_a",
            "zurich_city_10_b",
            "zurich_city_11_a",
            "zurich_city_11_b",
            "zurich_city_11_c",
            "thun_00_a"
            ]

        if seqs != -1 :
            self.seqs_selected = seqs
            self.seq_names = [self.dsec_seq_names[i]
                              for i in self.seqs_selected]
        else :
            self.seqs_selected = list(range(len(self.dsec_seq_names)))
            self.seq_names = self.dsec_seq_names

        self.seqs_map = {}
        for i, s in enumerate(self.seqs_selected) :
            self.seqs_map[s] = i

        self.res = (640, 480) if not self.crop_keep_full_res and not self.crop else self.crop

        self.seqs_flow_names = [sorted(
            os.listdir(self.get_flow_dir(seq))) for seq in self.seq_names]
        if self.include_backward :
            self.seqs_flow_back_names = [sorted(
                os.listdir(self.get_flow_dir(seq, backward=True))) for seq in self.seq_names]

        # TODO maybe change this later???
        # Currently input files are cropped to selected frames
        if frames != -1 :
            # TODO: manually create list of indices
            # TODO: allow intermittent -1 in list to include all frames for coresponding sequence
            for i, seq in enumerate(self.seq_names) :
                if frames[i] != -1 :
                    self.seqs_flow_names[i] = [self.seqs_flow_names[i][j] for j in frames[i]]
                    if self.include_backward :
                        self.seqs_flow_back_names[i] = [
                            self.seqs_flow_back_names[i]
                                [(j+1) % len(self.seqs_flow_back_names[i])]
                            for j in frames[i]]

        self.seq_lens = np.array([len(self.seqs_flow_names[i])
                                  for i in range(len(self.seqs_flow_names))])
        self.seq_len_csum = self.seq_lens.cumsum()
        self.len = self.seq_lens.sum()

        if self.include_backward :
            self.seq_lens_back = np.array([len(self.seqs_flow_back_names[i])
                                           for i in range(len(self.seqs_flow_back_names))])
            self.seq_len_back_csum = self.seq_lens_back.cumsum()
            self.len_back = self.seq_lens_back.sum()

            self.len_forw = self.len
            self.len = self.len_forw + self.len_back


        # TODO: how to do this with backward
        self.idx2seq_map = np.repeat(np.arange(len(self.seq_names)),
                                     self.seq_lens)
        if self.include_backward :
            self.idx2seq_map_back = np.repeat(np.arange(len(self.seq_names)),
                                              self.seq_lens_back)

        # self.event_files = [h5py.File(self.dir / seq / (seq + "_events_left/events.h5")) for seq in self.seqs]
        self.rectify_maps = [h5py.File(self.dir / seq / "events" / self.event_set / "rectify_map.h5") for seq in self.seq_names]
        self.cams = [yaml.safe_load(open(self.dir / seq / "calibration" / "cam_to_cam.yaml")) for seq in self.seq_names]

        self.flow_ts = []
        for i, seq in enumerate(self.seq_names) :
            self.flow_ts.append(np.loadtxt(self.dir / seq / "flow" /
                                           "forward_timestamps.txt",
                                           delimiter=','))

            if frames != -1 and frames[i] != -1 :
                self.flow_ts[-1] = self.flow_ts[-1][frames[i], :]

            if self.include_backward :
                self.flow_back_ts = [np.loadtxt(self.dir / seq / "flow" /
                                                "backward_timestamps.txt",
                                                delimiter=',')
                                     for seq in self.seq_names]
                if frames != -1 and frames[i] != -1 :
                    # backward flows cover earlier timespans
                    # TODO: find better way of doing this
                    self.flow_back_ts[-1] = self.flow_back_ts[-1][(np.array(frames[i])+1) % len(self.flow_back_ts[-1]), :]
    # end def __init__

    def set_crop(self,
                 crop = None,
                 random_crop_offset = False,
                 fixed_crop_offset = None,
                 random_moving = False,
                 scale_crop = False,
                 crop_keep_full_res = False) :
        self.crop = crop
        self.res = (640, 480) if not self.crop_keep_full_res and not self.crop else self.crop
        self.random_crop_offset = random_crop_offset
        self.fixed_crop_offset = fixed_crop_offset
        self.random_moving = random_moving
        self.scale_crop = scale_crop
        self.crop_keep_full_res = crop_keep_full_res

    def get_local_idx(self, idx):
        if self.include_backward and idx >= self.len_forw :
            idx_back = idx - self.len_forw
            seq_idx = self.idx2seq_map_back[idx_back]
            if seq_idx > 0 :
                idx_back -= self.seq_len_back_csum[seq_idx - 1]
            return seq_idx, idx_back, True
        else :
            seq_idx = self.idx2seq_map[idx]
            if seq_idx > 0:
                idx -= self.seq_len_csum[seq_idx - 1]
            return seq_idx, idx, False

    # TODO move to end of file
    def write_flow_ts_cv(self, seq) :
        # from_timestamp_us, to_timestamp_us, file_index
        file_idx = [int(file_name.split('.')[0])for file_name in self.seqs_flow_names[seq]]

        np.savetxt(self.dir / self.seq_names[seq] / "forward_flow_timestamps.csv",
                   np.concatenate([self.flow_ts[seq],
                                   np.array(file_idx).reshape(-1, 1)],
                                  axis=1),
                   fmt='%i', delimiter=', ',
                   header="from_timestamp_us, to_timestamp_us, file_index")

    def get_seq_len(self, seq):
        l = len(self.seqs_flow_names[seq])
        if self.include_backward :
            l += len(self.seqs_flow_back_names)
        return l

    def get_frame_name(self, idx) :
        seq_idx, idx, back = self.get_local_idx(idx)
        if back:
            return self.seqs_flow_back_names[seq_idx][idx].split('.')[0]
        else:
            return self.seqs_flow_names[seq_idx][idx].split('.')[0]

    def get_seq_name(self, idx) :
        return self.seq_names[self.get_local_idx(idx)[0]]

    def get_flow_dir(self, seq, backward=False):
        if backward :
            return self.dir / seq / "flow" / "backward"
        else :
            return self.dir / seq / "flow" / "forward"

    def get_event_file(self, seq):
        return h5py.File(self.dir / seq / "events" / self.event_set / "events.h5", 'r')

    def distort_flows(self, coords, flows, seq_idx) :
        res, K_rect, R, dist, K_dist = get_event_left_params(self.cams[seq_idx])
        xys_dist = rect2dist(coords, K_rect, R, dist, K_dist).astype(np.float32)
        xys_target = coords + flows
        xys_target_dist = rect2dist(xys_target, K_rect, R, dist, K_dist).astype(np.float32)
        flows_dist = xys_target_dist - xys_dist

        return xys_dist, flows_dist


    # TODO: since this is a member function we should supply only indices for easier use from outside
    def get_valid_flows(self, *args, **kwargs):
        return read_valid_flows(*args, **kwargs)


    def augment_sample(self, event_slice, xys_dist, flows):
        return augment_sample(event_slice, xys_dist, flows,
                              (640, 480), self.crop,
                              self.random_crop_offset, self.fixed_crop_offset,
                              self.random_moving, self.crop_keep_full_res, self.scale_crop,
                              self.random_flip_horizontal, self.random_flip_vertical)


    def read_events(self, seq_idx, t_begin_global_us, t_end_global_us):
        event_file = self.get_event_file(self.seq_names[seq_idx])
        t_begin_ms = (t_begin_global_us - event_file['t_offset']) / 1000.
        t_end_ms = (t_end_global_us - event_file['t_offset']) / 1000.

        event_begin_idx = event_file['ms_to_idx'][math.floor(t_begin_ms)]
        event_end_idx = event_file['ms_to_idx'][math.ceil(t_end_ms)]

        event_slice = np.concatenate(
            [event_file['events']['x'][event_begin_idx:event_end_idx].reshape(-1, 1),
             event_file['events']['y'][event_begin_idx:event_end_idx].reshape(-1, 1),
             event_file['events']['t'][event_begin_idx:event_end_idx].reshape(-1, 1) / 1000.,
             event_file['events']['p'][event_begin_idx:event_end_idx].reshape(-1, 1)], axis=1)
        if event_file['events']['p'].dtype == "|u1":
            event_slice[event_slice[:, 3] < 1, 3] = -1

        event_file.close()

        if self.event_set != 'left' :
            event_slice[:, 2] = remap_linear(event_slice[:, 2],
                                             (math.ceil(t_begin_ms) + 0.001,
                                              math.floor(t_end_ms) - 0.001),
                                             (math.floor(t_begin_ms),
                                              math.ceil(t_end_ms)))

        return event_slice, t_begin_ms, t_end_ms


    def prep_item(self, seq_idx, gt_idx, backward=False) :
        seq = self.seq_names[seq_idx]
        # res, K_rect, R, dist, K_dist = get_event_left_params(self.cams[seq_idx])

        flow_path = self.get_flow_dir(seq, backward)
        flow_path /= self.seqs_flow_names[seq_idx][gt_idx] if not backward else \
            self.seqs_flow_back_names[seq_idx][gt_idx]

        if backward :
            flow_ts = self.flow_back_ts[seq_idx]
        else :
            flow_ts = self.flow_ts[seq_idx]

        event_begin_t, event_end_t = flow_ts[gt_idx, :]

        # should only happen for backward flow
        if event_begin_t > event_end_t :
            assert(backward)
            event_begin_t, event_end_t = event_end_t, event_begin_t
        if self.dt != 0 :
            event_end_t = event_begin_t + self.dt * 1000.

        # TODO: could reorganize time management
        event_slice, event_begin_t, event_end_t = self.read_events(seq_idx, event_begin_t, event_end_t)
        event_slice[:, 2] -= event_begin_t
        dt = event_end_t - event_begin_t

        # event_slice = np.flip(event_slice, axis=0)
        flows, xys_rect = self.get_valid_flows(flow_path)
        xys_dist, flows = self.distort_flows(xys_rect, flows, seq_idx)

        event_slice, xys_dist, flows = self.augment_sample(event_slice, xys_dist, flows)

        if self.num_bins :
            if self.event_set != 'left' and self.bin_type == 'interpolation' :
                assert(len(np.unique(event_slice[:, 2])) <= self.num_bins)
            elif self.event_set == 'left' :
                if self.bin_type == 'interpolation' :
                    event_slice = bin_interp_polarity(event_slice, self.res, self.num_bins, dt)

                elif self.bin_type == 'sum' :
                    event_slice = bin_sum_polarity(event_slice, self.num_bins, dt)

        if backward :
            event_slice[:, 2] *= -1
            event_slice[:, 2] += dt
            event_slice[:, 3] *= -1
            event_slice = np.flip(event_slice, axis=0).copy()
        event_slice[:, 2] = np.around(event_slice[:, 2], 9)

        res = {'in' : {'events' : event_slice,
                       'dt' : dt, 'res' : self.res,
                       'tbins' : self.num_bins
                                 if self.num_bins and self.bin_type == 'interpolation'
                                 else 100},
               'out' : {'coords' : xys_dist, 'flows' : flows,
                        'coords_grid_idx' : get_idx_in_grid(xys_dist, self.res)},
               'frame_id' : (seq_idx, gt_idx, backward)}

        return res


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        res_dict = self.prep_item(*self.get_local_idx(idx))
        res_dict['idx'] = idx
        return res_dict

