import torch.utils.data
from torch.utils.data import Dataset

import h5py
import hdf5plugin
#import tables
import yaml

from pathlib import Path

from data.preprocessing.event_modules import *
from data.preprocessing.augmentation_functional import *
from data.volume_modules import *
from data.flow_modules import *
from data.image_functional import ImagePair2PatchArray
from data.data_functional import *
from data.flow_datasets import *
from events import interp_volume

from data.utils import *

from event_slicer import EventSlicer

from dsec import get_event_left_params
from utils import dist2rect

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# [] add previous frame inside or outside
# this is just a useless wrapper
class EventStream :
    def __init__(self, dir, include_endpoint=False) :
        self.dir = dir
        print(self.dir)
        self.slicer = EventSlicer(h5py.File(self.dir, locking=False))
        self.include_endpoint = include_endpoint

    # TODO which time unit
    # Better not to put this into a __getitem__
    def get_events(self, t_begin, t_end) :
        events = self.slicer.get_events(int(t_begin), int(t_end), self.include_endpoint)
        if events is None :
            events = {'x' : np.array([], dtype=np.uint16),
                      'y' : np.array([], dtype=np.uint16),
                      't' : np.array([], dtype=np.uint32),
                      'p' : np.array([], dtype=np.uint8)}
        events['p'] = events['p'].astype(np.int8)
        events['p'][events['p'] == 0] = -1
        #events['t'] -= t_begin

        # should we construct array already here? maybe not
        # but would have to rewrite functions to handle this stupid dict?
        return events

class EventFrames :
    def __init__(self, dir) :
        self.dir = Path(dir)
        self.f_names = sorted(os.listdir(self.dir))
        self.f_names = list(filter(lambda f : Path.is_file(self.dir / f),
                                   self.f_names))

    def __len__(self) :
        return len(self.f_names)

    def __getitem__(self, idx) :
        events_zip = np.load(self.dir / self.f_names[idx])
        events = {'x' : events_zip['x'],
                  'y' : events_zip['y'],
                  't' : events_zip['t'],
                  'p' : events_zip['p']}
        return events


# TODO: events_exist_only_at_flow_frames
# TODO: what if we require previous frame: if events_exist_only_at_flow_frames: check previous timestamp
# TODO:

class BasicEventFlowFrameSequence(Dataset) :
    def __init__(self, event_dir, flow_dir, ts_dir,
                 flow_back_dir=None, flow_back_ts=None) :
        self.event_sequence = EventFrames(event_dir)
        self.flow_sequence = FlowFrameSequence(flow_dir,
                                               ts_dir,
                                               ts_to_us=1e9)
        if flow_back_dir is not None and flow_back_ts is not None:
            self.flow_back_sequence = FlowFrameSequence(flow_back_dir,
                                                        flow_back_ts,
                                                        ts_to_us=1e9)
        else:
            self.flow_back_sequence = None

    def __len__(self) : return len(self.flow_sequence)

    def __getitem__(self, idx) :
        flow_data = self.flow_sequence[idx]
        events = self.event_sequence[idx]
        return {'event_data': events, **flow_data}


class BasicEventFlowSequenceV2(Dataset) :
    def __init__(self, event_dir, flow_dir, ts_dir,
                 flow_dir_bwd, ts_dir_bwd,
                 min_events=0, max_events=np.inf,
                 events_exist_only_at_ts=False,
                 require_prev=False,
                 skip_flow_freq=1
                 ) :
        self.event_stream = EventStream(event_dir)
        if flow_dir is not None :
            self.flow_sequence = FlowFrameSequence(flow_dir,
                                                   ts_dir,
                                                   skip_flow_freq=skip_flow_freq)
        else :
            self.flow_sequence = EvalFlowSequence(ts_dir)
        if flow_dir_bwd is not None and ts_dir_bwd is not None :
            self.flow_sequence_bwd = FlowFrameSequence(flow_dir_bwd,
                                                       ts_dir_bwd)
        else :
            self.flow_sequence_bwd = None

        if events_exist_only_at_ts and require_prev :
            if skip_flow_freq != 1 : raise NotImplementedError("")
            # TODO: implement for other dts move into FlowFrameSequence
            idx_valid_fwd = self.flow_sequence.ts[1:, 0] == self.flow_sequence.ts[:-1, 1]
            idx_valid_fwd = np.concatenate([[False], idx_valid_fwd])
            self.idx_map = np.where(idx_valid_fwd)[0]
            if self.flow_sequence_bwd is not None :
                idx_valid_bwd = np.min(self.flow_sequence_bwd.ts[1:, :], axis=1) == \
                                np.max(self.flow_sequence_bwd.ts[:-1, :], axis=1)
                idx_valid_bwd = np.concatenate([idx_valid_bwd, [False]])
                self.idx_map_bwd = np.where(idx_valid_bwd)[0]
                assert len(self.idx_map) == len(self.idx_map_bwd)
        else :
            self.idx_map = None
            self.idx_map_bwd = None

        if min_events != 0 and max_events < np.inf :
            raise NotImplementedError()

    def map_idx(self, idx, backward=False) :
        if backward :
            return self.map_idx_bwd(idx)
        if self.idx_map is None : return idx
        else : return self.idx_map[idx]
    def map_idx_bwd(self, idx) :
        if self.idx_map_bwd is None : return idx
        else : return self.idx_map_bwd[idx]

    def get_ts(self, idx, include_prev=False, backward=False) :
        if backward :
            return self.get_ts_bwd(include_prev)
        if include_prev :
            return self.get_ts_prev(idx, combined=True)
        return tuple(self.flow_sequence.get_ts(self.map_idx(idx)))
    def get_ts_bwd(self, idx, include_prev=False) :
        if include_prev :
            return self.get_ts_prev_bwd(idx, combined=True)
        return tuple(self.flow_sequence_bwd.get_ts(self.map_idx_bwd(idx)))

    def get_flow(self, idx, backward=False) :
        if backward :
            return self.get_flow_bwd(idx)
        idx = self.map_idx(idx)
        return self.flow_sequence[idx]
    def get_flow_bwd(self, idx) :
        idx = self.map_idx_bwd(idx)
        return self.flow_sequence_bwd[idx]

    def get_events(self, idx, backward=False) :
        if backward :
            return self.get_events_bwd(idx)
        return self.event_stream.get_events(*self.get_ts(idx))
    def get_events_bwd(self, idx) :
        return self.event_stream.get_events(*reversed(self.get_ts_bwd(idx)))
    def get_events_ts(self, ts):
        return self.event_stream.get_events(*ts)
    def get_event_data_ts(self, ts) :
        return {'event_data' : self.event_stream.get_events(*ts),
                'dt' : (max(*ts) - min(*ts)) / 1000,
                'ts' : ts}

    def idx_fwd2bwd_ts0(self, idx) :
        ts_fwd = self.get_ts(idx)
        if self.get_ts_bwd(idx)[0] == ts_fwd[0] :
            return idx
        elif idx > 0 and self.get_ts_bwd(idx-1)[0] == ts_fwd[0] :
            return idx - 1
        else :
            return (idx - 1) % len(self.idx_map_bwd)

    def idx_fwd2bwd_tsrange(self, idx) :
        ts_fwd = self.get_ts(idx)
        if (np.array(list(reversed(self.get_ts_bwd(idx)))) == ts_fwd).all() :
            return idx
        if idx < len(self) - 1 and (np.array(list(reversed(self.get_ts_bwd(idx+1)))) == ts_fwd).all() :
            return idx + 1
        else :
            return (idx + 1) % len(self)

    def get_ts_prev(self, idx, combined=False, backward=False) :
        if backward :
            return self.get_ts_prev_bwd(idx, combined)
        idx_m = self.map_idx(idx)
        ts = self.get_ts(idx)
        if idx_m > 0 and self.flow_sequence.get_ts(idx_m-1)[1] == ts[0] :
            ts_prev = tuple(self.flow_sequence.get_ts(idx_m-1))
        elif self.flow_sequence_bwd is not None and self.flow_sequence_bwd.get_ts(idx_m)[0] == ts[0] :
            ts_prev = tuple(reversed(self.flow_sequence_bwd.get_ts(idx_m)))
        else :
            dt = ts[1] - ts[0]
            ts_prev = (ts[0] - dt, ts[1] - dt)
        if combined :
            ts_prev = (ts_prev[0], ts[1])
        return tuple(ts_prev)

    def get_ts_prev_bwd(self, idx, combined=False) :
        idx_m = self.map_idx_bwd(idx)
        ts = self.get_ts_bwd(idx)
        if idx_m < len(self) - 1 and self.flow_sequence_bwd.get_ts(idx_m+1)[1] == ts[0] :
            ts_prev = tuple(self.flow_sequence_bwd.get_ts(idx_m + 1))
        elif self.flow_sequence.get_ts(idx_m)[0] == ts[0] :
            ts_prev = tuple(reversed(self.flow_sequence.get_ts(idx_m)))
        else :
            dt = ts[0] - ts[1]
            ts_prev = (ts[0] + dt, ts[1] + dt)
        if combined :
            ts_prev = (ts_prev[0], ts[1])
        return tuple(ts_prev)

    def __len__(self) :
        if self.idx_map is not None :
            return len(self.idx_map)
        else :
            return len(self.flow_sequence)

    def __getitem__(self, idx, cat_prev=False, backward=False) :
        flow_data = self.get_flow(idx, backward)
        ts = flow_data['ts']
        if cat_prev :
            ts = self.get_ts_prev(idx, True, backward)
        event_data = self.get_event_data_ts(tuple(reversed(ts)) if backward else ts)
        return {**flow_data,
                **event_data,
                'ts' : ts}




class BasicEventFlowSequence(Dataset) :
    def __init__(self, event_dir, flow_dir, ts_dir,
                 flow_back_dir=None, flow_back_ts=None,
                 min_events=0, max_events=np.inf,
                 skip_first=False, skip_last=False) :
        self.event_stream = EventStream(event_dir)
        self.flow_sequence = FlowFrameSequence(flow_dir,
                                               ts_dir)
        if flow_back_dir is not None and flow_back_ts is not None :
            self.flow_back_sequence = FlowFrameSequence(flow_back_dir,
                                                        flow_back_ts)
        else :
            self.flow_back_sequence = None

        if min_events > 0 or max_events < np.inf :
            event_counts = np.loadtxt(Path(os.path.dirname(event_dir)) / "flow_frames_event_counts.txt")
            assert len(event_counts) == len(self.flow_sequence)
            if skip_first :
                event_counts = event_counts[1:]
            if skip_last :
                event_counts = event_counts[:-1]
            self.idx_map = np.argwhere((event_counts > min_events) & (event_counts < max_events)).reshape(-1)
        elif skip_first or skip_last :
            self.idx_map = np.arange(0 if not skip_first else 1,
                                     len(self.flow_sequence) if not skip_last else len(self.flow_sequence))
        else :
            self.idx_map = None

    def map_idx(self, idx) :
        if self.idx_map is None : return idx
        else : return self.idx_map[idx]

    def get_events(self, idx) :
        return self.get_events_all(self.map_idx(idx))

    def get_flow_forward(self, idx) :
        return self.get_flow_forward_all(self.map_idx(idx))

    def get_flow_backward(self, idx) :
        return self.get_flow_backward_all(self.map_idx(idx))

    def get_best_backward_idx_for_forward(self, idx) :
        assert self.idx_map == None
        assert self.flow_back_sequence != None
        ts_f = self.flow_sequence.ts
        ts_b = self.flow_back_sequence.ts
        tf = ts_f[idx]
        if ts_b[idx][0] == tf[1] and ts_b[idx][1] == tf[0] :
            return idx
        elif idx < len(ts_b) - 1 and ts_b[idx+1][0] == tf[1] and ts_b[idx+1][1] == tf[0] :
            return idx+1
        else :
            return (idx+1) % len(ts_b)

    def __len__(self) :
        if self.idx_map is None : return len(self.flow_sequence)
        else : return len(self.idx_map)

    def __getitem__(self, idx) :
        return self.get_item_all(self.map_idx(idx))

    def get_events_backward_all(self, idx) :
        ts = self.flow_back_sequence.get_ts(idx)
        return {'event_data': self.event_stream.get_events(*reversed(ts)),
                'ts': ts,
                'dt': (ts[0] - ts[1]) / 1000}

    def get_events_all(self, idx) :
        ts = self.flow_sequence.get_ts(idx)
        return {'event_data' : self.event_stream.get_events(*ts),
                'ts' : ts,
                'dt' : (ts[1] - ts[0]) / 1000}

    def get_flow_forward_all(self, idx) :
        return self.flow_sequence[idx]

    def get_flow_backward_all(self, idx) :
        if self.flow_back_sequence is not None :
            idx = self.get_best_backward_idx_for_forward(idx)
            return self.flow_back_sequence[idx]
        else :
            return None

    def get_item_all(self, idx) :
        flow_data = self.get_flow_forward_all(idx)
        events = self.event_stream.get_events(*flow_data['ts'])
        return {'event_data' : events, **flow_data}

    def len_all(self): return len(self.flow_sequence)


    # TODO
    # [] do we need forward backward here
    # [] how to handle dsec
    def get_events_before(self, idx) :
        pass

    def get_events_after(self, idx) :
        pass



class EventFlowSequence(Dataset) :

    def __init__(self, dir,
                 seq_idx=None,
                 batch_backward=False,
                 append_backward=False,
                 backward_hack=False,
                 match_fwd_bwd_ts0=False,
                 include_prev_events=False,
                 cat_prev_events=False,
                 return_volume=False,
                 unfold_volume=False,
                 unfold_volume_spacetime=False,
                 unfold_volume_spacetime_stack_t_dim = False,
                 unfold_volume_spacetime_cat_ts = True,
                 unfold_volume_time_flat=False,
                 flat_volume=False,
                 volume_flatten_spacetime=False,
                 event_array_remove_zeros=False,
                 event_array_masked_batch=False,
                 normalize_volume=True,
                 unit_scale_volume=False,
                 volume_polarity_onehot=False,
                 event_images=False,
                 event_patch_context=False,
                 event_patch_context_mode=None,
                 event_patch_context_relative=True,
                 patch_size=None,
                 patch_size_t=None,
                 stride=1,
                 stride_t=1,
                 t_bins=None,
                 random_flip_horizontal=False,
                 random_flip_vertical=False,
                 spatial_downsample=1,
                 crop_t=None,
                 crop=None,
                 random_crop_offset=False,
                 dropout=False,
                 noise=False,
                 flow_at_events=False,  # eval_flow_at_active_coords
                 flow_at_events_area=1,
                 flow_at_coords=False,  # query_flow_at_all_input_coords
                 query_full_frame=True,
                 min_events=0,
                 max_events=np.inf,
                 dataset_format='custom',
                 undistort_events=False,
                 is_test_set=False
                 ) :
        self.dir = Path(dir)
        self.seq_idx = seq_idx

        self.backward_hack = backward_hack
        self.batch_backward = batch_backward
        self.append_backward = append_backward
        if self.append_backward : raise NotImplementedError()
        self.include_backward = self.batch_backward or self.append_backward or self.backward_hack
        self.match_fwd_bwd_ts0 = match_fwd_bwd_ts0
        self.include_prev_events = include_prev_events
        self.cat_prev_events = cat_prev_events
        self.return_volume = return_volume
        self.unfold_volume = unfold_volume
        self.unfold_volume_spacetime = unfold_volume_spacetime
        self.unfold_volume_spacetime_stack_t_dim = unfold_volume_spacetime_stack_t_dim
        self.unfold_volume_spacetime_cat_ts = unfold_volume_spacetime_cat_ts
        self.unfold_volume_time_flat = unfold_volume_time_flat
        self.flat_volume = flat_volume
        self.volume_flatten_spacetime = volume_flatten_spacetime
        self.event_array_remove_zeros = event_array_remove_zeros; assert not (self.event_array_remove_zeros and self.return_volume)
        self.event_array_masked_batch = event_array_masked_batch; assert not (self.event_array_masked_batch and self.return_volume)
        self.normalize_volume = normalize_volume
        self.unit_scale_volume = unit_scale_volume
        self.volume_polarity_onehot = volume_polarity_onehot
        self.event_images = event_images
        self.event_patch_context = event_patch_context
        self.event_patch_context_mode = event_patch_context_mode
        self.event_patch_context_relative = event_patch_context_relative
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.stride = stride; assert not (self.stride > 1 and not self.unfold_volume)
        self.stride_t = stride_t
        self.t_bins = t_bins

        self.random_flip_horizontal = random_flip_horizontal
        self.random_flip_vertical = random_flip_vertical

        self.spatial_downsample = spatial_downsample
        self.flow_at_events = flow_at_events
        self.flow_at_events_area = flow_at_events_area
        self.flow_at_all_coords = flow_at_coords
        self.query_full_frame = query_full_frame; assert query_full_frame

        self.crop_t = crop_t

        self.crop = crop
        self.random_crop_offset = random_crop_offset
        self.dropout = dropout
        self.noise = noise

        self.dataset_format = dataset_format
        self.undistort_events = undistort_events

        self.compute_volume = (return_volume or
                               unfold_volume or
                               unfold_volume_spacetime or
                               unfold_volume_time_flat or
                               flat_volume or
                               volume_flatten_spacetime or
                               event_images)
        self.return_batch = (self.batch_backward)

        if self.dataset_format != "carla" :
            self.event_flow_data = BasicEventFlowSequenceV2(
                self.dir / "events" / ("left" if self.dataset_format == 'dsec' else '') / "events.h5",
                self.dir / "flow/forward/" if not is_test_set else None,
                self.dir / "flow/forward_timestamps.txt" if not is_test_set else self.dir / ".." / (os.path.basename(self.dir) + ".csv"),
                self.dir / "flow/backward/" if not is_test_set else None,
                self.dir / "flow/backward_timestamps.txt" if not is_test_set else None,
                events_exist_only_at_ts=self.dataset_format != 'dsec',
                require_prev=include_prev_events or cat_prev_events
            )
        else :
            self.event_flow_data = BasicEventFlowSequenceV2(
                self.dir / "evs" / "events.h5",
                self.dir / "flow_100" / "forward",
                self.dir / "flow_100" / "fwd_tss_us.txt",
                self.dir / "flow_100" / "backward",
                self.dir / "flow_100" / "bwd_tss_us.txt",
                require_prev=include_prev_events or cat_prev_events,
                skip_flow_freq=100
            )
            pass
            """self.event_flow_data = BasicEventFlowSequence(self.dir / "events" / ("left" if self.dataset_format == 'dsec' else '') / "events.h5",
                                                          self.dir / "flow/forward/",
                                                          self.dir / "flow/forward_timestamps.txt",
                                                          self.dir / "flow/backward/"
                                                          if self.include_backward and not backward_hack else None,
                                                          self.dir / "flow/backward_timestamps.txt"
                                                          if self.include_backward and not backward_hack else None,
                                                          min_events, max_events,
                                                          include_old_events, include_old_events and batch_backward)"""
        """
        else :

            self.event_flow_data = BasicEventFlowFrameSequence(self.dir / "events",
                                                               self.dir / "flow/backward/",
                                                               self.dir / "timestamps.txt",
                                                               self.dir / "flow/forward/",
                                                               self.dir / "timestamps.txt")
            """

        if self.dataset_format == 'dsec' :
            self.rectify_map = h5py.File(self.dir / "events" / "left" / "rectify_map.h5", locking=False)['rectify_map'][:]
            if not is_test_set :
                self.cam = yaml.safe_load(open(self.dir / "calibration" / "cam_to_cam.yaml"))
                res, K_rect, R, dist, K_dist = get_event_left_params(self.cam)
                self.cam_intrinsics = (K_dist, dist, R, K_rect)


    @staticmethod
    def is_valid_dir(dir) :
        dir = Path(dir)
        if not os.path.isfile(dir / "events/events.h5") : return False
        if not os.path.isfile(dir / "flow/forward_timestamps.txt") : return False
        if not os.path.isdir(dir / "flow/forward/") : return False
        l = len(np.loadtxt(str(dir / "flow/forward_timestamps.txt")))
        if not (l ==
                len(os.listdir(dir / "flow/forward/"))) : return False
        if os.path.isfile(dir / "events/flow_frames_event_counts.txt") :
            if not (l == len(np.loadtxt(str(dir / "events/flow_frames_event_counts.txt")))) : return False
        if os.path.isdir(dir / "flow/backward/") or os.path.isfile(dir / "flow/backward_timestamps.txt"):
            if not (os.path.isdir(dir / "flow/backward/") and
                    os.path.isfile(dir / "flow/backward_timestamps.txt")) : return False
            if not (l ==
                    len(np.loadtxt(str(dir / "flow/backward_timestamps.txt")))): return False
            if not (l ==
                    len(os.listdir(dir / "flow/forward/"))): return False
        return True

    """
    def write_event_prerpocessing_to_folder(self) :
        if self.return_volume :
            volumes = []
            for i in range(len(self)) :
                data = self[i]
                volumes.append(data['event_volume'])
            volumes = np.stack(volumes)
            np.save(self.dir / "events" / "volumes", volumes)

        else :
            raise NotImplementedError("")
    """


    def __getitem__(self, idx) :
        assert idx < len(self)

        get_backward = False
        if self.append_backward and idx > len(self.event_flow_data) :
            idx = idx % len(self.event_flow_data)
            get_backward = True
        if self.batch_backward :
            get_backward = True

        flip_horizontal = self.random_flip_horizontal and np.random.binomial(1, 0.5)
        flip_vertical = self.random_flip_vertical and np.random.binomial(1, 0.5)

        # TODO: what to do if we do not want to batch backward??

        if not get_backward or self.batch_backward :
            data = {**self.event_flow_data.__getitem__(idx, self.cat_prev_events),
                    'frame_id' : self.get_item_id(idx)}
        more_events = {}


        """
        if self.crop_t is not None :
            data = CropTimeFlowFrameEventData(self.crop_t)(data)
        if self.crop is not None :
            data = CropFlowFrameEventData(self.crop,
                                          offset='random'
                                          if self.random_crop_offset
                                          else 'center')(data)
        """

        if get_backward :
            if self.match_fwd_bwd_ts0 :
                idx_back = self.event_flow_data.idx_fwd2bwd_ts0(idx)
            else :
                idx_back = self.event_flow_data.idx_fwd2bwd_tsrange(idx)
            ts_back = self.event_flow_data.get_ts_bwd(idx_back, self.cat_prev_events)
            assert len(ts_back) == 2
            data_back = self.event_flow_data.get_flow_bwd(
                idx_back)

            if flip_vertical : data_back = FlowFrameFlipVertical()(data_back)
            if flip_horizontal : data_back = FlowFrameFlipHorizontal()(data_back)

            data_back['flow_frame_eval'] = data_back['flow_frame_valid'].copy()
            data_back = {**data_back, 'frame_id' : (*data['frame_id'], -1)}

            if (min(ts_back) != min(data['ts']) and max(ts_back) != max(data['ts'])) or \
               self.cat_prev_events :
                more_events['events_back'] = {**self.event_flow_data.get_event_data_ts(np.flip(ts_back)),
                                              'ts' : ts_back,
                                              'res' : data['res']}
            if not self.batch_backward :
                # TODO move this into data
                # TODO: finish this
                data = {**data_back}
                pass

        if flip_vertical : data = FlowFrameFlipVertical()(data)
        if flip_horizontal : data = FlowFrameFlipHorizontal()(data)

        if self.dataset_format == 'carla' :
            data['flow_frame_valid'][-20:, ] = 0

        data['flow_frame_eval'] = data['flow_frame_valid'].copy()

        if self.include_prev_events :
            ts_prev = self.event_flow_data.get_ts_prev(idx)
            event_data_prev = self.event_flow_data.get_event_data_ts(ts_prev)
            more_events['events_prev'] = {
                **event_data_prev,
                'res' : data['res']}
            if self.batch_backward :
                # Are these the previous events for backward?
                ts_prev_bwd =  self.event_flow_data.get_ts_prev_bwd(idx_back)
                more_events['events_prev_bwd'] = {
                    **self.event_flow_data.get_event_data_ts(tuple(reversed(ts_prev_bwd))),
                    'ts' : ts_prev_bwd,
                    'res' : data['res']}

        if self.crop is not None :
            if self.random_crop_offset :
                crop_offset = (np.random.randint(0, data['res'][1] - self.crop[0]),
                               np.random.randint(0, data['res'][0] - self.crop[1]))
            else : # crop to center
                crop_offset = ((data['res'][1] - self.crop[0]) // 2,
                               (data['res'][0] - self.crop[1]) // 2)
            if not self.compute_volume :
                raise NotImplementedError("")

        [EventData2EventArray()(d) for d in [data, *more_events.values()]]

        if flip_vertical : [EventArrayFlipVertical()(d) for d in [data, *more_events.values()]]
        if flip_horizontal : [EventArrayFlipHorizontal()(d) for d in [data, *more_events.values()]]
        # TODO: generate random offset for crop
        if self.spatial_downsample > 1 :
            [EventArraySpatialDownsample(self.spatial_downsample)(d) for d in [data, *more_events.values()]]
            data = FlowFrameSpatialDownsample(self.spatial_downsample)(data)
            if self.include_backward :
                data_back = FlowFrameSpatialDownsample(self.spatial_downsample)(data_back)

        if len(data['event_array']) == 0 :
            print("Zero events in: " + str(self.dir) + " at " + str(idx))

        if self.dropout:
            [Dropout(p=self.dropout)([d]) for d in [data, *more_events.values()]]
        if self.noise:
            [Noise(self.noise, self.compute_volume)([d]) for d in [data, *more_events.values()]]


        if self.undistort_events :
            if self.spatial_downsample == 1 :
                [EventArrayUndistortMap(self.rectify_map)(d) for d in [data, *more_events.values()]]
            else :
                fn = EventArrayUndistortIntrinsics(self.cam_intrinsics, self.spatial_downsample)
                [fn(d) for d in [data, *more_events.values()]]

        if self.flow_at_events :
            e_mask = EventMask(self.flow_at_events_area)(data)
            data['flow_frame_eval'] = data['flow_frame_eval'] & e_mask
            if self.batch_backward :
                data_back['flow_frame_eval'] = data_back['flow_frame_eval'] & e_mask



        if self.compute_volume :
            [EventArray2Volume(self.t_bins,
                               self.normalize_volume,
                               self.unit_scale_volume,
                               self.volume_polarity_onehot,
                               xyfloat=self.undistort_events)(d)
             for d in [data, *more_events.values()]]
            if self.crop is not None :
                [EventVolumeCrop(crop_offset, self.crop)(d) for d in [data, *more_events.values()]]
                data = FlowFrameCrop(crop_offset, self.crop)(data)
                if self.batch_backward :
                    data_back = FlowFrameCrop(crop_offset, self.crop)(data_back)
            if self.batch_backward:
                data_back = {**data, **data_back,
                             'event_volume' : more_events['events_back']['event_volume'] if 'events_back' in more_events else
                                              data['event_volume'].copy()}
                data_back = EventVolumeBackward()(data_back)
                if self.return_volume and self.include_prev_events :
                    data['event_volume_prev'] = more_events['events_prev']['event_volume']
                    data_back['event_volume_prev'] = EventVolumeBackward()(more_events['events_prev_bwd'])['event_volume']
                data = [data, data_back]
            else :
                if self.return_volume and self.include_prev_events :
                    data['event_volume_prev'] = more_events['events_prev']['event_volume']
                data = [data]

            if self.unfold_volume :
                data = [Volume2PatchArray(patch_size=self.patch_size,
                                          stride=self.stride,
                                          input_name='event_volume',
                                          output_name='event_array')(d) for d in data]
                if self.event_array_remove_zeros :
                    data = [ArrayRemoveZeros(raw_dims=range(2, 2+self.patch_size**2*self.t_bins))(d)
                            for d in data]
            elif self.unfold_volume_spacetime :
                fn = Volume23DPatchArray(self.patch_size, patch_size_3=self.patch_size_t,
                                         stride_3=self.stride_t,
                                         stack_t_dim=self.unfold_volume_spacetime_stack_t_dim,
                                         cat_ts=self.unfold_volume_spacetime_cat_ts)
                data = [fn(d) for d in data]
                if self.event_array_remove_zeros :
                    if self.patch_size_t is not None : raise NotImplementedError("")
                    data = [ArrayRemoveZeros(raw_dims=range(3, 3+self.patch_size**3))(d)
                            for d in data]
            elif self.unfold_volume_time_flat :
                data = [EventVolume2PatchArrayFlatTime()(d) for d in data]
                if self.event_array_remove_zeros: raise NotImplementedError("")
            elif self.flat_volume :
                data = [EventVolumeFlat()(d) for d in data]
                if self.event_array_remove_zeros: raise NotImplementedError("")
            elif self.volume_flatten_spacetime :
                data = [EventVolume2Array()(d) for d in data]
                if self.event_array_remove_zeros: raise NotImplementedError("")
            elif self.event_images :
                data = [EventVolume2ImagePair()(d) for d in data]
                data = [ImagePair2PatchArray()(d) for d in data]

        else :
            if self.batch_backward :
                data_back = {**data, **data_back,
                             'event_array' :
                                more_events['events_back']['event_array'] if 'events_back' in more_events else
                                data['event_array'].copy()}
                data_back = EventArrayBackward()(data_back)
                if self.batch_backward :
                    data = [data, data_back]
                else :
                    data = data_back
                    print("not implemented")
            else :
                data = [data]

            if self.event_patch_context :
                add_context_fn = EventArrayContext(patch_size=self.patch_size,
                                                   num_events=self.t_bins,
                                                   mode=self.event_patch_context_mode)
                data = [add_context_fn(d) for d in data]

        if self.flow_at_all_coords :
            data = [FlowFrame2FlowArrayAtAllCoordinates('event_array' if 'event_array' in d
                                         else 'patch_array')(d) for d in data]
        elif not self.return_volume :
            if self.query_full_frame :
                data = [FlowFrame2MaskedArray()(d) for d in data]
            else :
                data = [FlowFrame2Array()(d) for d in data]
        return data if self.return_batch else data[0]

    def __len__(self) :
        if not self.append_backward :
            return len(self.event_flow_data)
        else :
            return 2 * len(self.event_flow_data)

    def get_item_id(self, idx) :
        if self.seq_idx is not None:
            id = (self.seq_idx, idx)
        else:
            id = (idx,)
        return id

    def collate(self, items) :
        if self.batch_backward :
            items = [i for sublist in items for i in sublist]
        # TODO: need to update this
        # TODO: hoe does flow at events work again???
        if not self.compute_volume or \
                (self.flow_at_events and not self.return_volume) or \
                (self.event_array_remove_zeros and not self.return_volume) :
            if self.event_array_masked_batch :
                return collate_dict_adaptive(items, ['event_array'])
            else :
                return collate_dict_adaptive(items)
        else :
            return collate_dict_adaptive(items)

    def t_bins_total(self) :
        if self.t_bins is None :
            return None
        elif type(self.t_bins) is list :
            return np.sum(self.t_bins)
        else :
            return self.t_bins

    def data_format(self) :
        if self.flat_volume:
            return {'xy': [0, 1],
                    't': [],
                    'p': range(2, 2 + self.t_bins_total()),
                    'raw': []}
        elif self.event_images:
            return {'xy': [0, 1],
                    't': [],
                    'p': range(2, 56),
                    'raw': []}
        elif self.unfold_volume:
            return {'xy': [0, 1],
                    't': [],
                    'p': range(2, 2 + self.t_bins_total() * self.patch_size ** 2),
                    'raw': []}
        elif self.unfold_volume_spacetime:
            p_begin = 3 if self.unfold_volume_spacetime_cat_ts else 2
            return {'xy': [0, 1],
                    't': [2] if self.unfold_volume_spacetime_cat_ts else [],
                    'p': range(p_begin,
                               p_begin + self.patch_size ** 3 if self.patch_size_t is None else
                               p_begin + self.patch_size**2 * self.patch_size_t),
                    'raw': []}
        elif self.unfold_volume_time_flat:
            return {'xy': [0, 1],
                    't': [2],
                    'p': range(3, 3 + self.patch_size ** 2),
                    'raw': []}
        elif self.event_patch_context:
            if self.event_patch_context_mode == 'volume':
                num_polarities = self.patch_size ** 2 * self.t_bins_total()
                num_other = self.patch_size ** 2 * self.t_bins_total()
                return {'xy': [0, 1],
                        't': [2],
                        'p': range(3, 4 + num_polarities),
                        'raw': range(4 + num_polarities, 4 + num_polarities + num_other)}
            elif self.event_patch_context_mode == 'time_surface':
                return {'xy': [0, 1],
                        't': [2],
                        'p': [3],
                        'raw': range(4, 4 + self.patch_size ** 2 * 2)}
        elif self.return_volume :
            return {'t_bins' : self.t_bins_total()}
        else :
            return {'xy' : [0, 1], 't' : [2], 'p' : [3], 'raw' : []}


class EventFlowDataset(Dataset) :
    def __init__(self, dir, seqs=True,
                 batch_backward=False,
                 append_backward=False,
                 backward_hack=False,
                 match_fwd_bwd_ts0=False,
                 include_prev_events=False,
                 cat_prev_events=False,
                 return_volume=False,
                 unfold_volume=False,
                 unfold_volume_spacetime=False,
                 unfold_volume_spacetime_stack_t_dim=False,
                 unfold_volume_spacetime_cat_ts=True,
                 unfold_volume_time_flat=False,
                 flat_volume=False,
                 volume_flatten_spacetime=False,
                 event_array_remove_zeros=False,
                 event_array_masked_batch=False,
                 normalize_volume=True,
                 unit_scale_volume=False,
                 volume_polarity_onehot=False,
                 event_images=False,
                 event_patch_context=False,
                 event_patch_context_mode=None,
                 event_patch_context_relative=True,
                 patch_size=None,
                 patch_size_t=None,
                 stride=1,
                 stride_t=1,
                 t_bins=None,
                 random_flip_horizontal=False,
                 random_flip_vertical=False,
                 spatial_downsample=1,
                 crop_t=None,
                 crop=None,
                 random_crop_offset=False,
                 dropout=False,
                 noise=False,
                 flow_at_events=False,
                 flow_at_events_area=1,
                 flow_at_coords=False,
                 query_full_frame=True,
                 min_events=0, max_events=np.inf,
                 check_valid=False,
                 dataset_format='custom',
                 undistort_events=False,
                 is_test_set=False
                 ) :
        self.dir = Path(dir)
        if dataset_format == 'carla' :
            seq_names = get_folder_depth(self.dir, 2)
        else :
            seq_names = sorted(os.listdir(self.dir))
        if is_test_set :
            assert dataset_format == 'dsec'
            seq_names = [sn.split('.')[0] for sn in seq_names if os.path.isfile(self.dir / sn) and sn.split('.')[1] == 'csv']
        seq_names = select(seq_names, seqs)
        if check_valid :
            self.seq_names = []
            for n in seq_names :
                if not os.path.isdir(self.dir / n) or not EventFlowSequence.is_valid_dir(self.dir / n) :
                    print(n)
                else : self.seq_names.append(n)
        else :
            self.seq_names = seq_names

        self.batch_backward = batch_backward
        self.append_backward = append_backward
        self.backward_hack = backward_hack
        self.match_fwd_bwd_ts0 = match_fwd_bwd_ts0
        self.include_prev_events = include_prev_events
        self.cat_prev_events = cat_prev_events
        self.return_volume = return_volume
        self.unfold_volume = unfold_volume
        self.unfold_volume_spacetime = unfold_volume_spacetime
        self.unfold_volume_spacetime_stack_t_dim = unfold_volume_spacetime_stack_t_dim
        self.unfold_volume_spacetime_cat_ts = unfold_volume_spacetime_cat_ts
        self.unfold_volume_time_flat = unfold_volume_time_flat
        self.flat_volume = flat_volume
        self.volume_flatten_spacetime = volume_flatten_spacetime
        self.event_array_remove_zeros = event_array_remove_zeros
        self.event_array_masked_batch = event_array_masked_batch
        self.normalize_volume = normalize_volume
        self.unit_scale_volume = unit_scale_volume
        self.volume_polarity_onehot = volume_polarity_onehot
        self.event_images = event_images
        self.event_patch_context = event_patch_context
        self.event_patch_context_mode = event_patch_context_mode
        self.event_patch_context_relative = event_patch_context_relative
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.stride = stride
        self.stride_t = stride_t
        self.t_bins = t_bins

        self.random_flip_horizontal = random_flip_horizontal
        self.random_flip_vertical = random_flip_vertical

        self.spatial_downsample = spatial_downsample

        self.crop_t = crop_t

        self.crop = crop
        self.random_crop_offset = random_crop_offset
        self.dropout = dropout
        self.noise = noise
        self.flow_at_events = flow_at_events
        self.flow_at_events_area = flow_at_events_area
        self.flow_at_coords = flow_at_coords
        self.query_full_frame = query_full_frame

        self.dataset_format = dataset_format
        self.undistort_events = undistort_events

        seq_params = (self.batch_backward,
                      self.append_backward,
                      self.backward_hack,
                      self.match_fwd_bwd_ts0,
                      self.include_prev_events,
                      self.cat_prev_events,
                      self.return_volume,
                      self.unfold_volume,
                      self.unfold_volume_spacetime,
                      self.unfold_volume_spacetime_stack_t_dim,
                      self.unfold_volume_spacetime_cat_ts,
                      self.unfold_volume_time_flat,
                      self.flat_volume,
                      self.volume_flatten_spacetime,
                      self.event_array_remove_zeros,
                      self.event_array_masked_batch,
                      self.normalize_volume,
                      self.unit_scale_volume,
                      self.volume_polarity_onehot,
                      self.event_images,
                      self.event_patch_context,
                      self.event_patch_context_mode,
                      self.event_patch_context_relative,
                      self.patch_size,
                      self.patch_size_t,
                      self.stride,
                      self.stride_t,
                      self.t_bins,
                      self.random_flip_horizontal,
                      self.random_flip_vertical,
                      self.spatial_downsample,
                      self.crop_t,
                      self.crop,
                      self.random_crop_offset,
                      self.dropout,
                      self.noise,
                      self.flow_at_events,
                      self.flow_at_events_area,
                      self.flow_at_coords,
                      self.query_full_frame,
                      min_events, max_events,
                      self.dataset_format,
                      self.undistort_events,
                      is_test_set)

        self.seqs = [EventFlowSequence(self.dir / seq_name, i, *seq_params)
                     for i, seq_name in enumerate(self.seq_names)]
        self.cat_dataset = torch.utils.data.ConcatDataset(self.seqs)

    def __getitem__(self, idx) : return self.cat_dataset[idx]
    def __len__(self) : return len(self.cat_dataset)

    def collate(self, items) :
        return self.seqs[0].collate(items)

    def data_format(self) :
        return self.seqs[0].data_format()

    def t_bins_total(self) :
        return self.seqs[0].t_bins_total()


# TODO: could make child of some  super class to keep attributes
class LimitedEventsDataset(Dataset) :
    def __init__(self, full_dataset, max_events):
        self.full_dataset = full_dataset
        idx_map = -np.ones(len(full_dataset), dtype=np.float64)
        idx = 0
        for i, data in enumerate(full_dataset) :
            if type(data) is list :
                for d in data :
                    if d['event_array'].shape[-2] > max_events :
                        continue
            else :
                if data['event_array'].shape[-2] > max_events :
                    continue
            idx_map[idx] = i
            idx += 1
        self.idx_map = idx_map[:idx]

        if hasattr(full_dataset, 'collate') :
            self.__dict__['collate'] = full_dataset.collate

    def __len__(self) : return len(self.idx_map)

    def __getitem__(self, idx) :
        return self.full_dataset[self.idx_map[idx]]
