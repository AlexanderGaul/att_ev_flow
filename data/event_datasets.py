import torch.utils.data
from torch.utils.data import Dataset

import h5py
import yaml

from pathlib import Path

from data.preprocessing.event_modules import *
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



# [] add previous frame inside or outside
# this is just a useless wrapper
class EventStream :
    def __init__(self, dir, include_endpoint=False) :
        self.dir = dir
        self.slicer = EventSlicer(h5py.File(self.dir))
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



class EventFlowSequence(Dataset) :
    def __init__(self, dir,
                 seq_idx=None,
                 batch_backward=False,
                 append_backward=False,
                 backward_hack=False,
                 include_old_events=False,
                 return_volume=False,
                 unfold_volume=False,
                 unfold_volume_spacetime=False,
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
                 stride=1,
                 t_bins=None,
                 spatial_downsample=1,
                 crop_t=None,
                 crop=None,
                 random_crop_offset=False,
                 dropout=False,
                 flow_at_events=False,  # eval_flow_at_active_coords
                 flow_at_events_area=1,
                 flow_at_coords=False,  # query_flow_at_all_input_coords
                 query_full_frame=True,
                 min_events=0,
                 max_events=np.inf,
                 dataset_format='custom',
                 undistort_events=False
                 ) :
        self.dir = Path(dir)
        self.seq_idx = seq_idx

        self.backward_hack = backward_hack
        self.batch_backward = batch_backward
        self.append_backward = append_backward
        self.include_backward = self.batch_backward or self.append_backward or self.backward_hack
        self.include_old_events = include_old_events
        self.return_volume = return_volume
        self.unfold_volume = unfold_volume
        self.unfold_volume_spacetime = unfold_volume_spacetime
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
        self.stride = stride; assert not (self.stride > 1 and not self.unfold_volume)
        self.t_bins = t_bins
        self.spatial_downsample = spatial_downsample
        self.flow_at_events = flow_at_events
        self.flow_at_events_area = flow_at_events_area
        self.flow_at_all_coords = flow_at_coords
        self.query_full_frame = query_full_frame; assert query_full_frame

        self.crop_t = crop_t

        self.crop = crop
        self.random_crop_offset = random_crop_offset
        self.dropout = dropout

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

        self.event_flow_data = BasicEventFlowSequence(self.dir / "events" / ("left" if self.dataset_format == 'dsec' else '') / "events.h5",
                                                      self.dir / "flow/forward/",
                                                      self.dir / "flow/forward_timestamps.txt",
                                                      self.dir / "flow/backward/"
                                                      if self.include_backward and not backward_hack else None,
                                                      self.dir / "flow/backward_timestamps.txt"
                                                      if self.include_backward and not backward_hack else None,
                                                      min_events, max_events,
                                                      include_old_events, include_old_events and batch_backward)

        if self.dataset_format == 'dsec' :
            self.rectify_map = h5py.File(self.dir / "events" / "left" / "rectify_map.h5")['rectify_map'][:]
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
        data = {**self.event_flow_data[idx],
                'frame_id' : self.get_item_id(idx)}
        more_events = {}
        if self.include_old_events :
            more_events['events_old'] = {
                **self.event_flow_data.get_events_all(self.event_flow_data.map_idx(idx)-1),
                'res' : data['res']}
            if self.batch_backward :
                more_events['events_new'] = {
                    **self.event_flow_data.get_events_all(self.event_flow_data.map_idx(idx)+1),
                    'res' : data['res']}



        if self.crop_t is not None :
            data = CropTimeFlowFrameEventData(self.crop_t)(data)
        if self.crop is not None :
            data = CropFlowFrameEventData(self.crop,
                                          offset='random'
                                          if self.random_crop_offset
                                          else 'center')(data)

        data['flow_frame_eval'] = data['flow_frame_valid'].copy()
        if self.include_backward :
            if self.backward_hack :
                data_back = {'flow_frame' : data['flow_frame'] * -1,
                             'flow_frame_valid' : data['flow_valid'].copy(),
                             'flow_frame_eval' : data['flow_frame_eval'].copy()}
            else :
                data_back = self.event_flow_data.get_flow_backward(idx)
                data_back['flow_frame_eval'] = data_back['flow_frame_valid'].copy()
            data_back = {**data_back, 'frame_id' : (*data['frame_id'], -1)}
            if min(data_back['ts']) != min(data['ts']) and max(data_back['ts']) != max(data['ts']) :
                more_events['events_back'] = {**self.event_flow_data.get_events_backward_all(
                    self.event_flow_data.get_best_backward_idx_for_forward(idx)), 'res' : data['res']}


        [EventData2EventArray()(d) for d in [data, *more_events.values()]]

        if self.spatial_downsample > 1 :
            [EventArraySpatialDownsample(self.spatial_downsample)(d) for d in [data, *more_events.values()]]
            data = FlowFrameSpatialDownsample(self.spatial_downsample)(data)
            if self.include_backward :
                data_back = FlowFrameSpatialDownsample(self.spatial_downsample)(data_back)

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
            if self.batch_backward:
                data_back = {**data, **data_back,
                             'event_volume' : more_events['events_back']['event_volume'] if 'events_back' in more_events else
                                              data['event_volume'].copy()}
                data_back = EventVolumeBackward()(data_back)
                if self.return_volume and self.include_old_events :
                    data['event_volume_old'] = more_events['events_old']['event_volume']
                    data_back['event_volume_old'] = EventVolumeBackward()(more_events['events_new'])['event_volume']
                data = [data, data_back]
            else :
                if self.return_volume and self.include_old_events :
                    data['event_volume_old'] = more_events['events_old']['event_volume']
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
                fn = Volume23DPatchArray(self.patch_size)
                data = [fn(d) for d in data]
                if self.event_array_remove_zeros :
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
            if self.dropout :
                data = Dropout(p=0.9)([data])[0]
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
        return len(self.event_flow_data)

    def get_item_id(self, idx) :
        if self.seq_idx is not None:
            id = (self.seq_idx, idx)
        else:
            id = (idx,)
        return id

    def collate(self, items) :
        if self.batch_backward :
            items = [i for sublist in items for i in sublist]
        if not self.compute_volume or \
                (self.flow_at_events and not self.return_volume) or \
                (self.event_array_remove_zeros and not self.return_volume) :
            if self.event_array_masked_batch :
                return collate_dict_adaptive(items, ['event_array'])
            else :
                return collate_dict_list(items)
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
            return {'xy': [0, 1],
                    't': [2],
                    'p': range(3, 3 + self.patch_size ** 3),
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
            return {'xy' : [0, 1], 't' : [2], 'p' : [3]}


class EventFlowDataset(Dataset) :
    def __init__(self, dir, seqs=True,
                 batch_backward=False,
                 append_backward=False,
                 backward_hack=False,
                 include_old_events=False,
                 return_volume=False,
                 unfold_volume=False,
                 unfold_volume_spacetime=False,
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
                 stride=1,
                 t_bins=None,
                 spatial_downsample=1,
                 crop_t=None,
                 crop=None,
                 random_crop_offset=False,
                 dropout=False,
                 flow_at_events=False,
                 flow_at_events_area=1,
                 flow_at_coords=False,
                 query_full_frame=True,
                 min_events=0, max_events=np.inf,
                 check_valid=False,
                 dataset_format='custom',
                 undistort_events=False
                 ) :
        self.dir = Path(dir)
        seq_names = sorted(os.listdir(self.dir))
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
        self.include_old_events = include_old_events
        self.return_volume = return_volume
        self.unfold_volume = unfold_volume
        self.unfold_volume_spacetime = unfold_volume_spacetime
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
        self.stride = stride
        self.t_bins = t_bins

        self.spatial_downsample = spatial_downsample

        self.crop_t = crop_t

        self.crop = crop
        self.random_crop_offset = random_crop_offset
        self.dropout = dropout
        self.flow_at_events = flow_at_events
        self.flow_at_events_area = flow_at_events_area
        self.flow_at_coords = flow_at_coords
        self.query_full_frame = query_full_frame

        self.dataset_format = dataset_format
        self.undistort_events = undistort_events

        seq_params = (self.batch_backward,
                      self.append_backward,
                      self.backward_hack,
                      self.include_old_events,
                      self.return_volume,
                      self.unfold_volume,
                      self.unfold_volume_spacetime,
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
                      self.stride,
                      self.t_bins,
                      self.spatial_downsample,
                      self.crop_t,
                      self.crop,
                      self.random_crop_offset,
                      self.dropout,
                      self.flow_at_events,
                      self.flow_at_events_area,
                      self.flow_at_coords,
                      self.query_full_frame,
                      min_events, max_events,
                      self.dataset_format,
                      self.undistort_events)

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
