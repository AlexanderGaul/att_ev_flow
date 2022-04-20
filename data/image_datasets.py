import torch.utils.data
import imageio

from data.event_datasets import *
from data.flow_datasets import FlowFrameSequence, FlowFrame2MaskedArray
from data.image_functional import ImagePair2PatchArray


class ImageSequence :
    def __init__(self, dir:Path) :
        self.dir = Path(dir)
        self.im_names = sorted(os.listdir(dir))

    def __len__(self) :
        return len(self.im_names) - 1

    def __getitem__(self, idx) :
        im1 = np.asarray(imageio.imread(self.dir / self.im_names[idx]))
        im2 = np.asarray(imageio.imread(self.dir / self.im_names[idx+1]))
        return {'im1' : im1, 'im2' : im2}


class ImageSwap :
    def __init__(self) : pass
    def __call__(self, data) :
        data['im1'], data['im2'] = data['im2'], data['im1']
        return data

class UniBackwardHack :
    def __init__(self):
        pass
    def __call__(self, dict_in) :
        dict_in['im1'], dict_in['im2'] = dict_in['im2'], dict_in['im1']
        dict_in['flow_frame'] = -dict_in['flow_frame']
        return dict_in


class ImageFlowSequence(CombinedProcessedSequence) :
    def __init__(self, dir:Path, seq_idx=None,
                 backward_hack=False) :
        self.image_sequence = ImageSequence(dir / "ims")
        self.flow_sequence = FlowFrameSequence(dir / "flow/forward/",
                                               dir / "flow/forward_timestamps.txt")
        self.backward_hack = backward_hack
        preps = []
        if self.backward_hack : preps += [UniBackwardHack()]
        preps += [FlowFrame2MaskedArray(),
                  ImagePair2PatchArray()]
        self.seq_idx = seq_idx
        super().__init__([self.flow_sequence, self.image_sequence],
                         preps, self.seq_idx)

    def collate(self, items) :
        item_batched = {}
        to_stack = ['flow_array', 'coords', 'coords_mask', 'patch_array']
        for key in to_stack :
            item_batched[key] = torch.stack([torch.tensor(item[key],
                                                          dtype=torch.float32 if item[key].dtype=='float64'
                                                          else None)
                                             for item in items])
        for key in items[0] :
            if key not in to_stack :
                item_batched[key] = [item[key] for item in items]
        return item_batched


class ImageFlowDataset(Dataset) :
    def __init__(self, dir:Path, seqs=None,
                 return_backward_hack=False) :
        self.dir = Path(dir)
        self.seq_names = sorted(os.listdir(self.dir))
        if seqs is not None and hasattr(seqs, '__len__') :
            self.seq_names = [self.seq_names[i] for i in seqs]
        elif seqs is not None:
            if seqs > 0 :
                self.seq_names = self.seq_names[:seqs]
            elif seqs < 0 :
                self.seq_names = self.seq_names[seqs:]
        self.return_backward_hack = return_backward_hack
        self.seqs = [ImageFlowSequence(self.dir / seq_name, i)
                     for i, seq_name in enumerate(self.seq_names)]
        self.cat_dataset = torch.utils.data.ConcatDataset(self.seqs)
        if self.return_backward_hack :
            self.seqs_back = [ImageFlowSequence(self.dir / seq_name, i, True)
                              for i, seq_name in enumerate(self.seq_names)]
            self.cat_dataset_back = torch.utils.data.ConcatDataset(self.seqs_back)

    def __len__(self) :
        return len(self.cat_dataset)

    def __getitem__(self, idx) :
        item = self.cat_dataset[idx]
        if self.return_backward_hack :
            item_back = self.cat_dataset_back[idx]
            item_back['frame_id'] = (*item_back['frame_id'], -1)
            item = [item, item_back]
        return item

    def collate(self, items) :
        if self.return_backward_hack :
            items = [i for sublist in items for i in sublist]
        return self.seqs[0].collate(items)