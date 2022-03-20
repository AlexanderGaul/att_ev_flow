import torch.utils.data
import imageio

from data.edata import *
from data.flow_datasets import FlowFrameSequence, FlowFrame2MaskedArrayPrep

from utils import get_grid_coordinates


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


class ImageStackPatchPrep :
    def __init__(self, patch_size=3, stride=1,
                 normalization_type='uni') :
        self.patch_size = patch_size
        self.stride = stride

        self.normalization_type = normalization_type

        self.im_mean_rgb = np.array((0.485, 0.456, 0.406))
        self.im_std_rgb = np.array((0.229, 0.224, 0.225))

    def unfold_image(self, im) :
        im_torch = torch.from_numpy(im)
        im_unfold = im_torch.unfold(
            0, self.patch_size, self.stride).unfold(
            1, self.patch_size, self.stride)
        return im_unfold.numpy()

    def __call__(self, dict_in) :
        im1 = dict_in['im1']
        im2 = dict_in['im2']
        if im1.dtype == np.int :
            im1 = (im1.astype(np.float32) - self.im_mean_rgb*255) / (self.im_std_rgb*255)
            im2 = (im2.astype(np.float32) - self.im_mean_rgb * 255) / (self.im_std_rgb * 255)
        else :
            im1 = (im1 - self.im_mean_rgb) / self.im_std_rgb
            im2 = (im2 - self.im_mean_rgb) / self.im_std_rgb

        imstack_unfold = np.concatenate([
            self.unfold_image(im1),
            self.unfold_image(im2)], axis=2)
        H_unfold, W_unfold = imstack_unfold.shape[:2]

        imstack_patchflat = imstack_unfold.reshape(H_unfold, W_unfold, -1)
        patch_array = imstack_patchflat.reshape(H_unfold*W_unfold, -1)

        coords = get_grid_coordinates((W_unfold, H_unfold), (self.patch_size / 2 - 0.5,
                                                             self.patch_size / 2 - 0.5))
        patch_array = np.concatenate([coords, patch_array], axis=1)

        dict_out = {'patch_array' : patch_array,
                    'res' : (im1.shape[1], im1.shape[0]),
                    **dict_in}
        return dict_out

class UniBackwardHack :
    def __init__(self):
        pass
    def __call__(self, dict_in) :
        dict_in['im1'], dict_in['im2'] = dict_in['im2'], dict_in['im1']
        dict_in['flow_frame'] = -dict_in['flow_frame']
        return dict_in


class ImageFlowSequence(Dataset) :
    def __init__(self, dir:Path, seq_idx=None,
                 backward_hack=False) :
        self.image_sequence = ImageSequence(dir / "ims")
        self.flow_sequence = FlowFrameSequence(dir / "flow/forward/",
                                                      dir / "flow/forward_timestamps.txt")
        self.backward_hack = backward_hack
        self.preprocessors = []
        if self.backward_hack : self.preprocessors += [UniBackwardHack()]
        self.preprocessors += [FlowFrame2MaskedArrayPrep(),
                              ImageStackPatchPrep()]
        self.seq_idx = seq_idx

    def get_unprocessed(self, idx) :
        return {**self.flow_sequence[idx],
                **self.image_sequence[idx]}
    def preprocess(self, data) :
        for prep in self.preprocessors :
            data = prep(data)
        return data

    def __len__(self) :
        return len(self.flow_sequence)

    def __getitem__(self, idx) :
        data = self.get_unprocessed(idx)
        data = self.preprocess(data)
        frame_id = (self.seq_idx, idx) if self.seq_idx is not None else idx
        item = {**data,
                'frame_id' : frame_id}
        return item

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