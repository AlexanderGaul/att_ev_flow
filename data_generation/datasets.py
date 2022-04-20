import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

import os
from pathlib import Path



class MOTSForegroundSequence :
    def __init__(self, im_dir:Path, inst_dir:Path, min_size=0) :
        self.im_dir = im_dir
        self.inst_dir = inst_dir
        dir_content = sorted(os.listdir(self.im_dir))
        self.im_names = [f for f in dir_content if f.split('.')[-1] == "png"]

        self.min_size = min_size

    def __len__(self) :
        return len(self.im_names)

    def get_num_objects(self, idx) :
        inst_im = np.array(Image.open(self.inst_dir / self.im_names[idx]))
        num_valid = 0
        obj_ids = np.unique(inst_im)
        for id in obj_ids[:]:
            if id // 1000 != 1 and id // 1000 != 2:
                continue
            num_valid += 1
        return num_valid

    def get_objects(self, idx) :
        im_full = np.array(Image.open(self.im_dir / self.im_names[idx]))
        inst_im = np.array(Image.open(self.inst_dir / self.im_names[idx]))
        masks = []

        obj_ids = np.unique(inst_im)

        for id in obj_ids[:]:
            if id // 1000 != 1 and id // 1000 != 2:
                continue
            masks.append((inst_im == id))

        ims = []
        for mask in masks:
            idxs_y, idxs_x = np.where(mask)
            y_min, y_max = idxs_y.min(), idxs_y.max()
            x_min, x_max = idxs_x.min(), idxs_x.max()

            im_obj = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 4),
                              dtype=im_full.dtype)
            if im_obj.shape[0] < self.min_size or im_obj.shape[1] < self.min_size:
                continue

            mask_cut = mask[y_min:y_max + 1, x_min:x_max + 1]
            im_obj[mask_cut, :3] = im_full[mask]
            im_obj[mask_cut, 3] = 255

            ims.append(im_obj)

        return ims

    def compute_num_objects(self) :
        num = 0
        for i in len(self):
            num += len(self.get_objects(i))
        return num



class MOTSForeground :
    def __init__(self, dir=Path("/storage/user/gaul/gaul/thesis/data/MOTS/")) :
        self.dir = dir
        self.inst_dir = self.dir / "instances"
        self.im_dir = self.dir / "training/image_02/"

        self.seq_names = sorted(os.listdir(self.im_dir))
        self.seqs = []
        self.seq_lens = []
        for name in self.seq_names :
            seq = MOTSForegroundSequence(self.im_dir / name,
                                                    self.inst_dir / name)
            self.seqs.append(seq)
            self.seq_lens.append(len(seq))

        self.seq_lens_acc = list(np.cumsum(self.seq_lens))

    def __len__(self) :
        return np.sum(self.seq_lens)

    def get_objects(self, idx) :
        seq_idx = np.searchsorted(self.seq_lens_acc, idx, side='right')
        offset = 0 if seq_idx == 0 else self.seq_lens_acc[seq_idx-1]
        return self.seqs[seq_idx].get_objects(idx - offset)

    def get_random_item(self) :
        objs = []
        while len(objs) == 0 :
            idx = np.random.randint(0, len(self))
            objs = self.get_objects(idx)

        idx = np.random.randint(0, len(objs))
        return objs[idx]


class DSECImageSequence(torch.utils.data.Dataset) :
    def __init__(self, dir:Path) :
        self.dir = dir
        self.im_names = sorted(os.listdir(self.dir))

    def __len__(self) :
        return len(self.im_names)

    def __getitem__(self, idx) :
        im = np.asarray(Image.open(self.dir / self.im_names[idx]))
        return im


class DSECBackground :
    def __init__(self, dir:Path=Path("/storage/user/gaul/gaul/thesis/data/DSEC_images/")) :
        self.dir = dir
        dir_content = os.listdir(self.dir)
        self.seq_names = [f for f in dir_content if os.path.isdir(self.dir / f)]
        self.sequences = [DSECImageSequence(self.dir / sn) for sn in self.seq_names]
        self.dataset = torch.utils.data.ConcatDataset(self.sequences)

    def __len__(self) :
        return len(self.dataset)

    def __getitem__(self, idx) :
        return self.dataset[idx]

    def get_random_item(self) :
        idx = np.random.randint(0, len(self))
        return self[idx]

