import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

import os
from pathlib import Path
import cv2 as cv

import math

from data_generation.utils import get_connected_components


def cut_image(mask, im_full) :
    idxs_y, idxs_x = np.where(mask)
    y_min, y_max = idxs_y.min(), idxs_y.max()
    x_min, x_max = idxs_x.min(), idxs_x.max()

    im_obj = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 4),
                      dtype=im_full.dtype)

    mask_cut = mask[y_min:y_max + 1, x_min:x_max + 1]
    im_obj[mask_cut, :3] = im_full[mask]
    im_obj[mask_cut, 3] = 255
    return im_obj

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
        for mask_full in masks:
            comps, comp_ids = get_connected_components(mask_full)
            for comp_id in comp_ids :
                mask = comps == comp_id
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

    def random_item(self):
        return self.get_random_item(), (-1, )


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

class CityscapesCity :
    def __init__(self, dir_rgb, dir_seg) :
        self.dir_rgb = dir_rgb
        self.dir_seg = dir_seg

        self.rgb_suffix = "_leftImg8bit.png"
        self.label_suffix = "_gtFine_labelIds.png"
        self.inst_suffix = "_gtFine_instanceIds.png"
        self.names = [f[:-len(self.rgb_suffix)] for f in sorted(os.listdir(self.dir_rgb))]

        self.pick_from = range(4, 34)
        self.smooth_boundary = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 21, 22, 23]
        self.has_instance = range(24, 34)
        self.boost_probability = [17, 18, 19, 20, 26, 27, 28]

        """
        labels = [
                #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
                Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
                Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
                Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
                Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
                Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
                Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
                Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
                Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
                Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
                Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
                Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
                Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
                Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
                Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
                Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
                Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
                Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
                Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
                Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
                Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
                Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
                Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
                Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
                Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
                Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
                Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
                Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
                Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
                Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
                Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
                Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]       """

    def random_item(self) :
        idx_rng = np.random.randint(0, len(self.names))

        rgb_im, label_im, inst_im = self.load_image(idx_rng)

        labels = np.unique(label_im)
        labels = [l for l in labels if l in self.pick_from]

        label_rng = labels[np.random.randint(0, len(labels))]
        if label_rng not in self.boost_probability :
            label_rng = labels[np.random.randint(0, len(labels))]
        # how to increase probability for certain categories

        comp_im, comp_ids = self.get_components_for_label(label_im, inst_im, label_rng)
        if len(comp_ids) == 0 :
            import matplotlib.pyplot as plt
            plt.imshow(label_im == label_rng)
            plt.show()
            pass
        comp_rng = comp_ids[np.random.randint(0, len(comp_ids))]

        mask = comp_im == comp_rng

        mask_area = mask.sum()
        if math.sqrt(mask_area) > 60 :
            if math.sqrt(mask_area) > 200 :
                kernel_size = int(math.sqrt(mask_area) / 5)
            else :
                kernel_size = int(math.sqrt(mask_area) / 10)
            kernel_size = kernel_size + (kernel_size + 1) % 2
            kernel_size_erode = kernel_size
            flip = False
            if label_rng == 7 or label_rng == 23:
                flip = True
                kernel_size_erode += 5
                mask = self.dilate_erode(mask, kernel_size, kernel_size_erode, flip_order=flip)
            #mask = self.dilate_erode(mask, kernel_size, kernel_size_erode, flip_order=flip)

            if flip :
                comp_shape, comp_ids_shape = get_connected_components(mask)
                max_id = comp_ids_shape
                max_area = 0
                for csid in comp_ids_shape :
                    area = (comp_shape == csid).sum()
                    if area >= max_area :
                        max_area = area
                        max_id = csid
                mask = comp_shape == max_id
        """
        import matplotlib.pyplot as plt;
        plt.imshow(cut_image(mask, rgb_im));
        plt.show()
        """
        im_obj = cut_image(mask, rgb_im)


        return im_obj, (idx_rng, label_rng, comp_rng)


    def dilate_erode(self, mask, kernel_size, kernel_size_erode=None, flip_order=False) :
        if kernel_size_erode is None :
            kernel_size_erode = kernel_size
        mask_dil_er = mask.astype(np.uint8)
        kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                (kernel_size_erode,
                                                 kernel_size_erode))
        kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                (kernel_size,
                                                 kernel_size))
        if not flip_order :
            mask_dil_er = cv.dilate(mask_dil_er, kernel_dilate)
            mask_dil_er = cv.erode(mask_dil_er, kernel_erode)
        else :
            mask_dil_er = cv.erode(mask_dil_er, kernel_erode)
            mask_dil_er = cv.dilate(mask_dil_er, kernel_dilate)
        return mask_dil_er.astype(mask.dtype)



    def load_image(self, idx) :
        name = self.names[idx]
        rgb = np.array(Image.open(self.dir_rgb / (name + self.rgb_suffix)))
        label = np.array(Image.open(self.dir_seg / (name + self.label_suffix)))
        inst = np.array(Image.open(self.dir_seg / (name + self.inst_suffix)))
        return rgb, label, inst


    def get_instances_for_label(self, inst_im, label_id) :
        instances = (inst_im + 1) % (label_id * 1000)
        instances[inst_im // 1000 != label_id] = 0
        assert (instances > 0).any()
        return instances


    def get_components_for_label(self, label_im, inst_im, label) :
        if label not in self.has_instance :
            return get_connected_components(label_im == label)
        else :
            comp_count = 0
            comps = np.zeros(label_im.shape, label_im.dtype)
            comp_ids = np.zeros(0)
            label_inst = self.get_instances_for_label(inst_im, label)
            inst = np.unique(label_inst)
            inst = inst[inst > 0]
            assert len(inst) > 0
            for i in inst :
                mask = (label_inst == i) & (label_im == label)
                comps_inst, comp_ids_inst = get_connected_components(mask)
                comp_ids_inst += comp_count
                comps[mask] = comps_inst[mask] + comp_count
                comp_ids = np.concatenate([comp_ids, comp_ids_inst])
                comp_count += len(comp_ids_inst)
            return comps, comp_ids

    def get_component_mask(self, label_im, inst_im) :
        component_mask = np.zeros(label_im.shape, dtype=np.uint64)
        labels = np.unique(label_im)
        for label in labels :
            label_comp_count = 0
            if label not in self.pick_from :
                continue
            if label in self.has_instance :
                instances = self.get_instances_for_label(inst_im, label)
                instances = np.unique(instances)
                instances = instances[instances > 0]
                for i in instances :
                    mask = (inst_im == label * 1000 + i) & (label_im == label)
                    comps, comp_ids = get_connected_components(mask)
                    for comp_id in comp_ids :
                        component_mask[comps == comp_id] = label * 1000 + label_comp_count
                        label_comp_count += 1
            else :
                mask = label_im == label
                comps, comp_ids = get_connected_components(mask)
                for comp_id in comp_ids:
                    component_mask[comps == comp_id] = label * 1000 + label_comp_count
                    label_comp_count += 1
        return component_mask


    def get_object(self, label_im, inst, label_id, inst_idx=None, comp_idx=None) :
        if label_id in self.has_instance :
            inst_found = self.get_instances_for_label(inst, label_id)
            instances = sorted(np.unique(inst_found))[1:]
            num_instances = len(instances)
            if inst_idx is None :
                inst_idx = np.random.randint(0, num_instances)
            inst_n = instances[inst_idx]
            mask = (inst_found == inst_n)
        else :
            mask = label_im == label_id
            inst_idx = 0
        comps = measure.label(mask)
        comp_ids = np.unique(comps)
        background = np.unique(comps[~mask])
        comp_ids = [id for id in comp_ids if id not in background]

        num_comps = len(comp_ids)
        if comp_idx is None :
            comp_idx = np.random.randint(0, num_comps)
        comp = comp_ids[comp_idx]

        mask = comps == comp

        return mask, inst_idx, comp_idx

class Cityscapes :
    def __init__(self, dir=Path("/storage/user/gaul/gaul/thesis/data/Cityscapes"),
                 subset="train") :
        self.dir_rgb = dir / "leftImg8bit_trainvaltest/leftImg8bit" / subset
        self.dir_seg = dir / "gtFine_trainvaltest/gtFine" / subset

        self.citiy_names = sorted(os.listdir(self.dir_rgb))

        self.cities = [CityscapesCity(self.dir_rgb / city,
                                      self.dir_seg / city) for city in self.citiy_names]

    def random_item(self) :
        city_idx_rng = np.random.randint(0, len(self.cities))
        im, (im_id, label_id, comp_id) = self.cities[city_idx_rng].random_item()
        return im, (city_idx_rng, im_id, label_id, comp_id)


class CatRandomDataset :
    def __init__(self, datasets, probs) :
        self.datasets = datasets
        self.probs = probs / probs.sum()

    def random_item(self) :
        idx = int(np.argwhere(np.random.multinomial(1, self.probs))[0])
        return self.datasets[idx].random_item()


class PNGSequence(Dataset) :
    def __init__(self, dir) :
        self.dir = Path(dir)
        self.im_names = sorted(os.listdir(self.dir))
        self.im_names = [im for im in self.im_names if ".png" in im]

    def __len__(self) : return len(self.im_names)

    def __getitem__(self, idx) :
        return np.array(Image.open(self.dir / self.im_names[idx]))

class PNGDataset :
    def __init__(self, dir):
        self.dir = Path(dir)
        self.seq_names = sorted(os.listdir(self.dir))
        self.seqs = [PNGSequence(self.dir / sq_name) for sq_name in self.seq_names]
        self.dataset = torch.utils.data.ConcatDataset(self.seqs)

    def random_item(self) :
        idx_rng = np.random.randint(0, len(self.dataset))
        return self.dataset[idx_rng], idx_rng



