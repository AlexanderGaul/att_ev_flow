import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path



def test_load_image_with_mask() :
    im_path = "/storage/user/gaul/gaul/thesis/data/MOTS/training/image_02/0001/000000.png"
    inst_path = "/storage/user/gaul/gaul/thesis/data/MOTS/instances/0001/000000.png"

    inst = np.array(Image.open(inst_path))
    obj_ids = np.unique(inst)
    print(obj_ids)
    print(inst.shape)
    # to correctly interpret the id of a single object
    obj_id = obj_ids[0]
    class_id = obj_id // 1000
    obj_instance_id = obj_id % 1000

    mask = (inst == obj_ids[2])

    im = np.array(Image.open(im_path))

    return im, mask



def load_mots_object_masks(seq_idx, im_idx) :
    im_suffix = Path(("0" * (4 - len(str(seq_idx))) + str(seq_idx)) + "/" +
                     ("0" * (6 - len(str(im_idx)))) + str(im_idx) + ".png")

    im_path = Path("/storage/user/gaul/gaul/thesis/data/MOTS/training/image_02")
    inst_path = Path("/storage/user/gaul/gaul/thesis/data/MOTS/instances")

    im = np.array(Image.open(im_path / im_suffix))
    inst_im = np.array(Image.open(inst_path / im_suffix))

    masks = []

    obj_ids = np.unique(inst_im)

    for id in obj_ids[:] :
        if id // 1000 != 1 and id // 1000 != 2 :
            continue
        masks.append((inst_im == id))

    return im, masks


def load_mots_object_ims(seq_idx, im_idx, min_size=0) :
    im_full, masks = load_mots_object_masks(seq_idx, im_idx)

    assert im_full.dtype == np.uint8

    ims = []
    for mask in masks :
        idxs_y, idxs_x = np.where(mask)
        y_min, y_max = idxs_y.min(), idxs_y.max()
        x_min, x_max = idxs_x.min(), idxs_x.max()

        im_obj = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 4),
                          dtype=im_full.dtype)
        if im_obj.shape[0] < min_size or im_obj.shape[1] < min_size :
            continue

        mask_cut = mask[y_min:y_max+1, x_min:x_max+1]
        im_obj[mask_cut, :3] = im_full[mask]
        im_obj[mask_cut, 3] = 255

        ims.append(im_obj)

    return ims