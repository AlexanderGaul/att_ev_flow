import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import imageio
from PIL import Image

import matplotlib.pyplot as plt

import scipy
import cv2 as cv

import numpy as np
import json
import time
from data_generation.datasets import *


np.random.seed(0)

data = Cityscapes()
city = data.cities[0]


for i in range(100) :
    rgb, label, inst = city.load_image(i)
    """
    plt.imshow(rgb)
    plt.show()
    """
    """
    plt.imshow(label)
    plt.show()
    """
    """
    plt.imshow(inst)
    plt.show()
    
    inst_26 = city.get_instances_for_label(inst, 26)
    plt.imshow(inst_26)
    plt.show()
    """
    mask, _, _ = city.get_object(label, inst, 23, inst_idx=1, comp_idx=0)
    rgb_sky = rgb.copy()
    rgb_sky[~mask] = 0
    plt.imshow(rgb_sky)
    plt.show()
    print(rgb_sky[mask].mean())
"""
plt.imshow(mask.astype(float))
plt.show()
"""
for i in range(50) :
    im, id = city.random_item()
    #plt.imshow(im)
    #plt.show()


comps, ids = city.get_components_for_label(label, inst, 26)
plt.imshow(comps)
plt.show()

t = time.time()
mask = city.get_component_mask(label, inst)
print(time.time() - t)

plt.imshow(mask // 1000 == 11)
plt.show()

for i in [11000] :
    building = (mask == i).astype(np.uint8)
    building_dil = cv.dilate(building, np.ones((7, 7), dtype=np.uint8))
    building_dil_er = cv.erode(building_dil, np.ones((7, 7), dtype=np.uint8))
    buildingfill = scipy.ndimage.morphology.binary_fill_holes(building_dil_er)

    cut_im = cut_image(building > 0, rgb)
    plt.imshow(cut_im)
    plt.show()

    cut_im = cut_image(buildingfill, rgb)
    plt.imshow(cut_im)
    plt.show()


plt.imshow(mask)
plt.show()
mask_ids = np.unique(mask)
plt.imshow(mask == 0)
plt.show()
for mask_id in mask_ids :
    plt.imshow(mask == mask_id)
    plt.show()

mask_renumerate = np.zeros(mask.shape, mask.dtype)
for i, id in enumerate(mask_ids) :
    mask_renumerate[mask == id] = i
plt.imshow(mask_renumerate)
plt.show()

dir_seg = Path("/storage/user/gaul/gaul/thesis/data/Cityscapes/gtFine_trainvaltest/gtFine/train/aachen/")
dir_rgb = Path("/storage/user/gaul/gaul/thesis/data/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/")

ims_rgb = sorted(os.listdir(dir_rgb))

content = sorted(os.listdir(dir_seg))
instance_id_files = [f for f in content if "_instanceIds.png" in f]
label_id_files = [f for f in content if "_labelIds.png" in f]
json_files = [f for f in content if ".json"  in f]

for i in range(1) :
    im = imageio.imread(dir_seg / content[1], format='PNG-FI')
    im = np.array(Image.open(dir_seg / instance_id_files[i]))

    plt.imshow(im)
    plt.show()

    im = np.array(Image.open(dir_seg / label_id_files[i]))
    plt.imshow(im)
    plt.show()

    im_label = im
    im_building = (im_label == 11).astype(np.uint8)
    plt.imshow(im_building)
    plt.show()
    im_building_blur = cv.medianBlur(im_building, 31)
    plt.imshow(im_building_blur)
    plt.show()

    jfile = json.load(open(dir_seg / json_files[i]))

    im = np.array(Image.open(dir_rgb / ims_rgb[i]))
    plt.imshow(im)
    for o in range(len(np.array(jfile['objects']))) :
        p = np.array(jfile['objects'][o]['polygon'])

        plt.plot(p[:, 0], p[:, 1], linewidth=1)
    plt.show()

print("EOF")
