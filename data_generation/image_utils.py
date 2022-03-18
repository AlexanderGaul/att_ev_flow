import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

def overlay_image(im1, im2, topleft=(0, 0)) :
    assert topleft[0] >= 0
    assert topleft[1] >= 0
    assert topleft[0] + im2.shape[0] <= im1.shape[0]
    assert topleft[1] + im2.shape[1] <= im1.shape[1]

    if im2.shape[2] < 4 :
        return im2

    if im2[:, :, [3]].max() == 0 :
        return im1

    alpha = im2[:, :, [3]] / im2[:, :, [3]].max()

    result = im1.copy()
    result[topleft[0]:(topleft[0]+im2.shape[0]),
           topleft[1]:(topleft[1]+im2.shape[1]),
           :3] = \
        im1[topleft[0]:(topleft[0]+im2.shape[0]),
           topleft[1]:(topleft[1]+im2.shape[1]),
           :3] * (1 - alpha) + \
        im2[:, :, :3] * alpha

    return result


def overlay_images(ims1, ims2, topleft=(0, 0)) :
    assert len(ims1) == len(ims2)
    ims = []
    for i in range(len(ims1)) :
        ims.append(overlay_image(ims1[i], ims2[i], topleft))
    return ims


def overlay_image_masked(f1, f2, mask) :
    assert f1.shape == f2.shape
    assert f2.shape == mask.shape

    f_out = f1.copy()
    f_out[mask] = f2[mask]

    return f_out


def overlay_images_masked(f1s, f2s, masks) :
    assert len(f1s) == len(f2s)
    assert len(f2s) == len(masks)

    f_out = []
    for i in range(len(f1s)) :
        f_out.append(overlay_image_masked(f1s[i], f2s[i], masks[i]))

    return f_out


def warp_backward(coords, flow, im_next):
    im_warped = np.zeros(im_next.shape, dtype=im_next.dtype)
    coords_next = coords + flow

    pixels_warped = []
    chunk_size = 32766
    for i in range(len(coords_next) // chunk_size + 1):
        pixels_warped.append(cv.remap(im_next,
                                      coords_next[i * chunk_size:(i + 1) * chunk_size, 0].astype(np.float32),
                                      coords_next[i * chunk_size:(i + 1) * chunk_size, 1].astype(np.float32),
                                      cv.INTER_LINEAR))

    pixels_warped = np.concatenate(pixels_warped, axis=0)

    im_warped[coords[:, 1], coords[:, 0]] = pixels_warped.reshape(len(coords), im_next.shape[-1])

    return im_warped
