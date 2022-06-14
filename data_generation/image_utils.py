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
                                      cv.INTER_CUBIC))

    pixels_warped = np.concatenate(pixels_warped, axis=0)

    im_warped[coords[:, 1], coords[:, 0]] = pixels_warped.reshape(len(coords), im_next.shape[-1])

    return im_warped


def blur_with_alpha(im, ksize_im, sigma_im,
                    ksize_alpha=0, sigma_alpha=0,
                    ksize_outline=None, sigma_outline=0) :
    if ksize_im == 0 and ksize_alpha == 0 and ksize_outline == 0 :
        return im
    if ksize_alpha > 1 and ksize_outline==None :
        ksize_outline = ksize_alpha
    elif ksize_outline == None :
        ksize_outline = 1

    dtype = im.dtype
    cvtype, max_value = {np.dtype('float64') : (cv.CV_64F, 1.),
                         np.dtype('float32') : (cv.CV_32F, 1.),
                         np.dtype('uint8') : (cv.CV_8U, 255)}[dtype]

    # TODO: if blur increase res
    max_blur = max(ksize_im, ksize_alpha, ksize_outline)
    pad = max(0, (max_blur - 1) // 2)
    pad_blur = 0 if ksize_alpha <=1 and ksize_outline <= 1 else max(0, (ksize_alpha-1)//2, (ksize_outline-1)//2)
    im_blur = np.zeros((im.shape[0] + 2 * pad,
                        im.shape[1] + 2 * pad,
                        im.shape[2]), dtype=dtype)
    im_blur[pad:im_blur.shape[0]-pad, pad:im_blur.shape[1]-pad, :] = im

    mask_tight = (im_blur[..., -1] > 0)
    if ksize_alpha > 1 :
        kernel_alpha = cv.getGaussianKernel(ksize_alpha, sigma_alpha)
        im_blur[..., -1] = cv.sepFilter2D(im_blur[..., -1], cvtype, kernel_alpha, kernel_alpha,
                                          borderType=cv.BORDER_CONSTANT)
    mask = (im_blur[..., -1] > 0)


    if ksize_im > 1 :
        kernel_im = cv.getGaussianKernel(ksize_im, sigma_im)
        blur = cv.sepFilter2D(im_blur[..., :-1], cv.CV_64F, kernel_im, kernel_im,
                              borderType=cv.BORDER_CONSTANT)
        weight = cv.sepFilter2D(mask_tight.astype(np.float64), cv.CV_64F, kernel_im, kernel_im,
                                borderType=cv.BORDER_CONSTANT)
        weight[weight == 0] = 1

    if ksize_outline is not None and ksize_outline > 1 :
        kernel_out= cv.getGaussianKernel(ksize_outline, sigma_outline)
        blur_out = cv.sepFilter2D(im_blur[..., :-1], cv.CV_64F, kernel_out, kernel_out,
                              borderType=cv.BORDER_CONSTANT)
        weight_out = cv.sepFilter2D(mask_tight.astype(np.float64), cv.CV_64F, kernel_out, kernel_out,
                                borderType=cv.BORDER_CONSTANT)
        weight_out[weight_out == 0] = 1

    if ksize_im > 1 :
        im_blur[mask, :-1] = (blur / weight[..., None])[mask].astype(dtype)
    if ksize_outline is not None and ksize_outline > 1 :
        im_blur[~mask_tight, :-1] = (blur_out / weight_out[..., None])[~mask_tight].astype(dtype)

    if ksize_outline <= ksize_alpha :
        im_blur[im_blur[..., -1]  == 0, :-1] = 0

    return im_blur[pad-pad_blur:im_blur.shape[0]-pad+pad_blur, pad-pad_blur:im_blur.shape[1]+pad_blur-pad, :]
