import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt

import imageio
import cv2 as cv
import os

from data_generation.datasets import *
from data_generation.warp_sequences import *
from data_generation.image_utils import warp_backward
from plot import plot_flow_grid



# want to keep outline if larger alpha
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



foreground_data = MOTSForeground()


im_obj = foreground_data.get_random_item()
im_obj = im_obj.astype(np.float64) / 255.

plt.imshow(im_obj)
plt.show()
im_obj_cp = im_obj.copy()
im_obj_cp[..., -1] = 1
plt.imshow(im_obj_cp)
plt.show()

im = blur_with_alpha(im_obj, ksize_im=3, sigma_im=0)
plt.imshow(im)
plt.show()
im[..., -1] = 1
plt.imshow(im)
plt.show()

im = blur_with_alpha(im_obj, ksize_im=3, sigma_im=0,
                     ksize_alpha=3)
plt.imshow(im)
plt.show()
im[..., -1] = 1
plt.imshow(im)
plt.show()

im = blur_with_alpha(im_obj, ksize_im=0, sigma_im=0,
                     ksize_alpha=3)
plt.imshow(im)
plt.show()
im[..., -1] = 1
plt.imshow(im)
plt.show()


im = blur_with_alpha(im_obj, ksize_im=0, sigma_im=0,
                     ksize_alpha=1, ksize_outline=9)
plt.imshow(im)
plt.show()
im[..., -1] = 1
plt.imshow(im)
plt.show()