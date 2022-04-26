import numpy as np
import torch

from events import interp_volume, interp_volume_jit_xyfloat
from plot import create_event_frame_picture
from utils import get_grid_coordinates


def event_array_2_volume(event_array, res, dt, num_bins, normalize, xyfloat=False) :
    if xyfloat :
        return interp_volume_jit_xyfloat(event_array, res, num_bins, 0, dt)
    else :
        return interp_volume(event_array, res, num_bins, 0, dt, normalize)


def event_volume_flatten_space(event_volume) :
    T, H, W = event_volume.shape
    event_array = event_volume.reshape(T, H * W).T
    coords = get_grid_coordinates((W, H))
    event_array = np.concatenate([coords, event_array], axis=1)
    return event_array


def event_volume_flatten_spacetime(event_volume, dt) :
    T, H, W = event_volume.shape
    events = event_volume.reshape(-1)
    coords = np.tile(get_grid_coordinates((W, H)), (T, 1))
    ts = np.linspace(0, dt, T).repeat(H * W)

    event_array = np.concatenate([coords,
                                  ts.reshape(-1, 1),
                                  events.reshape(-1, 1)],
                                 axis=1)
    return event_array





def event_volume_2_image_pair(event_volume) :
    n_bins = event_volume.shape[0]
    ev_1 = event_volume[:(n_bins // 2)]
    ev_2 = event_volume[-(n_bins // 2):]
    return create_event_frame_picture(ev_1), create_event_frame_picture(ev_2)


def  event_volume_backward(event_volume) :
    return np.flip(event_volume * -1, axis=0).copy()


def unfold(volume, patch_size, stride) :
    im_torch = torch.from_numpy(volume)
    im_unfold = im_torch.unfold(
        0, patch_size, stride).unfold(
        1, patch_size, stride)
    return im_unfold.numpy()


def unfold_3d(volume, patch_size, stride) :
    im_torch = torch.from_numpy(volume)
    im_unfold = im_torch.unfold(
        0, patch_size, stride).unfold(
        1, patch_size, stride).unfold(
        2, patch_size, stride)
    return im_unfold.numpy()


def volume_2_patch_array(volume, patch_size, stride, format='torch') :
    if format == 'torch' :
        volume = volume.transpose(1, 2, 0)
    H, W, T = volume.shape
    volume_unfold = unfold(volume, patch_size, stride)
    H_unfold, W_unfold = volume_unfold.shape[:2]
    patch_array = volume_unfold.reshape(H_unfold, W_unfold, -1)
    patch_array = patch_array.reshape(H_unfold * W_unfold, -1)

    offset = patch_size / 2 - 0.5
    coords = get_grid_coordinates((W-2*offset, H-2*offset),
                                  (offset, offset),
                                  (stride, stride))
    assert len(patch_array) == len(coords)
    return coords, patch_array


def volume_2_3d_patch_array(volume, patch_size, stride, dt, format='torch') :
    if format == 'torch' :
        volume = volume.transpose(1, 2, 0)
    if stride > 1 : raise NotImplementedError()
    volume_unfold = unfold_3d(volume, patch_size, stride)
    H_unfold, W_unfold, T_unfold = volume_unfold.shape[:3]
    patch_array = volume_unfold.reshape(H_unfold, W_unfold, T_unfold, -1)
    patch_array = patch_array.reshape(H_unfold * W_unfold, T_unfold, -1)
    patch_array = patch_array.transpose(1, 0, 2)
    patch_array = patch_array.reshape(H_unfold * W_unfold * T_unfold, -1)

    coords = get_grid_coordinates((W_unfold, H_unfold), (patch_size / 2 - 0.5,
                                                         patch_size / 2 - 0.5))
    coords = np.tile(coords, (T_unfold, 1))
    ts = np.linspace(0, dt, T_unfold+2)[1:-1].repeat(H_unfold * W_unfold)

    return coords, ts, patch_array


def event_volume_2_patch_array_flatten_time(volume, patch_size, stride, dt, format='torch') :
    if format == 'torch' :
        volume = volume.transpose(1, 2, 0)
    if stride > 1 : raise NotImplementedError()
    H, W, T = volume.shape
    volume_unfold = unfold(volume, patch_size, stride)
    H_unfold, W_unfold = volume_unfold.shape[:2]
    patch_array = volume_unfold.reshape(H_unfold, W_unfold, T, -1)
    patch_array = patch_array.transpose(2, 0, 1, 3)
    patch_array = patch_array.reshape(T * H_unfold * W_unfold, -1)

    coords = get_grid_coordinates((W_unfold, H_unfold), (patch_size / 2 - 0.5,
                                                         patch_size / 2 - 0.5))
    coords = np.tile(coords, (T, 1))
    ts = np.linspace(0, dt, T).repeat(H_unfold * W_unfold)

    patch_array = np.concatenate([coords,
                                  ts.reshape(-1, 1),
                                  patch_array], axis=1)
    return patch_array


