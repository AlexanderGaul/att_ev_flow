import numpy as np
import torch
import torch.nn.functional as F

from events import interp_volume, interp_volume_jit_xyfloat
from plot import create_event_frame_picture
from utils import get_grid_coordinates

from ERAFT.model.utils import coords_grid


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


def unfold(volume, patch_size, stride, padding=0) :
    im_torch = torch.from_numpy(volume)
    im_torch = F.pad(im_torch, (0, 0, padding, padding, padding, padding))
    im_unfold = im_torch.unfold(
        0, patch_size, stride).unfold(
        1, patch_size, stride)
    return im_unfold.numpy()


# TODO: introduce patch_size per dimension
def unfold_3d(volume, patch_size, stride, padding=0,
              patch_size_3=None, stride_3=None, padding_3=0) :
    if patch_size_3 is None :
        patch_size_3 = patch_size
    if stride_3 is None :
        stride_3 = stride
    im_torch = torch.from_numpy(volume)
    im_torch = F.pad(im_torch, (padding_3, padding_3, padding, padding, padding, padding))
    im_unfold = im_torch.unfold(
        0, patch_size, stride).unfold(
        1, patch_size, stride).unfold(
        2, patch_size_3, stride_3)
    return im_unfold.numpy()


def volume_2_patch_array(volume, patch_size, stride, padding, format='torch') :
    if format == 'torch' :
        volume = volume.transpose(1, 2, 0)
    H, W, T = volume.shape
    volume_unfold = unfold(volume, patch_size, stride, padding)
    H_unfold, W_unfold = volume_unfold.shape[:2]
    patch_array = volume_unfold.reshape(H_unfold, W_unfold, -1)
    patch_array = patch_array.reshape(H_unfold * W_unfold, -1)

    offset = patch_size / 2 - 0.5 - padding
    coords = get_grid_coordinates((W-2*offset, H-2*offset),
                                  (offset, offset),
                                  (stride, stride))
    assert len(patch_array) == len(coords)
    return coords, patch_array


def volume_2_3d_patch_array(volume, patch_size, stride, padding,
                            patch_size_3, stride_3, padding_3,
                            dt, format='torch', stack_dim_3=False) :
    if format == 'torch' :
        volume = volume.transpose(1, 2, 0)
    if stride > 1 : raise NotImplementedError()
    volume_unfold = unfold_3d(volume, patch_size, stride, padding,
                              patch_size_3, stride_3, padding_3)

    H_unfold, W_unfold, T_unfold = volume_unfold.shape[:3]
    patch_array = volume_unfold.reshape(H_unfold, W_unfold, T_unfold, -1)
    patch_array = patch_array.reshape(H_unfold * W_unfold, T_unfold, -1)
    patch_array = patch_array.transpose(1, 0, 2)

    patch_array = patch_array.reshape(H_unfold * W_unfold * T_unfold, -1)

    coords = get_grid_coordinates((W_unfold, H_unfold), (patch_size / 2 - 0.5 - padding,
                                                         patch_size / 2 - 0.5 - padding))
    coords = np.tile(coords, (T_unfold, 1))
    ts = np.linspace(0, dt, T_unfold+2)[1:-1].repeat(H_unfold * W_unfold)
    if stack_dim_3 :
        coords = coords.reshape(T_unfold, -1, 2)
        ts = ts.reshape(T_unfold, -1)
        patch_array = patch_array.reshape(T_unfold, H_unfold*W_unfold, -1)
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



def tile_tensor(tensor, res, num_tiles_per_dim) :
    N, D, H, W = tensor.shape
    tensor_tiled = torch.zeros((N * num_tiles_per_dim**2, D, *res),
                               dtype=tensor.dtype, device=tensor.device)
    # keep tiles in batch together maybe easier to keep batch together
    # offsets are 0, ..., H - res[0]
    offsets_x = torch.linspace(0, W-res[1], num_tiles_per_dim, dtype=int)
    offsets_y = torch.linspace(0, H-res[0], num_tiles_per_dim, dtype=int)
    # unfold with stride???
    i = 0
    for x in offsets_x :
        for y in offsets_y :
            tensor_tiled[i*N:(i+1)*N] = tensor[..., y:y+res[0], x:x+res[1]]
            i += 1

    return tensor_tiled

# TODO: untile to coordinates
def untile_tensor(tensor, res, num_tiles_per_dim) :
    NT, D, HT, WT = tensor.shape
    T = num_tiles_per_dim ** 2
    N = NT // T
    offsets_x = torch.linspace(0, res[1] - WT, num_tiles_per_dim, dtype=int)
    offsets_y = torch.linspace(0, res[0] - HT, num_tiles_per_dim, dtype=int)
    tensor_untiled = torch.zeros((NT, D, *res),
                                 dtype=tensor.dtype, device=tensor.device)
    i = 0
    for x in offsets_x:
        for y in offsets_y:
            tensor_untiled[i*N : (i+1)*N, :, y:y+HT, x:x+WT] = tensor[i*N : (i+1)*N]
            i += 1

    return tensor_untiled


def untile_weight_tensor(tensor, res, num_tiles_per_dim) :
    NT, D, HT, WT = tensor.shape
    T = num_tiles_per_dim**2
    N = NT // T

    grid = coords_grid(1, HT, WT).to(tensor.device)
    grid = grid / torch.tensor([WT, HT], device=tensor.device).reshape((1, 2, 1, 1))
    grid = grid - 0.5
    grid_norm = grid.norm(dim=1, keepdim=True)
    w = torch.exp(torch.distributions.Normal(0., 0.2).log_prob(grid_norm).to(tensor.device))

    tensor_weighted = tensor * w

    tensor_untiled = untile_tensor(tensor_weighted, res, num_tiles_per_dim).reshape(T, N, D, *res)
    # TODO: need to untile here
    w_untiled = untile_tensor(w.repeat(T, 1, 1, 1), res, num_tiles_per_dim).reshape(T, 1,*res)
    w_untiled = w_untiled.sum(dim=0)
    w_untiled[w_untiled==0.] = 1.

    tensor_untiled = tensor_untiled.sum(dim=0) / w_untiled

    return tensor_untiled


