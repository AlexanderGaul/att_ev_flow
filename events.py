import numpy as np
from scipy import sparse

from utils import default

from numba import jit

# TODO: how to do this within cropped frame without moving the crop
# NOTE: by supplying offset crop within subwindow
# TODO: how to choose horizontally or vertically
def flip_within_offset(locs, res, offset) :
    locs -= offset
    locs *= -1
    locs += res
    locs -= 1
    locs += offset
    return locs


def sum_polarity(events) :
    locs, inv, cnts = np.unique(events[:, :2],
                                return_inverse=True,
                                return_counts=True,
                                axis=0)
    t_mean = np.zeros(len(locs))
    np.add.at(t_mean, inv, events[:, 2])
    t_mean /= cnts

    p_sum = np.zeros(len(locs))
    np.add.at(p_sum, inv, events[:, 3])
    events_summed = np.concatenate([locs,
                                    t_mean.reshape(-1, 1),
                                    p_sum.reshape(-1, 1)],
                                   axis=1)
    return events_summed

# TODO change signature in function calls
def sum_polarity_sparse(events, res=None) :
    if len(events) == 0 :
        return events
    res_1 = events[:, 1].max() + 1
    indxs = (events[:, 0] * res_1 + events[:, 1]).astype(int)

    t_matrix = sparse.csr_matrix((events[:, 2],
                                  (indxs,  np.arange(events.shape[0]))),
                                  (indxs.max() + 1, events.shape[0]))
    counts = np.diff(t_matrix.indptr)

    indxs_dense = np.arange(indxs.max() + 1)
    indxs_unique = indxs_dense[counts > 0]
    coords_unique = np.concatenate([(indxs_unique // res_1).reshape(-1, 1),
                                    (indxs_unique % res_1).reshape(-1, 1)], axis=1)

    t_mean = t_matrix.sum(1).A1[indxs_unique]
    t_mean /= counts[indxs_unique]
    # TODO: change data directly??
    #pol_matrix = sparse.csr_matrix((events[:, 3],
    #                              (indxs,  np.arange(events.shape[0]))),
    #                             (indxs.max() + 1, events.shape[0]))
    #p_sum = pol_matrix.sum(1).A1[indxs_unique]
    t_matrix[indxs, np.arange(events.shape[0])] = events[:, 3]
    p_sum = t_matrix.sum(1).A1[indxs_unique]
    return np.concatenate([coords_unique,
                           t_mean.reshape(-1, 1).astype(int),
                           p_sum.reshape(-1, 1)],
                          axis=1)

# TODO: use begin and end time instead
def bin_sum_polarity(events, num_bins, T) :
    agg_dt = T / num_bins
    agg_ts = [int(i * agg_dt) for i in range(num_bins)]
    agg_ts.append(T + 1)

    split_idxs = np.searchsorted(events[:, 2], agg_ts)
    split_idxs[0] = 0
    split_idxs[-1] = -1

    events_binned = np.concatenate([
        sum_polarity_sparse_var(events[split_idxs[i]:split_idxs[i + 1], :])
        # sum_polarity(event_slice[split_idxs[i]:split_idxs[i+1], :])
        for i in range(num_bins)
    ], axis=0)

    return events_binned

# TODO change signature in function calls
def sum_polarity_sparse_var(events) :
    if len(events) == 0 :
        return events
    res_1 = events[:, 1].max() + 1
    indxs = (events[:, 0] * res_1 + events[:, 1]).astype(int)

    count_matrix = sparse.csr_matrix((np.ones(len(indxs)),
                                      (np.zeros(events.shape[0]), indxs)),
                                     (1, indxs.max() + 1))
    counts = count_matrix.data
    indxs_unique = count_matrix.indices
    coords_unique = np.concatenate([(indxs_unique // res_1).reshape(-1, 1),
                                    (indxs_unique % res_1).reshape(-1, 1)], axis=1)

    t_matrix = sparse.csr_matrix((events[:, 2],
                                  (np.zeros(events.shape[0]), indxs)),
                                  (1, indxs.max() + 1))

    t_mean = t_matrix.data
    t_mean /= counts
    # TODO: change data directly??
    p_matrix = sparse.csr_matrix((events[:, 3],
                                  (np.zeros(events.shape[0]), indxs)),
                                 (1, indxs.max() + 1))
    p_sum = p_matrix.data

    return np.concatenate([coords_unique,
                           t_mean.reshape(-1, 1),
                           p_sum.reshape(-1, 1)],
                          axis=1)


def bin_interp_polarity(*args) :
    return interp_polarity_sparse(*args)


@jit
def interp_volume_jit(events, res, num_bins, t_begin, t_end):
    volume = np.zeros((num_bins, res[1], res[0]))

    t_norm = events[:, 2]
    t_norm = (num_bins - 1) * (t_norm - t_begin) / (t_end - t_begin)

    x0 = events[:, 0].astype(np.int64)
    y0 = events[:, 1].astype(np.int64)
    t0 = (t_norm + 1).astype(np.int64) - 1

    value = events[:, 3]

    #for xlim in (x0, x0 + 1):
    #    for ylim in (y0, y0 + 1):
    xlim = x0
    ylim = y0
    for tlim in (t0, t0 + 1):
        mask = (xlim < res[0]) & (xlim >= 0) & (ylim < res[1]) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
        interp_weights = value * (
                1 - np.abs(tlim - t_norm)) #* (1 - np.abs(xlim - events[:, 0])) * (1 - np.abs(ylim - events[:, 1]))

        index = (tlim[mask].astype(np.int64),
                 ylim[mask].astype(np.int64),
                 xlim[mask].astype(np.int64))
        interp_weights_masked = interp_weights[mask]
        for i in range(len(index[0])):
            volume[index[0][i], index[1][i], index[2][i]] += interp_weights_masked[i]
        # np.add.at(volume, index, interp_weights[mask])

    return volume


@jit
def interp_volume_jit_mask(events, res, num_bins, t_begin, t_end):
    volume = np.zeros((num_bins, res[1], res[0]))
    return_mask = True
    if return_mask :
        mask_vol = np.zeros((num_bins, res[1], res[0]))

    t_norm = events[:, 2]
    t_norm = (num_bins - 1) * (t_norm - t_begin) / (t_end - t_begin)

    x0 = events[:, 0].astype(np.int64)
    y0 = events[:, 1].astype(np.int64)
    t0 = (t_norm + 1).astype(np.int64) - 1

    value = events[:, 3]

    #for xlim in (x0, x0 + 1):
    #    for ylim in (y0, y0 + 1):
    xlim = x0
    ylim = y0
    for tlim in (t0, t0 + 1):
        mask = (xlim < res[0]) & (xlim >= 0) & (ylim < res[1]) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
        interp_weights = value * (
                1 - np.abs(tlim - t_norm)) #* (1 - np.abs(xlim - events[:, 0])) * (1 - np.abs(ylim - events[:, 1]))

        index = (tlim[mask].astype(np.int64),
                 ylim[mask].astype(np.int64),
                 xlim[mask].astype(np.int64))
        interp_weights_masked = interp_weights[mask]
        for i in range(len(index[0])):
            volume[index[0][i], index[1][i], index[2][i]] += interp_weights_masked[i]
            if return_mask :
                mask_vol[index[0][i], index[1][i], index[2][i]] = 1
        # np.add.at(volume, index, interp_weights[mask])

    return volume, mask_vol

def interp_volume_nojit(events, res, num_bins, t_begin, t_end) :
    volume = np.zeros((num_bins, res[1], res[0]))

    t_norm = events[:, 2]
    t_norm = (num_bins - 1) * (t_norm - t_begin) / (t_end - t_begin)

    x0 = events[:, 0].astype(np.int64)
    y0 = events[:, 1].astype(np.int64)
    t0 = (t_norm + 1).astype(np.int64) - 1

    value = events[:, 3]

    #for xlim in (x0, x0 + 1):
    #    for ylim in (y0, y0 + 1):
    xlim = x0
    ylim = y0
    for tlim in (t0, t0 + 1):
        mask = (xlim < res[0]) & (xlim >= 0) & (ylim < res[1]) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
        interp_weights = value * (
                1 - np.abs(tlim - t_norm)) #* (1 - np.abs(xlim - events[:, 0])) * (1 - np.abs(ylim - events[:, 1]))

        index = (tlim[mask].astype(np.int64),
                 ylim[mask].astype(np.int64),
                 xlim[mask].astype(np.int64))

        np.add.at(volume, index, interp_weights[mask])

    return volume


def interp_volume(events, res, num_bins, t_begin, t_end, normalize=True, return_mask=False) :
    if return_mask :
        volume, mask = interp_volume_jit_mask(events, res, num_bins, t_begin, t_end)
    else :
        volume = interp_volume_jit(events, res, num_bins, t_begin, t_end)

    if normalize:
        if not return_mask :
            mask = volume != 0
        if mask.any():
            mean = volume[mask].mean()
            std = volume[mask].std()
            if std > 0:
                volume[mask] = (volume[mask] - mean) / std
            else:
                volume[mask] = volume[mask] - mean
    if return_mask :
        return volume, mask
    return volume



def interp_polarity_sparse(events_all, res, num_groups, t_end, t_begin=0.,
                           return_ts=True, mode = 'edges') :
    groups = []
    T = (t_end - t_begin)
    if mode == 'centered' :
        dt = T / num_groups
    elif mode == 'edges' :
        dt = T / (num_groups - 1)
    for g in range(num_groups) :
        if mode == 'centered' :
            t_group = 0.5 * dt + g * dt + t_begin
        elif mode == 'edges' :
            t_group = dt * g + t_begin
        t_min = t_group - dt
        t_max = t_group + dt

        idxs = np.searchsorted(events_all[:, 2], [t_min, t_max])
        events = events_all[idxs[0]:idxs[1]]
        if len(events) == 0 :
            continue

        indxs = (events[:, 0] * res[1] + events[:, 1]).astype(int)

        count_matrix = sparse.csr_matrix((np.ones(len(indxs)),
                                          (np.zeros(events.shape[0]), indxs)),
                                         (1, indxs.max() + 1))
        counts = count_matrix.data
        indxs_unique = count_matrix.indices
        coords_unique = np.concatenate([(indxs_unique // res[1]).reshape(-1, 1),
                                        (indxs_unique % res[1]).reshape(-1, 1)], axis=1)

        p_interp = events[:, 3] * (1 - np.abs(t_group - events[:, 2]) / dt)
        p_matrix = sparse.csr_matrix((p_interp,
                                      (np.zeros(events.shape[0]), indxs)),
                                     (1, indxs.max() + 1))
        p_sum = p_matrix.data

        groups.append(np.concatenate([coords_unique,
                                      np.ones([len(coords_unique), 1]) *
                                      (t_group if return_ts else g),
                                      p_sum.reshape(-1, 1)],
                                     axis=1))
    if len(groups) == 0 :
        return events_all
    return np.concatenate(groups, axis=0)


@jit
def spatial_downsample(events, patch_size) :
    if patch_size == 1 :
        return events
    res = (int(events[:, 1].max())+1, int(events[:, 0].max())+1)
    frame = np.zeros((res[0] // patch_size, res[1] // patch_size))
    events_out = np.zeros((len(events) // patch_size ** 2, 4))
    idx_out = 0
    for i in range(len(events)) :
        x, y = int(events[i, 0]) // patch_size, int(events[i, 1]) // patch_size
        frame[y, x] += events[i, 3] / (patch_size ** 2)
        if abs(frame[y, x]) >= 1 :
            events_out[idx_out, :] = np.array([float(x), float(y), events[i, 2], np.sign(frame[y, x])])
            frame[y, x] -= np.sign(frame[y, x])
            idx_out += 1

    return events_out[:idx_out, :].copy()


def downsample_flow(coords, flows, patch_size) :
    if patch_size == 1 :
        return coords, flows
    N = len(coords)
    coords_scale = (coords / patch_size).astype(np.int64)
    res_1 = coords_scale[:, 0].max() + 1
    indxs = coords_scale[:, 1] * res_1 + coords_scale[:, 0]

    count_matrix = sparse.csr_matrix((np.ones(len(indxs)),
                                    (np.zeros(N), indxs)),
                                   (1, indxs.max() + 1))

    counts = count_matrix.data
    indxs_unique = count_matrix.indices

    coords_flows = np.zeros((N, 4))
    coords_flows[:, :2] = coords - (patch_size - 1) / 2
    coords_flows[:, 2:] = flows
    acc_matrix = sparse.csr_matrix((coords_flows.reshape(-1),
                                    (np.tile(np.arange(0, 4), N), np.repeat(indxs, 4))),
                                   (4, indxs.max() + 1))
    accs = acc_matrix.data
    means = accs / np.tile(counts, 4) / patch_size

    down = means.reshape((-1, 4), order='F')
    return down[:, :2], down[:, 2:]



@jit
def downsample_flow_jit(coords, flows, patch_size) :
    res = (int(coords[:, 1].max()) + 1, int(coords[:, 0].max()) + 1)
    coords_scale = (coords / patch_size).astype(np.int64)

    # TODO: do for loops or csr matrices
    flow_frame = np.empty((res[0] // patch_size, res[1]  // patch_size, 4))
    flow_frame[:] = np.NaN

    valid_count = 0
    for i in range(flow_frame.shape[0]) :
        for j in range(flow_frame.shape[1]) :
            coords_ij = (coords_scale[:, 1] == i) & (coords_scale[:, 0] == j)
            if coords_ij.any() :
                flow_frame[i, j, 0] = (coords[coords_ij, 0] - (patch_size - 1) / 2).mean() / patch_size
                flow_frame[i, j, 1] = (coords[coords_ij, 1] - (patch_size - 1) / 2).mean() / patch_size
                flow_frame[i, j, 2] = flows[coords_ij, 0].mean() / patch_size
                flow_frame[i, j, 3] = flows[coords_ij, 1].mean() / patch_size
                valid_count +=1

    coords_down = np.zeros((valid_count, 2))
    flows_down = np.zeros((valid_count, 2))
    fill_idx = 0
    for i in range(flow_frame.shape[0]):
        for j in range(flow_frame.shape[1]):
            if np.isfinite(flow_frame[i, j, :]).all() :
                coords_down[fill_idx, :] = flow_frame[i, j, :2]
                flows_down[fill_idx, :] = flow_frame[i, j, 2:]
                fill_idx += 1

    return coords_down, flows_down

# DATA AUGMENTATION

def generate_augmentation(res, crop=None, crop_keep_full_res=False, scale_crop=False,
                          random_crop_offset=False, fixed_crop_offset=None,
                          random_moving=True,
                          random_flip_horizontal=False, random_flip_vertical=False) :
    crop_offset = np.zeros(2)
    crop_move = np.zeros(2)
    if crop:
        if random_crop_offset:
            crop_offset = np.array((np.random.randint(0, res[0] - crop[0]),
                                    np.random.randint(0, res[1] - crop[1])))
        else:
            crop_offset = default(fixed_crop_offset, np.zeros(2))
            crop_offset = np.array(crop_offset)

        # TODO: rename crop_move into something like crop_target
        if random_moving and crop_keep_full_res:
            crop_move = np.array([np.random.randint(0, res[0] - crop[0]),
                                  np.random.randint(0, res[1] - crop[1])])
        elif scale_crop and crop_keep_full_res:
            crop_move = np.zeros(2)
        elif not crop_keep_full_res:
            crop_move = np.zeros(2)
        else:
            crop_move = crop_offset

    if random_flip_horizontal :
        random_flip_horizontal = np.random.binomial(1, 0.5, 1)[0] > 0.5

    if random_flip_vertical :
        random_flip_vertical = np.random.binomial(1, 0.5, 1)[0] > 0.5

    return crop_offset, crop_move, random_flip_horizontal, random_flip_vertical


def augment_sample(events, flow_coords, flows, *args, **kwargs) :
    return (augment_events(events, *args, **kwargs),
            *augment_flows(flow_coords, flows, *args, **kwargs))

def augment_events(events, *args, **kwargs) :
    event_coords, selection, _ = augment_coordinates(events[:, :2], *args, **kwargs)

    events = events[selection, :]
    events[:, :2] = event_coords
    return events

def augment_flows(coords, flows, *args, **kwargs) :
    coords, selection, scale = augment_coordinates(coords, *args, **kwargs)
    flows = flows[selection, :] * scale
    return coords, flows


def augment_coordinates(coords, res, crop,
                        crop_keep_full_res=False, scale_crop=False,
                        crop_offset=np.zeros(2), crop_move=np.zeros(2),
                        flip_horizontal=False, flip_vertical=False) :
    scale = np.array([1., 1.])
    if crop :
        selection = (
                (coords[:, 0] < crop_offset[0] + crop[0]) &
                (coords[:, 0] >= crop_offset[0]) &
                (coords[:, 1] < crop_offset[1] + crop[1]) &
                (coords[:, 1] >= crop_offset[1]))

        coords = coords[selection, :]
        coords[:, :2] += crop_move - crop_offset

        if scale_crop :
            scale = np.array(res) / np.array(crop)
            coords[:, :2] *= scale

        if flip_horizontal :
            coords[:, 0] = flip_within_offset(coords[:, 0], crop[0] * scale[0], crop_move[0])
            scale[0] *= -1

        if flip_vertical :
            coords[:, 1] = flip_within_offset(coords[:, 1], crop[1] * scale[1], crop_move[1])
            scale[1] *= -1

    else:
        selection = np.array([True]).repeat(len(coords))
        if flip_horizontal :
            coords[:, 0] = flip_within_offset(coords[:, 0], res[0], 0)
            scale[0] *= -1

        if flip_vertical :
            coords[:, 1] = flip_within_offset(coords[:, 1], res[1], 0)
            scale[1] *= -1

    return coords, selection, scale



"""def augment_coordinates(
        coords, res, crop, random_crop_offset=False, fixed_crop_offset=None,
        random_moving=True, crop_keep_full_res=False, scale_crop=False,
        random_flip_horizontal=False, random_flip_vertical=False) :
    # TODO: remove scale crop
    scale = np.array([1., 1.])
    if crop:
        if random_crop_offset:
            crop_offset = np.array((np.random.randint(0, res[0] - crop[0]),
                                    np.random.randint(0, res[1] - crop[1])))
        else:
            crop_offset = default(fixed_crop_offset, np.zeros(2))
            crop_offset = np.array(crop_offset)

        # TODO: rename crop_move into something like crop_target
        if random_moving and crop_keep_full_res:
            crop_move = np.array([np.random.randint(0, res[0] - crop[0]),
                                  np.random.randint(0, res[1] - crop[1])])
        elif scale_crop and crop_keep_full_res:
            crop_move = np.zeros(2)
        elif not crop_keep_full_res:
            crop_move = np.zeros(2)
        else:
            crop_move = crop_offset

        #flows, xys_rect, flow_indxs_crop = self.get_valid_flows(flow_path,
        #                                                        res=default(self.crop,
        #                                                                    [640, 480]),
        #                                                        offset=crop_offset,
        #                                                        return_crop_indices=True)
        
        selection = (
            (coords[:, 0] < crop_offset[0] + crop[0]) &
            (coords[:, 0] >= crop_offset[0]) &
            (coords[:, 1] < crop_offset[1] + crop[1]) &
            (coords[:, 1] >= crop_offset[1]))

        coords = coords[selection, :]
        coords[:, :2] += crop_move - crop_offset

        if scale_crop:
            scale = np.array(res) / np.array(crop)
            coords[:, :2] *= scale

        if random_flip_horizontal and np.random.binomial(1, 0.5, 1)[0] > 0.5:
            coords[:, 0] = flip_within_offset(coords[:, 0], crop[0], crop_move[0])
            scale[0] *= -1

        if random_flip_vertical and np.random.binomial(1, 0.5, 1)[0] > 0.5:
            coords[:, 1] = flip_within_offset(coords[:, 1], crop[1], crop_move[1])
            scale[1] *= -1

    else:
        selection = np.array(len(coords)*[True])
        if random_flip_horizontal and np.random.binomial(1, 0.5, 1)[0] > 0.5:
            coords[:, 0] = flip_within_offset(coords[:, 0], res[0], 0)
            scale[0] *= -1

        if random_flip_vertical and np.random.binomial(1, 0.5, 1)[0] > 0.5:
            coords[:, 1] = flip_within_offset(coords[:, 1], res[1], 0)
            scale[1] *= -1

    return coords, selection, scale"""


def backward_events(events, dt) :
    events[:, 2] *= -1
    events[:, 2] += dt
    events[:, 3] *= -1
    events = np.flip(events, axis=0).copy()
    return events

def backward_flows(coords, flows, res=None) :
    coords = coords + flows
    flows *= -1
    if res :
        in_bounds = ((coords[:, 0] >= 0) & (coords[:, 1] >= 0) &
                     (coords[:, 0] < res[0]) & (coords[:, 1] < res[1]))
        coords = coords[in_bounds]
        flows = flows[in_bounds]
    return coords, flows

def backward_volume(volume) :
    assert len(volume.shape) == 3
    return -np.flip(volume, axis=0).copy()


"""
def interp_polarity_for(events_all, res, groups, T) :
    # assume events are sorted
    dt = T / groups
    for g in range(groups) :
        t_group = 0.5 * dt + g * dt
        t_min = t_group - dt
        t_max = t_group + dt

        # TODO find first in sorted
        idxs = np.searchsorted(events[:, 2], [t_min, t_max])
        events = events_all[idxs[0]:idxs[1]]

        p_interp = events

        unique_coords = {}
        for e in events :
            coord = (e[0], e[1])
            if coord not in unique_coords :
                unique_coords[coord] = 1
            else
                unique_coords[coord] += 1
"""


