import numpy as np
from scipy import sparse

from utils import default

# TODO: how to do this within cropped frame without moving the crop
# NOTE: by supplying offset crop within subwindow
# TODO: how to choose horizontally or vertically
def flip_within_offset(locs, res, offset) :
    locs -= offset
    locs *= -1
    locs += res - 1
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


def interp_polarity_sparse(events_all, res, num_groups, T) :
    groups = []
    mode = 'edges'
    if mode == 'centered' :
        dt = T / num_groups
    elif mode == 'edges' :
        dt = T / (num_groups - 1)
    for g in range(num_groups) :
        if mode == 'centered' :
            t_group = 0.5 * dt + g * dt
        elif mode == 'edges' :
            t_group = dt * g
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
                                      np.ones([len(coords_unique), 1]) * t_group,
                                      p_sum.reshape(-1, 1)],
                                     axis=1))
    if len(groups) == 0 :
        return events_all
    return np.concatenate(groups, axis=0)


# DATA AUGMENTATION

def augment_sample(events, flow_coords, flows, *args, **kwargs) :
    num_events_in = len(events)
    coords, selection, scale = augment_coordinates(
        np.concatenate([events[:, :2], flow_coords], axis=0),
        *args, **kwargs)
    events = events[selection[:len(events)], :]
    events[:, :2] = coords[:len(events), :]

    flow_coords = coords[len(events):, :]

    flows = flows[selection[num_events_in:], :] * scale

    return events, flow_coords, flows


def augment_coordinates(
        coords, res, crop, random_crop_offset=False, fixed_crop_offset=None,
        random_moving=True, crop_keep_full_res=False, scale_crop=False,
        random_flip_horizontal=False, random_flip_vertical=False) :
    # TODO: remove scale crop
    scale = np.array([1., 1.])
    if crop:
        if random_crop_offset:
            crop_offset = (np.random.randint(0, res[0] - crop[0]),
                           np.random.randint(0, res[1] - crop[1]))
        else:
            crop_offset = default(fixed_crop_offset, (0, 0))

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

        """flows, xys_rect, flow_indxs_crop = self.get_valid_flows(flow_path,
                                                                res=default(self.crop,
                                                                            [640, 480]),
                                                                offset=crop_offset,
                                                                return_crop_indices=True)"""

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

    return coords, selection, scale

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


