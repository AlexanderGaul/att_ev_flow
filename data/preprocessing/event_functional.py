import numpy as np

from numba import jit

from events import interp_volume

from scipy.signal import convolve2d


def event_data_2_array(event_data, t_0_us) :
    event_array = np.concatenate([event_data['x'][:, np.newaxis],
                                  event_data['y'][:, np.newaxis],
                                  event_data['t'][:, np.newaxis] - t_0_us,
                                  event_data['p'][:, np.newaxis]
                                  ], axis=1, dtype=np.float64)
    event_array[:, 2] /= 1000.
    return event_array


def event_mask(event_array, res, dt, area=1) :
    _, mask = interp_volume(event_array, res, 1, 0, dt, False, True)
    mask = mask.reshape(tuple(reversed(res)))
    if area > 1 :
        mask = convolve2d(mask.astype(float), np.ones((area, area)), 'same')

    return mask.astype(bool)



# TODO: jit
# TODO: test with simple array
@jit
def event_array_patch_context(event_array, res, dt, num_events, patch_size,
                              t_relative=False) :
    event_accumulator = -np.ones((*res, num_events), dtype=np.int64)
    event_array_context = np.zeros((len(event_array), 2 * num_events * patch_size * patch_size))
    # TODO: set timestamps -1
    event_array_context[:, patch_size*patch_size*num_events:] = -1
    for i in range(len(event_array)) :
        x, y = event_array[i, 0], event_array[i, 1]
        for ix in range(patch_size) :
            dx = ix - (patch_size) // 2
            xc = int(x + dx)
            if xc < 0 or xc >= res[0] :
                continue
            for iy in range(patch_size) :
                dy = iy - (patch_size) // 2
                yc = int(y + dy)
                if  yc < 0 or yc >= res[1] :
                    continue
                for ie in range(num_events) :
                    idx = event_accumulator[xc, yc, ie]
                    if event_accumulator[xc, yc, ie] < 0 :
                        continue
                    ic = (ix + patch_size * iy + patch_size*patch_size * ie)
                    #event_array_context[i, ic + 0] = dx / (patch_size // 2)
                    #event_array_context[i, ic + 1] = dy / (patch_size // 2)
                    if not t_relative :
                        event_array_context[i, ic + patch_size*patch_size*num_events] = event_array[idx, 2] / dt * 2 - 1
                    else :
                        event_array_context[i, ic + patch_size*patch_size*num_events] = 1. - (event_array[i, 2] - event_array[idx, 2]) / dt
                    event_array_context[i, ic] = event_array[idx, 3]
        # TODO update accumulator
        # have recent ones at earlier indices always have to move

        event_accumulator[int(x), int(y), 1:] = event_accumulator[int(x), int(y), :-1]
        event_accumulator[int(x), int(y), 0] = i

    return event_array_context

@jit
def event_array_patch_time_surface(event_array, res, dt, patch_size, t_relative=True) :
    time_surface = -np.ones((*res, 2))
    event_array_patch_ts = -np.ones((len(event_array), patch_size**2 * 2))
    for i in range(len(event_array)) :
        x, y = event_array[i, 0], event_array[i, 1]
        for ix in range(patch_size) :
            dx = ix - (patch_size) // 2
            xc = int(x + dx)
            if xc < 0 or xc >= res[0] : continue
            for iy in range(patch_size) :
                dy = iy - (patch_size) // 2
                yc = int(y + dy)
                if yc < 0 or yc >= res[1]: continue
                ic = (ix + patch_size * iy)  * 2
                if time_surface[xc, yc, 0] >= 0 :
                    if t_relative :
                        event_array_patch_ts[i, ic] = 1. - (event_array[i, 2] - time_surface[xc, yc, 0]) / dt
                    else :
                        event_array_patch_ts[i, ic] = time_surface[xc, yc, 0] / dt * 2 - 1
                if time_surface[xc, yc, 1] >= 0 :
                    if t_relative :
                        event_array_patch_ts[i, ic+1] = 1. - (event_array[i, 2] - time_surface[xc, yc, 1]) / dt
                    else :
                        event_array_patch_ts[i, ic+1] = time_surface[xc, yc, 1] / dt * 2 - 1

        if event_array[i, 3] < 0 :
            time_surface[int(x), int(y), 0] = event_array[i, 2]
        else :
            time_surface[int(x), int(y), 1] = event_array[i, 2]
    return event_array_patch_ts


def event_array_time_surface_sequence(event_array, res, dt, t_bins,
                                      into_the_future=False,
                                      crop_ts=False,
                                      repeat_values=False) :
    if into_the_future :
        event_array = np.flip(event_array, axis=0).copy()
        event_array[:, 2] = dt - event_array[:, 2]
    sign = -1 if into_the_future else 1

    time_surface_shots = -np.ones((*reversed(res), 2, t_bins))
    time_surface = -np.ones((*reversed(res), 2))
    surf_idx = 0
    t_seq = dt / t_bins
    t_seq_begin = 0.
    t_seq_end = t_seq
    t = 0.
    for i in range(len(event_array)) :
        event = event_array[i]
        t = event[2]
        if t > t_seq_end :
            if not crop_ts :
                time_surface_shots[..., surf_idx] = time_surface.copy()
            else :
                time_surface_shots[..., surf_idx] -= t_seq_begin
                time_surface_shots[..., surf_idx] /= t_seq

                raise NotImplementedError("")

            surf_idx += 1
            t_seq_begin = t_seq_end
            t_seq_end += t_seq

            # TODO: minus or something
        if event[3] > 0 :
            time_surface[int(event[1]), int(event[0]), 1] = event[2] / dt
        else :
            time_surface[int(event[1]), int(event[0]), 0] = event[2] / dt

    if t_seq_end >= t :
        if not crop_ts:
            time_surface_shots[..., surf_idx] = time_surface.copy()
    if repeat_values :
        for i in range(t_bins) :
            pass
    return time_surface_shots



# TODO: what name do we want for this
# volume_2_patches_timesplit


def event_array_2_image_pair(event_array, res, dt) :
    raise NotImplementedError("Implementation low priority")
    pass


def event_array_backward(event_array, dt) :
    event_array[:, 3] *= -1
    event_array[:, 2] *= -1
    event_array[:, 2] += dt
    return np.flip(event_array, axis=0).copy()


def event_array_remove_zeros(event_array, raw_dims=[3]) :
    return event_array[(event_array[:, raw_dims] != 0.).any(axis=-1), :]


def event_noise(l, dt, res, sort=False) :
    arrival_times = np.random.exponential(l, res)
    coords = np.argwhere(arrival_times < dt)
    times = arrival_times[coords[:, 0], coords[:, 1]]
    events = [np.concatenate([np.flip(coords, axis=1),
                             times.reshape(-1, 1),
                             np.random.binomial(1, 0.5, len(coords)).reshape(-1, 1) * 2 - 1], axis=1)]
    while len(coords) > 0 :
        times = np.random.exponential(l, len(coords)) + times
        coords = coords[times < dt]
        times = times[times < dt]
        events.append(np.concatenate([np.flip(coords, axis=1),
                                      times.reshape(-1, 1),
                                      np.random.binomial(1, 0.5, len(coords)).reshape(-1, 1) * 2 - 1], axis=1))
    if sort :
        events = np.concatenate([e[e[:, 2].argsort()] for e in events], axis=0)
    else :
        events = np.concatenate(events, axis=0)
    return events

@jit
def merge_sorted(a, b, axis=0) :
    array_sorted = np.zeros((len(a) + len(b), a.shape[1]),
                            dtype=a.dtype)
    i = 0
    j = 0
    for k in range(len(array_sorted)) :
        if i < len(a) and (j >= len(b) or a[i, axis] < b[j, axis]) :
            array_sorted[k, :] = a[i, :]
            i += 1
        else :
            array_sorted[k, :] = b[j, :]
            j += 1
        k += 1

    return array_sorted

