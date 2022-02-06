import numpy as np
import cv2


def remap_linear(values, bounds_old, bounds_new) :
    return ((values - bounds_old[0]) * (bounds_new[1] - bounds_new[0]) / (bounds_old[1] - bounds_old[0])
             + bounds_new[0])

def distort_radtan(xys, ks) :
    r_2 = xys[:, [0]]**2 + xys[:, [1]]**2
    r_4 = r_2 ** 2

    xys_rad = xys * (np.ones(xys.shape) + ks[0] * r_2 + ks[1] * r_4)

    xs_tan = (2 * ks[2] * xys[:, [0]] * xys[:, [1]] +
              ks[3] * (r_2 + 2 * xys[:, [0]]**2))
    ys_tan = (ks[2] * (r_2 + 2 * xys[:, [1]]**2) +
              2 * ks[3] * xys[:, [0]] * xys[:, [1]])

    xys_tan = np.concatenate([xs_tan, ys_tan], axis=1)

    return xys_rad + xys_tan

def dist2rect(xys, K_dist, dist, R, K_rect) :
    xys_K = cv2.undistortPoints(xys, K_dist, dist).reshape(-1, 2)
    xys_K_rot = np.concatenate([xys_K, np.ones([len(xys_K), 1]) ], axis=1).dot(R.transpose())
    return xys_K_rot[:, :2] / xys_K_rot[:, [2]] * K_rect[[[0, 1]], [[0, 1]]] + K_rect[:2, [2]].transpose()

def rect2dist(xys, K_rect, R, dist, K_dist):
    xys_cam = (xys - K_rect[:2, [2]].transpose()) / K_rect[[[0, 1]], [[0, 1]]]
    xys_cam_hom = np.concatenate([xys_cam, np.ones([len(xys_cam), 1])], axis=1)
    xys_cam_rot = xys_cam_hom.dot(R)
    return distort_radtan(xys_cam_rot[:, :2] / xys_cam_rot[:, [2]], dist) * K_dist[[[0, 1]], [[0, 1]]] + K_dist[:2, [2]].transpose()

def collate_tuple_list(batch):
    return tuple(list(d) for d in zip(*batch))

# TODO: how to do this??
def collate_dict_list(batch) :
    # batch is iterable
    res = {}
    for k in batch[0].keys() :
        sub_list = [batch[i][k] for i in range(len(batch))]
        if type(batch[0][k]) is dict :
            res[k] = collate_dict_list(sub_list)
        else :
            res[k] = sub_list
    return res

def reverse_events(events, dt=100) :
    events[:, 2] = dt - events[:, 2]
    events[:, 3] *= -1
    return events

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d