import numpy as np
import cv2

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
    xys_K_hom = np.concatenate([xys_K, np.ones([len(xys_K), 1]) ], axis=1)
    return xys_K_hom.dot(R.transpose())[:, :2] * K_rect[[[0, 1]], [[0, 1]]] + K_rect[:2, [2]].transpose()

def rect2dist(xys, K_rect, R, dist, K_dist):
    xys_cam = (xys - K_rect[:2, [2]].transpose()) / K_rect[[[0, 1]], [[0, 1]]]
    xys_cam_hom = np.concatenate([xys_cam, np.ones([len(xys_cam), 1])], axis=1)
    return distort_radtan(xys_cam_hom.dot(R)[:, :2], dist) * K_dist[[[0, 1]], [[0, 1]]] + K_dist[:2, [2]].transpose()

