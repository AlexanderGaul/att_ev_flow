import numpy as np

from utils import get_grid_coordinates


def H_euclidean(angle, tx, ty) :
    H = np.eye(3)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    H[:2, :2] = R
    H[:2, 2] = np.array([tx, ty])
    return H


def H_affine(sx, sy) :
    H = np.eye(3)
    H[1, 0] = sx
    H[0, 1] = sy
    return H


def H_projective(p1, p2) :
    H = np.eye(3)
    H[2, 0] = p1
    H[2, 1] = p2
    return H


def H_scale(s) :
    H = np.eye(3)
    H[0, 0] = s
    H[1, 1] = s
    return H

def M_scale(x, y)  :
    M = np.eye(3)
    M[0, 0] = x
    M[1, 1] = y
    return M

def t_matrix(tx, ty) :
    return np.array([[1., 0., tx], [0., 1., ty], [0., 0., 1.]])


def hom_matrix(tx, ty, angle, sx, sy, p1, p2, s=1.) :
    return H_euclidean(angle, tx, ty).dot(
            H_affine(sx, sy)).dot(
            H_projective(p1, p2)).dot(
            H_scale(s))


def hom_matrix_centered(h_param, center_xy=(0, 0), offset_xy=(0, 0), rot=0) :
    return t_matrix(offset_xy[0], offset_xy[1]).dot(
        t_matrix(center_xy[0], center_xy[1])).dot(
        H_euclidean(rot, 0, 0)).dot(
        hom_matrix(*h_param)).dot(
        t_matrix(-center_xy[0], -center_xy[1]))


def hom_flow(H, res, M_cal=np.eye(3)) :
    grid = get_grid_coordinates(res_xy=np.flip(res))
    grid = np.concatenate([grid, np.ones([len(grid), 1])], axis=1)
    grid_H = grid.dot(M_cal.dot(H).dot(np.linalg.inv(M_cal)).transpose())
    grid_H = grid_H / grid_H[:, 2].reshape(-1, 1)
    flow = grid_H - grid
    flow = flow[:, :2].reshape(*res, 2)

    return flow
