import numpy as np
import math

import bezier
from scipy.interpolate import CubicSpline

from homography import *

from warp_sequences import HomographyCurveSequence, HomographySplineSequence


# TODO: need to create genrators that can easily be replaced
# [] what class structure do we even use here?
# [] do we actually need classes here?


def t_rnd_uni_fixed_length(length) :
    angle = np.random.uniform(0., 2*math.pi)
    x = np.sin(angle) * length
    y = np.cos(angle) * length
    return x, y


def t_rnd_uni_length(l_min, l_max) :
    angle = np.random.uniform(0., 2 * math.pi)
    l = np.random.uniform(l_min, l_max)
    x = np.sin(angle) * l
    y = np.cos(angle) * l
    return x, y


def random_homography_beta(angle_max, tx_max, ty_max,
                           sx_max, sy_max,
                           p1_max, p2_max,
                           scale_min=1., scale_max=1.) :
    sign = np.random.binomial(1, 0.5, 8) * 2 - 1
    factor = np.random.beta(2, 4, 8) * - 1 + 1

    if scale_min == scale_max :
        factor[-1] = 1.
        sign[-1] = 1.
    else :
        if sign[-1] < 0 :
            factor[-1] = factor[-1] * -1 + 1
        factor[-1] = factor[-1] * (scale_max - scale_min) + scale_min
        sign[-1] = 1

    return sign * factor * np.array([tx_max, ty_max, angle_max, sx_max, sy_max, p1_max, p2_max, 1.])


def hom_rn_uni(t, a, s, p, z) :
    mins = np.array([-t, -t, -a, -s, -s, -p, -p, 1. - z])
    maxs = np.array([t, t, a, s, s, p, p, 1. + z])
    return np.random.uniform(mins, maxs)


def hom_seq_rn_limits_beta(T, num_curves, limits_hom, limit_dt_scale) :
    pass


def hom_seq_rn_uni_lim_steps(T, num_curves, lims) :
    # how to add homographies to previous ones with bezier curves
    # h0 -> h1 -> h2
    # h0 -> h1 * h0 -> h2 * h1 * h0
    # interp(t, h0, h1, h2) * h0
    # interp(0, ...) = id
    pass

# TODO: how to structure randomness parameters
# by type of transform
# most important is translation


class GenHomUniOffset :
    def __init__(self, t_mean, dt_max, a_max, s_max, p_max, dz_max) :
        self.t_mean = t_mean
        self.dt_max = dt_max
        self.a_max = a_max
        self.s_max = s_max
        self.p_max = p_max
        self.dz_max = dz_max

    def __call__(self) :
        _, _, a, s1, s2, p1, p2, z = hom_rn_uni(0., self.a_max,
                                                self.s_max, self.p_max, self.dz_max)
        x, y = t_rnd_uni_length(self.t_mean - self.dt_max,
                                self.t_mean + self.dt_max)

        return (x, y, a, s1, s2, p1, p2, z)


class GenHomLeftRight :
    def __init__(self, length) :
        self.length = length

    def __call__(self) :
        x = self.length
        if np.random.binomial(1, 0.5, 1)[0] > 0.5 :
            x = -x
        return (x, 0, 0, 0, 0, 0, 0, 1.)


class GenHomSeqLeftRightVarLength :
    def __init__(self, length_min, length_max) :
        self.length_min = length_min
        self.length_max = length_max

    def __call__(self, num) :
        sign = -1. if np.random.binomial(1, 0.5, 1)[0] < 0.5 else 1.
        hom_seq = []
        for i in range(num) :
            x = np.random.uniform(self.length_min, self.length_max) * sign
            hom_seq.append((x, 0, 0, 0, 0, 0, 0, 1.))
        return hom_seq


class GenHomSeqIID :
    def __init__(self, hom_gen) :
        self.hom_gen = hom_gen

    def __call__(self, num) :
        return [self.hom_gen() for i in range(num)]


class GenHomCurveSeqSteps :
    def __init__(self, hom_gen, T, num_curves, dt_rnd) :
        self.hom_gen = hom_gen
        self.T = T
        self.num_curves = num_curves
        self.dt_rnd = dt_rnd

    def __call__(self, center=(0, 0), offset=(0, 0), rot=0) :
        dt = self.T / self.num_curves
        dt_min = dt - self.dt_rnd * dt
        dt_max = dt + self.dt_rnd * dt

        t_steps = [0.]
        h_steps = [np.array(h) for h in self.hom_gen(self.num_curves)]
        curves = []

        h_id = np.zeros(8)
        h_id[-1] = 1.

        for i in range(self.num_curves) :
            t_min = max(t_steps[-1] + dt_min, self.T - (self.num_curves - i - 1) * dt_max)
            t_max = min(self.T - (self.num_curves - i - 1) * dt_min, t_min + dt_max)

            t_steps.append(np.random.uniform(t_min, t_max))

            curves.append(bezier.Curve(np.stack([h_id, h_steps[i]]).transpose(),
                                       1))

        hom_seq = HomographyCurveSequence(curves, h_steps, t_steps,
                                          center, offset, rot,
                                          curves_as_deltas=True)
        return hom_seq


class GenHomSplineSeq :
    def __init__(self, hom_gen, T, num_keypoints, dt_rnd) :
        self.hom_gen = hom_gen
        self.T = T
        self.num_keypoints = num_keypoints
        self.dt_rnd = dt_rnd

    def __call__(self, center=(0, 0), H_0=np.eye(3), offset=(0, 0)) :
        dt = self.T / (self.num_keypoints - 1)
        dt_min = dt - self.dt_rnd * dt
        dt_max = dt + self.dt_rnd * dt
        h_keypoints = [np.array(h) for h in self.hom_gen(self.num_keypoints)]
        t_steps = [0.]
        for i in range(self.num_keypoints - 1) :
            t_min = max(t_steps[-1] + dt_min, self.T - (self.num_keypoints - 1 - i - 1) * dt_max)
            t_max = min(self.T - (self.num_keypoints - 1 - i - 1) * dt_min, t_min + dt_max)
            t_steps.append(np.random.uniform(t_min, t_max))

        spline = CubicSpline(np.array(t_steps), np.stack(h_keypoints))
        hom_seq = HomographySplineSequence(spline, h_keypoints, t_steps, center, H_0)
        return hom_seq


def random_curves(T, num_curves, limits_hom, limit_dt_scale) :
    dt = T / num_curves
    dt_min = dt - limit_dt_scale * dt
    dt_max = dt + limit_dt_scale * dt

    t_steps = [0.]
    h_steps = [np.array([*([0.]*7), 1.])]
    curves = []
    for i in range(num_curves) :
        t_min = max(t_steps[-1] + dt_min, T - (num_curves-i-1) * dt_max)
        t_max = min(T - (num_curves-i-1) * dt_min, t_min + dt_max)

        t_steps.append(np.random.uniform(t_min, t_max))

        h_steps.append(random_homography_beta(*limits_hom))
        h_steps.append(random_homography_beta(*limits_hom))

        curves.append(bezier.Curve(np.stack(h_steps[-3:]).transpose(),
                                   2))

    return curves, h_steps, t_steps