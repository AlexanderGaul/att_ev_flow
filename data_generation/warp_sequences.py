from homography import *
from image_utils import *
from loading import *


def curve_flow(res_out, s_begin, s_end, hom_seq, curve_steps, M_cal=np.eye(3)) :
    #H_values_begin = evaluate_curves(s_begin, curves, curve_steps)
    #H_values_end = evaluate_curves(s_end, curves, curve_steps)

    H_begin = hom_seq.evaluate(s_begin) # hom_matrix_centered(H_values_begin, res_im, offset_xy)
    H_end = hom_seq.evaluate(s_end) # hom_matrix_centered(H_values_end, res_im, offset_xy)
    H_rel = H_end.dot(np.linalg.inv(H_begin))

    return hom_flow(H_rel, res_out, M_cal)


def evaluate_curves(s, curves, s_steps) :
    if s <= s_steps[0] :
        return curves[0].evaluate(0.).reshape(-1)
    elif s >= s_steps[-1] :
        return curves[-1].evaluate(1.).reshape(-1)

    idx = np.searchsorted(s_steps, s) - 1

    s_begin = s_steps[idx]

    return curves[idx].evaluate((s - s_begin) / (s_steps[idx+1] - s_begin)).reshape(-1)


def evaluate_curves_mul(s, curves, s_steps) :
    assert s >= s_steps[0] and s <= s_steps[-1]

    idx = np.searchsorted(s_steps, s) - 1

    s_begin = s_steps[idx]


# TODO curves as class to abstract from bezier
class HomographySequence :
    def __init__(self, curves, curve_steps, t_steps,
                 center=(0, 0), offset=(0, 0),
                 curves_as_deltas=False) :
        self.curves = curves
        self.curve_steps = curve_steps
        self.t_steps = t_steps

        self.center = center
        self.offset = offset

        self.curves_as_deltas=curves_as_deltas

    def evaluate(self, t) :
        if not self.curves_as_deltas :
            h_param = evaluate_curves(t, self.curves, self.t_steps)
            return hom_matrix_centered(h_param, self.center, self.offset)
        else :
            if t <= self.t_steps[0]:
                return hom_matrix_centered(self.curves[0].evaluate(0.).reshape(-1),
                                           self.center, self.offset)

            elif t >= self.t_steps[-1]:
                t = self.t_steps[-1]
                """
                return hom_matrix_centered(self.curves[-1].evaluate(1.).reshape(-1),
                                           self.center, self.offset)
                """

            idx = np.searchsorted(self.t_steps, t) - 1

            t_begin = self.t_steps[idx]
            H_acc = np.eye(3)
            for i in range(0, idx) :
                h_step = self.curves[i].evaluate(1.).reshape(-1)
                H_acc = hom_matrix_centered(h_step, self.center).dot(H_acc)
            h_step = self.curves[idx].evaluate((t - t_begin) / (self.t_steps[idx + 1] - t_begin)).reshape(-1)
            H_acc = hom_matrix_centered(h_step, self.center, self.offset).dot(H_acc)
            return H_acc



class HomographySequenceMul :
    def __init__(self, curves, curve_steps, t_steps, center=(0, 0), offset=(0, 0)) :
        pass


class ImageWarp :
    def __init__(self, im:np.ndarray, hom_seq:HomographySequence,
                 crop_offset=(0, 0), crop=None, res=None) :
        self.im = im

        self.hom_seq = hom_seq

        self.crop_offset = crop_offset
        self.crop = crop
        if res :
            self.res = res
        else :
            self.res = im.shape[:2]

        M_cal = np.eye(3)
        if self.crop :
            M_cal = t_matrix(-self.crop_offset[1], -self.crop_offset[0])
            M_cal = M_scale(self.res[1] / self.crop[1],
                            self.res[0] / self.crop[0]).dot(
                M_cal)
        else:
            M_cal = M_scale(self.res[1] / self.im.shape[1],
                            self.res[0] / self.im.shape[0]).dot(
                M_cal)

        self.M_cal = M_cal

    def get_image(self, t) :
        H = self.hom_seq.evaluate(t)

        return cv.warpPerspective(self.im, self.M_cal.dot(H), (self.res[1], self.res[0]))
        #return cv.warpPerspective(self.im, self.M_cal.dot(H), (self.im.shape[1], self.im.shape[0]))[:self.res[0], :self.res[1], :]

    def get_flow(self, t_begin, t_end) :
        # TODO: somehow add calibration matrix
        fl = curve_flow(self.res, t_begin, t_end,
                        self.hom_seq, self.hom_seq.t_steps,
                        M_cal=self.M_cal)
        return fl

    def get_mask_nearest(self, t) :
        if self.im.shape[2] < 4 :
            return np.ones(self.res, dtype=bool)
        else :
            H_t1 = self.hom_seq.evaluate(t)
            alpha = cv.warpPerspective(self.im[:, :, 3], self.M_cal.dot(H_t1),
                                       (self.res[1], self.res[0]),
                                       flags=cv.INTER_NEAREST)
            return alpha > 0


class MultiImageWarp :
    def __init__(self, warps) :
        self.warps = warps
        self.res = warps[0].res

    def get_image(self, t) :
        im = np.zeros((*self.res, 3))
        for warp in self.warps :
            im_warp = warp.get_image(t)
            im = overlay_image(im, im_warp)
        return im

    def get_flow(self, t_begin, t_end) :
        fl = np.zeros((*self.res, 2))
        for warp in self.warps :
            m = warp.get_mask_nearest(t_begin)
            fl[m] = warp.get_flow(t_begin, t_end)[m]
        return fl

