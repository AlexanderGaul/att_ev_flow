import numpy as np


def frame_patched(frame, patch_size) :
    res = frame.shape[:2]
    res_new = (res[0] // patch_size, res[1] // patch_size)

    frame_patched_x = frame.reshape(res[0], res_new[1], patch_size, -1)
    frame_patched_yx = frame_patched_x.transpose(1, 0, 2, 3).reshape(res_new[1], res_new[0],
                                                                     patch_size, patch_size, -1)
    frame_patched = frame_patched_yx.transpose(1, 0, 2, 3, 4)
    return frame_patched.reshape(res_new[0], res_new[1], patch_size, patch_size, *(frame.shape[2:]))

class FlowFrameSpatialDownsample :
    def __init__(self, factor=1) : self.factor = factor
    def __call__(self, data) :
        if self.factor != 1 :
            # TODO:
            # reshape frame and valid frame into factor**2 patches
            flow_frame = data.pop('flow_frame')
            flow_mask = data.pop('flow_frame_valid')
            flow_eval = data.pop('flow_frame_eval')

            flow_frame_patched = frame_patched(flow_frame, self.factor)
            res_new = flow_frame_patched.shape[:2]
            flow_patched = flow_frame_patched.reshape(*res_new, self.factor**2, 2)

            flow_mask_patched = frame_patched(flow_mask, self.factor)
            mask_patched = flow_mask_patched.reshape(*res_new, self.factor**2)

            flow_eval_patched = frame_patched(flow_eval, self.factor)
            eval_patched = flow_eval_patched.reshape(*res_new, self.factor**2)

            flow_dist = flow_patched[:, :, :, None, :] - flow_patched[:, :, None, :, :]
            flow_dist = np.abs(flow_dist.sum(axis=-1))
            flow_dist[~mask_patched] = 0.
            flow_dist = flow_dist.sum(-1)
            flow_dist[~mask_patched] = np.inf

            min_dist = np.argmin(flow_dist, axis=-1)
            flow_median = np.take_along_axis(flow_patched, min_dist[:, :, None, None], 2)[:, :, 0, :]
            mask_patched = mask_patched.any(axis=-1)

            data['flow_frame'] = flow_median / self.factor
            data['flow_frame_valid'] = mask_patched
            data['flow_frame_eval'] = eval_patched.any(axis=-1)

        return data
