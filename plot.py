import math
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

import io

from events import sum_polarity_sparse_var

def flow_frame_color(flow_frame, on_white=True, max_length=None, mask_valid=None) :
    assert len(flow_frame.shape) == 3
    assert flow_frame.shape[2] == 2

    im_hsv = np.zeros([*flow_frame.shape[:2], 3])  # shape: H, W, C

    im_hsv[:, :, 2].flat, im_hsv[:, :, 0].flat = \
        cv2.cartToPolar(flow_frame[:, :, 0].flat.copy(), flow_frame[:, :, 1].flat.copy(),
                        None, None, True)

    if mask_valid is not None :
        im_hsv[~mask_valid, :] = 0.

    im_hsv[..., 0] /= 2.
    im_hsv[..., 1] = 255.

    if max_length is None :
        im_hsv[:, :, 2] *= (255. / im_hsv[:, :, 2].max())
    else :
        im_hsv[..., 2] *= (255. / max_length)
        im_hsv[..., 2] = np.minimum(im_hsv[..., 2], 255.)

    if on_white :
        V = im_hsv[..., 2].copy()
        im_hsv[..., 2] = im_hsv[..., 1]
        im_hsv[..., 1] = V

    im_hsv = im_hsv.astype(int).astype(np.ubyte)
    im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
    return im_rgb


def flow_color(coords, flows, res, on_white=True, max_length=None) :
    # TODO: could acutally interpolate the coordinates if not integers
    coords = coords.astype(int)
    in_bound = ((coords[:, 0] >= 0) & (coords[:, 1] >= 0) &
                (coords[:, 0] < res[0]) & (coords[:, 1] < res[1]))
    coords = coords[in_bound, :]
    flows = flows[in_bound, :]

    # TODO: put None where no coordinate?
    flow_im = np.zeros((*np.flip(res), 2))
    flow_im[coords[:, 1], coords[:, 0], :] = flows

    return flow_frame_color(flow_im, on_white, max_length)

# TODO:
def from_plt(im) :
    pass


def plot_flow_color(coords, flows:np.ndarray, res,
                    on_white=True, max_length=None,
                    figsize=None) :
    im_rgb = flow_color(coords, flows, res, on_white, max_length)
    if figsize == None :
        figsize = (im_rgb.shape[0] // 16, im_rgb.shape[1] // 16)

    # TODO: refactor pyplot image buffer reading
    plt.figure(figsize=(im_rgb.shape[0] // 16, im_rgb.shape[1] // 16))
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    ax.set_xlim([0, im_rgb.shape[1]])
    ax.set_ylim([im_rgb.shape[0], 0])

    im_rgb_log = np.log(im_rgb.astype(float)+1)
    im_rgb_log /= im_rgb_log.max()
    plt.imshow(im_rgb)

    return capture_plt_as_np()


def plot_flow_error_abs(coords, pred, flows, res, figsize=None) :
    if figsize == None :
        figsize = (res[1] // 16, res[0] // 16)
    im = np.zeros(np.flip(res))
    im[coords[:, 1].astype(int),
       coords[:, 0].astype(int)] = np.abs(pred - flows).sum(axis=1)
    im = im / im.max()
    #im = (im  * 255.).astype(np.ubyte)

    plt_set_up_tight_figure(im.shape[:2], figsize=figsize)

    plt.imshow(im)
    return capture_plt_as_np()

def plot_flow_error_abs_frame(pred, flows, mask=None, figsize=None) :
    if mask is not None :
        im = np.abs(pred - flows).sum(axis=-1)
    else :
        im = np.zeros(pred.shape[:2])
        im[mask] = np.abs(pred - flows)[mask].sum(axis=-1)

    if figsize == None :
        figsize = (im.shape[0] // 16, im.shape[1] // 16)

    plt_set_up_tight_figure(im.shape[:2],
                            figsize=figsize)

    plt.imshow(im)
    return capture_plt_as_np()


def plot_flow_error_multi(coords, pred, flows, res=None, figsize=None) :
    if figsize == None :
        figsize = (res[1] // 16, res[0] // 16)
    pred_norm = np.linalg.norm(pred, axis=1, ord=2)
    flow_norm = np.linalg.norm(flows, axis=1, ord=2)
    l1_norm = np.linalg.norm(pred - flows, axis=1, ord=1)
    length_diff = np.abs(pred_norm - flow_norm)

    cosine_dist = 1 - np.abs((pred * flows).sum(axis=1) /
                        (pred_norm * flow_norm))
    length_diff_scale = np.minimum((length_diff / flow_norm / 2),
                                   np.ones(len(flow_norm)))

    rgb = np.zeros([len(pred), 3])
    rgb[:, 0] = cosine_dist * 3
    rgb[:, 1] = np.maximum(np.ones(len(pred)) - l1_norm / 4.,
                           np.zeros(len(pred)))
    rgb[:, 2] = length_diff_scale

    rgb = np.maximum(rgb, np.zeros(rgb.shape))
    rgb = np.minimum(rgb, np.ones(rgb.shape))

    im_rgb = np.zeros([*np.flip(res), 3])  # shape: H, W, C

    im_rgb[coords[:, 1].astype(int).reshape(-1),
           coords[:, 0].astype(int).reshape(-1), :] = rgb

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    ax.set_xlim([0, im_rgb.shape[1]])
    ax.set_ylim([im_rgb.shape[0], 0])

    #im_rgb_log = np.log(im_rgb.astype(float) + 1)
    #im_rgb_log /= im_rgb_log.max()
    plt.imshow(im_rgb)

    return capture_plt_as_np()


def plot_flow(img, coords, flows, flow_res=None, freq=10, figsize=None, c='k') :
    if figsize == None :
        figsize = (img.shape[1] / 10,
                   img.shape[0] / 10)
    if flow_res is None  :
        flow_res = img.shape
    scale_x = img.shape[0] / flow_res[0]
    scale_y = img.shape[1] / flow_res[1]

    plt_set_up_tight_figure(img.shape[:2], figsize=figsize)

    # TODO adapt freq to work on arbitrary numbers and be equally distributed
    coords_binned = coords.astype(int) // freq
    _, indx = np.unique(coords_binned, axis=0, return_index=True)

    plt.imshow(img, cmap='gray')
    plt.scatter(coords[indx, 0], coords[indx, 1], s=10, c=c)

    for i in indx  :
        plt.plot([coords[i, 0] * scale_x, coords[i, 0] * scale_x + flows[i, 0] * scale_x],
                 [coords[i, 1] * scale_y, coords[i, 1] * scale_y + flows[i, 1] * scale_y],
                 linewidth=1, c=c)


def plot_flow_grid(img, flows_grid, flow_res=None, freq=10, figsize=None, c='k',
                   flows_grid_valid=None) :
    if figsize == None :
        figsize = (img.shape[1] / 16,
                   img.shape[0] / 16)
    if flow_res is None  :
        flow_res = img.shape
    scale_x = img.shape[0] / flow_res[0]
    scale_y = img.shape[1] / flow_res[1]

    plt_set_up_tight_figure(img.shape[:2], figsize=figsize)

    plt.imshow(img, cmap='gray')
    # TODO: convert this
    #plt.scatter(coords[::freq, 0], coords[::freq, 1], s=6, c='g')
    for i in range(0, flows_grid.shape[1], freq) :
        for j in range(0, flows_grid.shape[0], freq) :
            if flows_grid_valid is not None :
                if not flows_grid_valid[j, i] : continue
            plt.scatter(i * scale_x, j * scale_y, s=10, c=c)
            plt.plot([i * scale_x, i * scale_x + flows_grid[j, i, 0] * scale_x],
                     [j * scale_y, j * scale_y + flows_grid[j, i, 1] * scale_y],
                     linewidth=1, c=c)


def save_plot_flow(path, img, coords, flows, flow_res=None, return_np_array=False) :
    plot_flow(img, coords, flows, flow_res)

    plt.savefig(path,
                pad_inches=0.)

    if return_np_array :
        return capture_plt_as_np()

    plt.close()


def capture_plt_as_np() :
    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im

def plt_set_up_tight_figure(res, figsize) :
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.axis('off')

    ax.set_xlim([0, res[1]])
    ax.set_ylim([res[0], 0])

def get_np_plot_flow(img, coords, flows, flow_res=None, freq=5, figsize=None) :
    if figsize == None :
        figsize = (img.shape[1] / 16,
                   img.shape[0] / 16)
    # TODO: this is very ghetto
    if len(coords) == img.shape[0] * img.shape[1] :
        plot_flow_grid(img, flows.reshape(img.shape[0], img.shape[1], 2),
                       freq=freq, figsize=figsize)
    else :
        plot_flow(img, coords, flows, flow_res,
                  freq=freq, figsize=figsize)

    return capture_plt_as_np()

def get_np_plot_flow_grid(img, flow_frame, flow_frame_valid=None,
                          freq=5, figsize=None) :
    if figsize == None :
        figsize = (img.shape[1] / 16,
                   img.shape[0] / 16)
    plot_flow_grid(img, flow_frame, freq=freq,
                   figsize=figsize,
                   flows_grid_valid=flow_frame_valid)
    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def background_black_to_white(im) :
    im_hsv = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_RGB2HSV)
    V = im_hsv[..., 2].copy()
    im_hsv[..., 2] = 1 # im_hsv[..., 1]
    im_hsv[..., 1] = V
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB).astype(np.float64)


# TODO: rename this to volume
def create_event_frame_picture(event_frame, on_white=True) :
    # TODO: do we want to move this outside of function
    event_frame = event_frame.transpose([1, 2, 0])
    acc = np.zeros([*event_frame.shape[:2], 3])

    E_up = event_frame.copy()
    E_up[E_up < 0.] = 0.
    E_up_sum = E_up.sum(axis=-1)

    E_down = event_frame.copy()
    E_down[E_down > 0.] = 0.
    E_down_sum = E_down.sum(axis=-1)

    #if not on_white :
    acc[:, :, 0] = E_up_sum
    acc[:, :, 2] = -E_down_sum

    acc_max = acc.max()
    if acc_max == 0 : acc_max = 1
    img = acc / acc_max

    if not on_white :
        img = np.log(img * 20 + 1)
        img = img / img.max()
    else :
        img = np.log(img * 5 + 1)
        img = img / img.max()

    if on_white :
        img = background_black_to_white(img)
    return img


def create_event_picture_format(event_frame, format, res, on_white=True) :
    acc = np.zeros((*res, 3))

    E_up_raw = event_frame[:, format['p']].copy()
    E_up_raw[E_up_raw < 0.]  = 0.
    E_up_sum_rowwise = E_up_raw.sum(axis=-1, keepdims=True)
    E_up_sum = sum_polarity_sparse_var(np.concatenate([event_frame[:, format['xy']],
                                                       np.zeros((len(E_up_raw), 1)),
                                                       E_up_sum_rowwise],
                                                      axis=1))

    E_down_raw = event_frame[:, format['p']].copy()
    E_down_raw[E_down_raw > 0.] = 0.
    E_down_sum_rowwise = E_down_raw.sum(axis=-1, keepdims=True)
    E_down_sum = sum_polarity_sparse_var(np.concatenate([event_frame[:, format['xy']],
                                                         np.zeros((len(E_down_raw), 1)),
                                                         E_down_sum_rowwise],
                                                        axis=1))

    acc[E_up_sum[:, 1].astype(int), E_up_sum[:, 0].astype(int), 0] = E_up_sum[:, 3]
    acc[E_down_sum[:, 1].astype(int), E_down_sum[:, 0].astype(int), 2] = -E_down_sum[:, 3]
    acc_max = acc.max()
    if acc_max == 0 : acc_max = 1
    img = acc / acc_max

    if not on_white:
        img = np.log(img * 20 + 1)
        img = img / img.max()
    else:
        img = np.log(img * 5 + 1)
        img = img / img.max()

    if on_white:
        img = background_black_to_white(img)
    return img



def create_event_picture(event_frame, res=(480, 640), on_white=True) :
    img = np.zeros([*res, 3])

    if event_frame.shape[1] == 4 and res[1] != 4:
        E_up = event_frame[event_frame[:, 3] > 0, :]
        E_down = event_frame[event_frame[:, 3] < 0, :]

        #x_max1 = event_frame[:, 0].max()
        #y_max1 = event_frame[:, 1].max()

        E_up_sum = sum_polarity_sparse_var(E_up)
        E_down_sum = sum_polarity_sparse_var(E_down)
        #x_max = max(E_up_sum[:, 0].max(), E_down_sum[:, 0].max())
        #y_max = max(E_up_sum[:, 1].max(), E_down_sum[:, 1].max())
        img[E_up_sum[:, 1].astype(int), E_up_sum[:, 0].astype(int), 0] = E_up_sum[:, 3]
        img[E_down_sum[:, 1].astype(int), E_down_sum[:, 0].astype(int), 2] = -E_down_sum[:, 3]
        # div by abs max and log
    else :
        event_frame = event_frame.transpose([1, 2, 0])
        E_up = event_frame.copy()
        E_up[E_up < 0.] = 0.
        E_up_sum = E_up.sum(axis=-1)

        E_down = event_frame.copy()
        E_down[E_down > 0.] = 0.
        E_down_sum = E_down.sum(axis=-1)

        img[:, :, 0] = E_up_sum
        img[:, :, 2] = -E_down_sum

    img = img / img.max()
    img = np.log(img * 40 + 1)
    img = img / img.max()
    if on_white :
        im_hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
        v = im_hsv[..., 2].copy()
        im_hsv[..., 2] = 1.
        im_hsv[..., 1] = v
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

    return img


def put_text(im, text) :
    for i in range(len(text)) :
        im = cv2.putText(im, text[i], (2, 20 * (i+1)), cv2.FONT_HERSHEY_TRIPLEX,
                             0.75, (0, 0, 0, 255), 4)
        im = cv2.putText(im, text[i], (2, 20 * (i+1)), cv2.FONT_HERSHEY_TRIPLEX,
                             0.75, (255, 255, 255, 255), 1)
    return im

