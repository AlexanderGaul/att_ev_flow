import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

import io

from events import sum_polarity_sparse_var

def plot_flow_color(coords, flows:np.ndarray, res=None) :
    coords = coords.astype(int)
    # TODO: maybe try this with scatter
    im_hsv = np.zeros([*np.flip(res), 3]) # shape: H, W, C

    im_hsv[coords[:, [1]], coords[:, [0]], 2], im_hsv[coords[:, [1]], coords[:, [0]], 0] = \
        cv2.cartToPolar(flows[:, 0], flows[:, 1], None, None, True)

    im_hsv[:, :, 0] /= 2.
    im_hsv[:, :, 1] = 255.
    im_hsv[:, :, 2] *= (255. / im_hsv[:, :, 2].max())
    im_hsv[:, :, 2] = np.log(im_hsv[:, :, 2] + 1)
    im_hsv[:, :, 2] *= (255. / im_hsv[:, :, 2].max())

    im_hsv = im_hsv.astype(int).astype(np.ubyte)
    im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

    # TODO: refactor pyplot image buffer reading
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    ax.set_xlim([0, im_rgb.shape[1]])
    ax.set_ylim([im_rgb.shape[0], 0])

    im_rgb_log = np.log(im_rgb.astype(float)+1)
    im_rgb_log /= im_rgb_log.max()
    plt.imshow(im_rgb)

    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im

    """
    Mat
    flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    // build hsv image 
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    """

def plot_flow_error(coords, pred, flows, res=None) :
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

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    ax.set_xlim([0, im_rgb.shape[1]])
    ax.set_ylim([im_rgb.shape[0], 0])

    #im_rgb_log = np.log(im_rgb.astype(float) + 1)
    #im_rgb_log /= im_rgb_log.max()
    plt.imshow(im_rgb)

    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def plot_flow(img, coords, flows, flow_res=None, freq=10) :
    if flow_res is None  :
        flow_res = img.shape
    scale_x = img.shape[0] / flow_res[0]
    scale_y = img.shape[1] / flow_res[1]

    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    ax.axis('tight')
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([img.shape[0], 0])

    plt.imshow(img, cmap='gray')
    plt.scatter(coords[::freq, 0], coords[::freq, 1], s=6, c='g')
    for i in range(0, len(coords), freq) :
        plt.plot([coords[i, 0] * scale_x, coords[i, 0] * scale_x + flows[i, 0] * scale_x],
                 [coords[i, 1] * scale_y, coords[i, 1] * scale_y + flows[i, 1] * scale_y])


def plot_flow_grid(img, flows_grid, flow_res=None, freq=10) :
    if flow_res is None  :
        flow_res = img.shape
    scale_x = img.shape[0] / flow_res[0]
    scale_y = img.shape[1] / flow_res[1]

    # plt.figure(figsize=(16, 12))

    plt.imshow(img, cmap='gray')
    # TODO: convert this
    #plt.scatter(coords[::freq, 0], coords[::freq, 1], s=6, c='g')
    for i in range(0, flows_grid.shape[1], freq) :
        for j in range(0, flows_grid.shape[0], freq) :
            plt.plot([i * scale_x, i * scale_x + flows_grid[j, i, 0] * scale_x],
                     [j * scale_y, j * scale_y + flows_grid[j, i, 1] * scale_y])


def save_plot_flow(path, img, coords, flows, flow_res=None, return_np_array=False) :
    plot_flow(img, coords, flows, flow_res)

    plt.savefig(path,
                pad_inches=0.)

    if return_np_array :
        with io.BytesIO() as buff :
            plt.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = plt.gcf().canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        plt.close()
        return im

    plt.close()


def get_np_plot_flow(img, coords, flows, flow_res=None) :
    plot_flow(img, coords, flows, flow_res)

    with io.BytesIO() as buff:
        plt.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = plt.gcf().canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def create_event_picture(event_frame, res=(480, 640)) :
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


    return np.log(np.log(img + 1) + 1) / np.log(np.log(img + 1) + 1).max()