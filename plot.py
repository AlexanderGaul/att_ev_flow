import numpy as np

import matplotlib
import matplotlib.pyplot as plt



def plot_flow(img, coords, flows, flow_res=None, freq=10) :
    if flow_res is None  :
        flow_res = img.shape
    scale_x = img.shape[0] / flow_res[0]
    scale_y = img.shape[1] / flow_res[1]

    plt.figure(figsize=(16, 12))

    plt.imshow(img, cmap='gray')

    for i in range(0, len(coords), freq) :
        plt.plot([coords[i, 0] * scale_x, coords[i, 0] * scale_x + flows[i, 0] * scale_x],
                 [coords[i, 1] * scale_y, coords[i, 1] * scale_y + flows[i, 1] * scale_y])


def save_plot_flow(path, img, coords, flows, flow_res = None) :
    plot_flow(img, coords, flows, flow_res)

    plt.savefig(path,
                pad_inches=0.)

    plt.close()


def create_event_picture(event_frame) :
    img = np.zeros([480, 640, 3])
    event_frame = event_frame.astype(int)
    E_up = event_frame[event_frame[:, 3] > 0, :]
    E_down = event_frame[event_frame[:, 3] < 1, :]
    E_up_uniq, E_up_count = np.unique(E_up[:, :2], axis=0, return_counts=True)
    E_down_uniq, E_down_count = np.unique(E_down[:, :2], axis=0, return_counts=True)
    img[E_up_uniq[:, 1], E_up_uniq[:, 0], 2] = E_up_count
    img[E_down_uniq[:, 1], E_down_uniq[:, 0], 0] = E_down_count
    # div by abs max and log
    return np.log(np.log(img + 1) + 1) / np.log(np.log(img + 1) + 1).max()