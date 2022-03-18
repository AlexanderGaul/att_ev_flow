import matplotlib.pyplot as plt
import numpy as np

import imageio
import cv2 as cv
import os

from datasets import *
from warp_sequences import *
from image_utils import warp_backward
from plot import plot_flow_grid

from generate_events import EventGenerator


class SequenceGenerator :
    def __init__(self, homseq_gen_background, homseq_gen_foreground,
                 crop_offset=(0, 0), crop=None, res=None,
                 min_objects=2, max_objects=6) :
        self.background_data = DSECBackground()
        self.foreground_data = MOTSForeground()

        self.homseq_gen_background = homseq_gen_background
        self.homseq_gen_foreground = homseq_gen_foreground

        self.crop_offset = crop_offset
        self.crop = crop
        self.res = res

        self.min_objects=min_objects
        self.max_objects=max_objects


    def __call__(self, seed=None) :
        if seed is not None :
            np.random.seed(seed)

        im = self.background_data.get_random_item()
        center=(im.shape[1]//2, im.shape[0]//2)

        if self.min_objects != self.max_objects :
            num_foreground = np.random.randint(self.min_objects,
                                               self.max_objects)
        else :
            num_foreground = 0
        print(num_foreground)
        foregrounds = []
        for i in range(num_foreground) :
            foregrounds.append(self.foreground_data.get_random_item())


        offset = (np.random.uniform(-im.shape[1] // 4, im.shape[1] // 4),
                  np.random.uniform(-im.shape[0] // 4, im.shape[0] // 4))

        warps = [ImageWarp(im,
                           self.homseq_gen_background(center=center,
                                                      offset=offset),
                           self.crop_offset, self.crop, self.res)]

        for obj in foregrounds :
            # TODO: pad object
            obj_pad = np.zeros((*im.shape[:2], obj.shape[2]))
            upper_left = ((im.shape[0] - obj.shape[0]) // 2,
                          (im.shape[1] - obj.shape[1]) // 2)
            obj_pad[upper_left[0]:(upper_left[0] + obj.shape[0]),
                    upper_left[1]:(upper_left[1] + obj.shape[1])] = obj

            if self.crop :
                offset = (np.random.uniform(-self.crop[1] // 2,
                                            self.crop[1] // 2),
                          np.random.uniform(-self.crop[0] // 2,
                                            self.crop[0] // 2))
            else :
                offset = (np.random.uniform(-im.shape[1] // 2, im.shape[1] // 2),
                          np.random.uniform(-im.shape[0] // 2, im.shape[0] // 2))

            warps.append(ImageWarp(obj_pad,
                                   self.homseq_gen_foreground(center=center,
                                                              offset=offset),
                                   self.crop_offset, self.crop, self.res))

        multi_warp = MultiImageWarp(warps)
        return multi_warp



class SequenceWriter :
    def __init__(self, T, im_fps, flow_fps,
                 event_file, flow_dir, im_dir, flow_ts_file, im_ts_file) :
        self.T = T
        self.im_fps = im_fps
        self.flow_fps = flow_fps
        self.event_file = event_file
        self.im_dir = im_dir
        self.flow_dir = flow_dir
        self.flow_ts_file = flow_ts_file
        self.im_ts_file = im_ts_file

        self.event_writer = EventGenerator()

    @staticmethod
    def write_flow(flow:np.ndarray, path) :
        h, w, _ = flow.shape
        flow_map = np.rint(flow * 128 + 2 ** 15)
        flow_map = flow_map.astype(np.uint16) #.transpose(1, 2, 0)
        flow_map = np.concatenate((flow_map, np.ones((h, w, 1), dtype=np.uint16)), axis=-1)

        imageio.imwrite(path, flow_map, format='PNG-FI')

    def write_sequence(self, seq:MultiImageWarp, dir:Path, write_video=False) :
        # [] create folders
        # [] what happens if folder already exists
        if not os.path.exists(dir) : os.makedirs(dir)
        if not os.path.exists(dir / self.im_dir) : os.makedirs(dir / self.im_dir)
        if not os.path.exists(dir / self.flow_dir) : os.makedirs(dir / self.flow_dir)
        if not os.path.exists(os.path.dirname(dir / self.event_file)) :
            os.makedirs(os.path.dirname(dir / self.event_file))
        if not os.path.exists(os.path.dirname(dir / self.flow_ts_file)) :
            os.makedirs(os.path.dirname(dir / self.flow_ts_file))
        if not os.path.exists(os.path.dirname(dir / self.im_ts_file)):
            os.makedirs(os.path.dirname(dir / self.im_ts_file))

        im_ts = np.linspace(0, self.T, int(self.T * self.im_fps + 1))
        np.savetxt(dir / self.im_ts_file, im_ts)
        # [x] write util function to convert flows to appropriate format
        # [] write flows to png file
        flow_ts_begin = np.linspace(0, self.T, int(self.T * self.flow_fps + 1))
        flow_ts = np.stack([flow_ts_begin[:-1] * 1e6, flow_ts_begin[1:] * 1e6]).transpose().astype(int)
        np.savetxt(dir / self.flow_ts_file, flow_ts)

        """
        # TODO:
        im = seq.get_image(0)
        plt.imshow(im)
        plt.show()
        im2 = seq.get_image(1.)
        plt.imshow(im2)
        plt.show()
        fl = seq.get_flow(0., 1.)
        coords = get_grid_coordinates((fl.shape[1], fl.shape[0]))
        im_warp = warp_backward(coords, fl.reshape(-1, 2),
                                im2)
        plt.imshow(im_warp)
        plt.show()

        plot_flow_grid(im, fl, freq=20)
        plt.show()

        return
        """
        # [] write images to disc
        for i, t in enumerate(im_ts) :
            im = seq.get_image(t)
            imageio.imwrite(dir / self.im_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"), im)

        # [] create events
        events = self.event_writer.generate_events(dir / self.im_dir,
                                                   dir / self.im_ts_file)
        self.event_writer.write_events(events, dir / self.event_file)


        # [] delete at least some of the images
        # [] keep images that correspond to flow timestamps??
        # [] how to delete files in the first place

        counter = 1. -  self.flow_fps / self.im_fps
        for i, t in enumerate(im_ts) :
            counter += self.flow_fps / self.im_fps
            if counter >= 1. :
                counter -= 1.
                continue
            os.remove(dir / self.im_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"))

        if write_video :
            ims = sorted(os.listdir(dir / self.im_dir))
            vid_writer = cv.VideoWriter(str(dir / "vid.avi"), 0, self.flow_fps,
                                        (seq.warps[0].res[1], seq.warps[0].res[0]))
            for im_name in ims :
                im = cv.imread(str(dir / self.im_dir / im_name))
                vid_writer.write(im)
            vid_writer.release()


        for i, t in enumerate(flow_ts_begin[:-1]) :
            fl = seq.get_flow(flow_ts_begin[i], flow_ts_begin[i+1])
            SequenceWriter.write_flow(fl, dir / self.flow_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"))

