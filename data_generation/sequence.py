import numpy as np
import cv2 as cv
import math
import os
from pathlib import Path

import cv2
import numpy

from data_generation.loading import load_mots_object_ims
from data_generation.warp_sequences import ImageWarp, HomographyCurveSequence, MultiImageWarp
from data_generation.hom_generation import random_curves


class ExampleSequence :
    def __init__(self, T, fps_flow, fps_im, num_curves, crop=(2*270, 2*360), output_res=(1*270, 1*360)) :
        im_res = (1080, 1440)

        self.T = T
        self.fps_flow = fps_flow
        self.fps_im = fps_im

        self.num_curves = num_curves

        self.crop = crop
        if self.crop :
            self.crop_offset = ((im_res[0] - crop[0]) // 2, (im_res[1]-crop[1]) // 2)
        else :
            self.crop_offset = None
        self.output_res = output_res

        angle_scale = math.pi * 0.05
        tx_scale = 60
        ty_scale = 30
        sx_scale = 0.0005
        sy_scale = 0.0005
        p1_scale = 0.00005
        p2_scale = 0.00005
        limits_background = (angle_scale, tx_scale, ty_scale,
                             sx_scale, sy_scale,
                             p1_scale, p2_scale, 1., 1.)

        angle_scale = math.pi * 0.05
        tx_scale = 60
        ty_scale = 30
        sx_scale = 0.001
        sy_scale = 0.001
        p1_scale = 0.0001
        p2_scale = 0.0001
        limits_foreground = (angle_scale, tx_scale, ty_scale,
                             sx_scale, sy_scale,
                             p1_scale, p2_scale, .5, 2.)

        dir = Path("/storage/user/gaul/gaul/thesis/data/DSEC_images/zurich_city_01_a_images_rectified_left/")
        images = sorted(os.listdir(dir))
        im = cv.imread(str(dir / images[0]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        warps = []
        warps.append(ImageWarp(im,
                               HomographyCurveSequence(*random_curves(self.T, self.num_curves,
                                                                      limits_background, 0.2),
                                                       center=(im.shape[1]//2, im.shape[0]//2)),
                               crop_offset=self.crop_offset, crop=self.crop, res=output_res
                               ))

        ims_objects = load_mots_object_ims(1, 0, min_size=25)
        for im_cut in ims_objects :
            im_pad = np.zeros((*im.shape[:2], 4))

            im_pad[(im.shape[0] // 2):(im.shape[0] // 2 + im_cut.shape[0]),
            (im.shape[1] // 2):(im.shape[1] // 2 + im_cut.shape[1])] = im_cut

            offset = (int(np.random.uniform(-im.shape[0] * 0.2, im.shape[0] * 0.2)),
                      int(np.random.uniform(-im.shape[1] * 0.2, im.shape[1] * 0.2)))

            warps.append(ImageWarp(im_pad,
                                   HomographyCurveSequence(*random_curves(self.T, self.num_curves,
                                                                          limits_foreground, 0.2),
                                                           center=(im_pad.shape[1]//2, im_pad.shape[0]//2),
                                                           offset=offset),
                                   crop_offset=self.crop_offset, crop=self.crop, res=output_res))


        self.ims_warp = MultiImageWarp(warps)


    def write_video(self, path) :
        vid = cv.VideoWriter(str(path) + "/vid.avi", 0, self.fps_im, (self.output_res[1], self.output_res[0]))
        ims_ts = np.linspace(0., self.T, self.T * self.fps_im + 1)
        np.savetxt(str(path) + "/timestamps.txt", ims_ts)
        for i, t in enumerate(ims_ts) :
            print(t)
            im = self.ims_warp.get_image(t)
            top_left = (im.shape[0] // 2 - self.crop[0] // 2,
                        im.shape[1] // 2 - self.crop[1] // 2)
            im_crop = im[top_left[0]:top_left[0]+self.crop[0],
                         top_left[1]:top_left[1]+self.crop[1], :]
            im_resize = cv.resize(im_crop, (self.output_res[1], self.output_res[0]))
            cv.imwrite(str(path) + "/ims/" + "0" * (6 - len(str(i))) + str(i) + ".png",
                       cv.cvtColor(im_resize, cv.COLOR_RGB2BGR))

            vid.write(cv.cvtColor(im_resize, cv.COLOR_RGB2BGR))
        vid.release()