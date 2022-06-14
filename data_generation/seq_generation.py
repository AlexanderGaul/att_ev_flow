import numpy.random

from data_generation.datasets import *
from data_generation.warp_sequences import *


#from plot import plot_flow_grid


class MovingEdgeGenerator :
    def __init__(self, homseq_gen) :
        self.homseq_gen = homseq_gen

    def __call__(self, seed=None) :
        if seed is not None :
            np.random.seed(seed)
        edge_image = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = (64, 64)
        crop_offset = ((edge_image.shape[0] - crop[0]) // 2,
                       (edge_image.shape[1] - crop[1]) // 2)

        edge_image[:] = 128
        edge_image[:, :crop_offset[1] + 1] -= 32
        edge_image[:, crop_offset[1] + 1:] += 32
        center = (edge_image.shape[1] // 2,
                  edge_image.shape[0] // 2)


        rotation = np.random.uniform(0, 2 * np.pi)

        warps = [ImageWarp(edge_image,
                           self.homseq_gen(center=center,
                                           offset=(0, 0),
                                           rot=rotation),
                           crop_offset=crop_offset , crop=crop, res=crop)]
        return MultiImageWarp(warps)


class SequenceGeneratorPolished :
    def __init__(self,
                 data_background,
                 data_foreground,
                 homseq_gen_background,
                 homseq_gen_midground,
                 homseq_gen_foreground,
                 crop,
                 res,
                 min_middleground, max_middleground,
                 min_foreground, max_foreground) :
        self.data_background = data_background
        self.data_foreground  = data_foreground
        self.homseq_gen_background = homseq_gen_background
        self.homseq_gen_midground  = homseq_gen_midground
        self.homseq_gen_foreground = homseq_gen_foreground

        self.crop = crop
        self.res = res

        self.min_mg = min_middleground
        self.max_mg = max_middleground

        self.min_fg = min_foreground
        self.max_fg = max_foreground

        self.max_obj_area = (self.crop[0] * self.crop[1]) / 1.5
        self.min_obj_area = 16
        self.softmax_obj_area = (self.crop[0] * self.crop[1]) / 3
        self.softmin_obj_area = 100
        self.softmin_obj_dim = 100

    def __call__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        im, im_id = self.data_background.random_item()
        if im.dtype == np.uint8 :
            im = im.astype(np.float32) / 255.

        center = (im.shape[1]//2, im.shape[0]//2)
        crop_offset = (im.shape[0] - self.crop[0]) // 2, (im.shape[1] - self.crop[1]) // 2

        if self.min_fg != self.max_fg :
            num_fg = np.random.randint(self.min_fg,
                                               self.max_fg)
        else :
            num_fg = self.min_fg

        if self.min_mg != self.max_mg :
            num_mg = np.random.randint(self.min_mg,
                                       self.max_mg)
        else :
            num_mg = self.min_mg

        num_obj = num_mg + num_fg

        hsq_bg = self.homseq_gen_background(center=center)
        warps = [ImageWarp(im,
                 hsq_bg, crop_offset, self.crop, self.res, border_reflect=True)]

        # TODO: craete background warp

        objects, obj_sizes = self.get_objects(num_obj, res = im.shape[:2])
        scales = [self.get_scaling(obj_im) for obj_im in objects]
        objects = [self.pad_object(obj_im, res = im.shape[:2]) for obj_im in objects]

        count_fg = 0
        count_mg = 0
        # TODO: how to get size of objects??
        for i in range(num_obj) :
            if np.random.binomial(1, (num_mg - count_mg) / (num_obj - i)) > 0.5 :
                count_mg += 1
                hsq = self.homseq_gen_midground(hsq_bg, center, *self.get_hom_init_midground(center, scales[i]))

            # random mid or foreground
            else :
                count_fg += 1
                hsq = self.homseq_gen_foreground(center , *self.get_hom_init_foreground(center, scales[i]), scale=self.crop[1]/max(obj_sizes[i]))

            warps.append(ImageWarp(objects[i],
                                   hsq,
                                   crop_offset, self.crop, self.res, border_reflect=False))

        return MultiImageWarp(warps)



    def get_hom_init_midground(self, center, scale=1.) :
        H_0 = H_euclidean(np.random.uniform(-math.pi / 4, math.pi / 4),
                          np.random.uniform(-int(self.crop[1] / 2 * 0.8),
                                            int(self.crop[1] / 2 * 0.8)),
                          np.random.uniform(-int(self.crop[0] / 2 * 0.8),
                                            int(self.crop[0] / 2 * 0.8)))
        H_1 = H_scale(0.8 * scale)
        # TODO: add downscaling
        return H_centered(H_0, center), H_centered(H_1, center)

    def get_hom_init_foreground(self, center, scale=1.) :
        h_0 = h_param_id()
        h_0[2] = np.random.uniform(-math.pi / 4, math.pi / 4)
        H_0 = np.eye(3)
        H_1 = H_euclidean(0,
                          np.random.uniform(-int(self.crop[1] / 2 * 1.0),
                                            int(self.crop[1] / 2 * 1.0)),
                          np.random.uniform(-int(self.crop[0] / 2 * 1.0),
                                            int(self.crop[0] / 2 * 1.0) // 2)).dot(
            H_scale(0.8 * scale))
        # TODO add downscaling
        return h_0, H_0, H_centered(H_1, center)

    def get_objects(self, num, res) :
        sizes = []
        objs = []
        while len(objs) < num :
            obj_im, obj_id = self.data_foreground.random_item()
            # TODO: discard or scale based on size
            mass = (obj_im[..., 3] > 0).sum()
            dim = max(obj_im.shape[:2])

            print(obj_im[obj_im[..., 3] > 0, :3].mean())
            if obj_im[obj_im[..., 3] > 0, :3].mean() > 240 :
                plt.imshow(obj_im)
                plt.show()
                continue
                """
                if mass > self.max_obj_area :
                    scale = np.random.uniform(0.5, 1)
                    obj_im = cv.resize(obj_im,
                                       (int(obj_im.shape[1] * scale),
                                        int(obj_im.shape[0] * scale)),
                                       interpolation=cv.INTER_CUBIC)
                    obj_im[..., 3] = np.around(obj_im[..., 3])
                    obj_im = np.maximum(obj_im, 0)
                    obj_im = np.minimum(obj_im, 255)
                elif mass > self.softmax_obj_area :
                    if numpy.random.binomial(1, 0.5, 1) < 0.5 :
                        # scale = np.random.uniform(self.scale_min * scale_over, scale_over)
                        scale = np.random.uniform(0.5, 1)
                        obj_im = cv.resize(obj_im,
                                           (int(obj_im.shape[1] * scale),
                                            int(obj_im.shape[0] * scale)),
                                           interpolation=cv.INTER_CUBIC)
                        obj_im[..., 3] = np.around(obj_im[..., 3])
                        obj_im = np.maximum(obj_im, 0)
                        obj_im = np.minimum(obj_im, 255)
                """
            elif mass < self.min_obj_area :
                continue
            elif dim < self.softmin_obj_dim :
                if numpy.random.binomial(1, 0.5, 1) > 0.5 :
                    continue
            elif mass < self.softmin_obj_area :
                if numpy.random.binomial(1, 0.8, 1) > 0.5 :
                    # do not scale up because of artefacts
                    continue
            sizes.append(obj_im.shape[:2])

            # TODO: remove
            obj_pad = np.zeros((*res, obj_im.shape[2]), dtype=obj_im.dtype)
            upper_left = ((res[0] - obj_im.shape[0]) // 2,
                          (res[1] - obj_im.shape[1]) // 2)
            obj_pad[upper_left[0]:(upper_left[0] + obj_im.shape[0]),
                    upper_left[1]:(upper_left[1] + obj_im.shape[1])] = obj_im

            if obj_pad.dtype == np.uint8:
                obj_pad = obj_pad.astype(np.float32) / 255.

            objs.append(obj_pad)

        return objs, sizes

    # TODO: move padding out of get_objects
    def get_scaling(self, obj_im) :
        mass = (obj_im[..., 3] > 0).sum()
        dim = max(obj_im.shape[:2])

        if mass > self.max_obj_area:
            scale = np.random.uniform(0.5, 1)
        elif mass > self.softmax_obj_area:
            if numpy.random.binomial(1, 0.5, 1) < 0.5:
                # scale = np.random.uniform(self.scale_min * scale_over, scale_over)
                scale = np.random.uniform(0.5, 1)
            else :
                scale = 1
        else :
            scale = 1
        return scale


    # TODO: move into utils
    def scale_object(self, obj_im, scale):
        obj_im = cv.resize(obj_im,
                           (int(obj_im.shape[1] * scale),
                            int(obj_im.shape[0] * scale)),
                           interpolation=cv.INTER_CUBIC)
        obj_im[..., 3] = np.around(obj_im[..., 3])
        obj_im = np.maximum(obj_im, 0)
        obj_im = np.minimum(obj_im, 255)
        return obj_im

    # TODO: move into utils
    def pad_object(self, obj_im, res):
        obj_pad = np.zeros((*res, obj_im.shape[2]), dtype=obj_im.dtype)
        upper_left = ((res[0] - obj_im.shape[0]) // 2,
                      (res[1] - obj_im.shape[1]) // 2)
        obj_pad[upper_left[0]:(upper_left[0] + obj_im.shape[0]),
                upper_left[1]:(upper_left[1] + obj_im.shape[1])] = obj_im

        return obj_pad



class SequenceGenerator :
    def __init__(self, homseq_gen_background, homseq_gen_foreground,
                 crop_offset=(0, 0), crop=None, res=None,
                 min_objects=2, max_objects=6,
                 blur_image=False, random_offset=True) :
        self.background_data = DSECBackground()
        self.foreground_data = MOTSForeground()

        self.homseq_gen_background = homseq_gen_background
        self.homseq_gen_foreground = homseq_gen_foreground

        self.crop_offset = crop_offset
        self.crop = crop
        self.res = res

        self.min_objects=min_objects
        self.max_objects=max_objects

        self.blur_image = blur_image

        self.random_offset = random_offset

        self.scale_foreground = True

        self.size_min_threshold = 160
        self.size_max_threshold = 360

        self.scale_probability = 0.8

        self.scale_max = 2.
        self.scale_min = 1.



    def __call__(self, seed=None) :
        if seed is not None :
            np.random.seed(seed)

        im = self.background_data.get_random_item()
        if im.dtype == np.uint8 :
            im = im.astype(np.float32) / 255.
        if self.blur_image :
            im = cv.GaussianBlur(im, (5, 5), 0)


        center=(im.shape[1]//2, im.shape[0]//2)

        if self.min_objects != self.max_objects :
            num_foreground = np.random.randint(self.min_objects,
                                               self.max_objects)
        else :
            num_foreground = self.min_objects

        foregrounds = []
        for i in range(num_foreground) :
            im_obj = self.foreground_data.get_random_item()
            if not self.blur_image and self.scale_foreground :
                im_obj = blur_with_alpha(im_obj, ksize_im=1, sigma_im=0,
                                         ksize_outline=9)
            if im_obj.dtype == np.uint8 :
                im_obj = im_obj.astype(np.float32) / 255.
            if self.blur_image :
                im_obj = blur_with_alpha(im_obj, ksize_im=5, sigma_im=0,
                                         ksize_outline=9 if self.scale_foreground else None)
            print(im_obj.shape)

            if self.scale_foreground :
                max_shape = max(im_obj.shape)
                min_shape = min(im_obj.shape)
                if max_shape < self.size_min_threshold and \
                        np.random.binomial(1, self.scale_probability) :
                    scale_over = self.size_min_threshold / min_shape
                    #scale = np.random.uniform(scale_over, self.scale_max * scale_over)
                    scale = np.random.uniform(2., self.scale_max)
                    im_obj = cv.resize(im_obj,
                              (int(im_obj.shape[1] * scale),
                               int(im_obj.shape[0] * scale)),
                              interpolation=cv.INTER_CUBIC)
                    im_obj[..., 3] = np.around(im_obj[..., 3])
                    im_obj = np.maximum(im_obj, 0.)
                    im_obj = np.minimum(im_obj, 1.)
                elif min_shape > self.size_max_threshold and \
                        np.random.binomial(1, self.scale_probability) :
                    scale_over = self.size_max_threshold / min_shape
                    #scale = np.random.uniform(self.scale_min * scale_over, scale_over)
                    scale = np.random.uniform(self.scale_min, 1)
                    im_obj = cv.resize(im_obj,
                              (int(im_obj.shape[1] * scale),
                               int(im_obj.shape[0] * scale)),
                              interpolation=cv.INTER_CUBIC)
                    im_obj[..., 3] = np.around(im_obj[..., 3])
                    im_obj = np.maximum(im_obj, 0.)
                    im_obj = np.minimum(im_obj, 1.)



            foregrounds.append(im_obj)

        if self.random_offset :
            offset_f = (np.random.uniform(-im.shape[1] // 4, im.shape[1] // 4),
                      np.random.uniform(-im.shape[0] // 4, im.shape[0] // 4))
            center_offset = (np.random.uniform(-self.crop[0] // 2, self.crop[0] //2),
                             np.random.uniform(-self.crop[1] // 2, self.crop[1] // 2))
            center_offset = (center[0] + center_offset[0] - offset_f[0],
                             center[1] + center_offset[1] - offset_f[1])
        else :
            offset_f = (0, 0)
            center_offset = (0, 0)

        warps = [ImageWarp(im,
                           self.homseq_gen_background(center=center_offset,
                                                      offset=offset_f),
                           self.crop_offset, self.crop, self.res,
                           border_reflect=True)]

        for obj in foregrounds :
            # TODO: pad object
            obj_pad = np.zeros((*im.shape[:2], obj.shape[2]), dtype=obj.dtype)
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
            #offset = (0, 0)
            #offset = (offset[0] + offset_f[0],
            #          offset[1] + offset_f[1])

            warps.append(ImageWarp(obj_pad,
                                   self.homseq_gen_foreground(center=center,
                                                              offset=offset),
                                   self.crop_offset, self.crop, self.res,
                                   border_reflect=False))

        multi_warp = MultiImageWarp(warps)
        return multi_warp



