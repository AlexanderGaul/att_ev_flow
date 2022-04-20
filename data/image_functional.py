import numpy as np

from data.volume_functional import unfold
from data.volume_modules import Volume2PatchArray


# TODO: move to volume functional

# TODO: move to volume functional


class ImagePair2PatchArray :
    def __init__(self, patch_size=3, stride=1,
                 normalization_type='uni') :
        self.volume_2_patch_array = Volume2PatchArray(patch_size,
                                                      stride,
                                                      format='np')

        self.normalization_type = normalization_type

        self.im_mean_rgb = np.array((0.485, 0.456, 0.406))
        self.im_std_rgb = np.array((0.229, 0.224, 0.225))

    def unfold_image(self, im) :
        return unfold(im, self.patch_size, self.stride)

    def __call__(self, data) :
        im1 = data['im1']
        im2 = data['im2']
        if im1.dtype == np.int :
            if self.normalization_type == 'uni' :
                im1 = (im1.astype(np.float32) / 255 * 2 - 1)
                im2 = (im2.astype(np.float32) / 255 * 2 - 1)
            else :
                im1 = (im1.astype(np.float32) - self.im_mean_rgb*255) / (self.im_std_rgb*255)
                im2 = (im2.astype(np.float32) - self.im_mean_rgb*255) / (self.im_std_rgb * 255)
        else :
            if self.normalization_type == 'uni':
                im1 = im1 * 2 - 1
                im2 = im2 * 2 - 1
            else :
                im1 = (im1 - self.im_mean_rgb) / self.im_std_rgb
                im2 = (im2 - self.im_mean_rgb) / self.im_std_rgb
        imstack = np.concatenate([im1, im2], axis=-1)
        data['volume'] = imstack

        return self.volume_2_patch_array(data)
        """
        imstack_unfold = np.concatenate([
            self.unfold_image(im1),
            self.unfold_image(im2)], axis=2)
        H_unfold, W_unfold = imstack_unfold.shape[:2]

        imstack_patchflat = imstack_unfold.reshape(H_unfold, W_unfold, -1)
        patch_array = imstack_patchflat.reshape(H_unfold*W_unfold, -1)

        coords = get_grid_coordinates((W_unfold, H_unfold), (self.patch_size / 2 - 0.5,
                                                             self.patch_size / 2 - 0.5))
        patch_array = np.concatenate([coords, patch_array], axis=1)

        dict_out = {'patch_array' : patch_array,
                    'res' : (im1.shape[1], im1.shape[0]),
                    **data}
        return dict_out
        """