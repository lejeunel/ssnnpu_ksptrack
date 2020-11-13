import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
from skimage import exposure

if hasattr(np.random, "_bit_generator"):
    np.random.bit_generator = np.random._bit_generator


class Normalize(Augmenter):
    def __init__(self, mean, std, name=None, random_state=None):
        super(Normalize, self).__init__(name=name, random_state=random_state)
        self.mean = mean
        self.std = std
        self.n_chans = len(self.mean)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            if (images[i].dtype == np.uint8):
                images[i] = images[i] / 255
            images[i] = [(images[i][..., c] - self.mean[c]) / self.std[c]
                         for c in range(self.n_chans)]

            images[i] = np.moveaxis(np.array(images[i]), 0, -1)
            images[i] = images[i].astype(float)
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.mean, self.std]


class ContrastStretching(Augmenter):
    def __init__(self, plow, phigh, name=None, random_state=None):
        super(ContrastStretching, self).__init__(name=name,
                                                 random_state=random_state)
        self.plow = plow
        self.phigh = phigh
        self.n_chans = len(self.plow)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            assert images[i].dtype == np.uint8, 'dtype must be uint8'
            im = images[i]
            im_new = np.zeros_like(im)
            for c in range(im.shape[-1]):
                im_new[..., c] = exposure.rescale_intensity(
                    im[..., c], in_range=(self.plow[c], self.phigh[c]))
            images[i] = im_new
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.bin_centers, self.cdfs]


class AdaptHistEqualize(Augmenter):
    def __init__(self, clip_limit=0.01, name=None, random_state=None):
        super(AdaptHistEqualize, self).__init__(name=name,
                                                random_state=random_state)
        self.clip_limit = clip_limit

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            assert images[i].dtype == np.uint8, 'dtype must be uint8'
            im = images[i]
            im_new = exposure.equalize_adapthist(im,
                                                 clip_limit=self.clip_limit)
            images[i] = im_new
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.bin_centers, self.cdfs]


class HistEqualize(Augmenter):
    def __init__(self, cdfs, bin_centers, name=None, random_state=None):
        super(HistEqualize, self).__init__(name=name,
                                           random_state=random_state)
        self.cdfs = cdfs
        self.bin_centers = bin_centers
        self.n_chans = len(self.cdfs)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            assert images[i].dtype == np.uint8, 'dtype must be uint8'
            im = images[i]
            im_new = np.zeros(im.shape)
            shape = im.shape[:2]
            for c in range(im.shape[-1]):
                im_new[..., c] = np.interp(im[...,
                                              c].flat, self.bin_centers[c],
                                           self.cdfs[c]).reshape(shape)
            images[i] = im_new
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.bin_centers, self.cdfs]


class Rescale(Augmenter):
    def __init__(self,
                 min_=[0., 0., 0.],
                 max_=[1., 1., 1.],
                 name=None,
                 random_state=None):
        super(Rescale, self).__init__(name=name, random_state=random_state)
        self.min_ = min_
        self.max_ = max_

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            if (images[i].dtype == np.uint8):
                images[i] = images[i] / 255

            n_chans = 1 if images[i].ndim == 2 else images[i].shape[-1]
            images[i] = [(images[i][..., c] - self.min_[c]) /
                         (self.max_[c] - self.min_[c]) for c in range(n_chans)]

            images[i] = np.moveaxis(np.array(images[i]), 0, -1)
            images[i] = images[i].astype(float)
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def get_parameters(self):
        return [self.min_, self.max_]


def rescale_images(images, random_state, parents, hooks):

    result = []
    for image in images:
        image_aug = np.copy(image)
        if (image.dtype == np.uint8):
            image_aug = image_aug / 255
        result.append(image_aug)
    return result


def center_rescaled_images(images, random_state, parents, hooks):

    result = []
    for image in images:
        image_aug = np.copy(image)
        if (image.dtype == np.uint8):
            image_aug = image_aug - 128
        else:
            image_aug = 2 * (image_aug - 0.5)
        result.append(image_aug)
    return result


void_fun = lambda x, random_state, parents, hooks: x

rescale_augmenter = iaa.Lambda(func_images=rescale_images,
                               func_segmentation_maps=void_fun,
                               func_heatmaps=void_fun,
                               func_keypoints=void_fun)

center_augmenter = iaa.Lambda(func_images=center_rescaled_images,
                              func_heatmaps=void_fun,
                              func_keypoints=void_fun)
