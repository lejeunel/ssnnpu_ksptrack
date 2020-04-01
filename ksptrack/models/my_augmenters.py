from imgaug import augmenters as iaa
import numpy as np
from imgaug.augmenters import Augmenter
import matplotlib.pyplot as plt
np.random.bit_generator = np.random._bit_generator

class Normalize(Augmenter):
    def __init__(self,
                 mean,
                 std,
                 name=None,
                 deterministic=False,
                 random_state=None):
        super(Normalize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.mean = mean
        self.std = std
        self.n_chans = len(self.mean)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            if (images[i].dtype == np.uint8):
                images[i] = images[i] / 255
            images[i] = [(images[i][..., c] - self.mean[c]) /
                         self.std[c] for c in range(self.n_chans)]
                
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

class Rescale(Augmenter):
    def __init__(self,
                 min_,
                 max_,
                 name=None,
                 deterministic=False,
                 random_state=None):
        super(Rescale, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.min_ = min_
        self.max_ = max_
        self.n_chans = len(self.min_)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in range(nb_images):
            if (images[i].dtype == np.uint8):
                images[i] = images[i] / 255
                
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
            image_aug = 2*(image_aug - 0.5)
        result.append(image_aug)
    return result


void_fun = lambda x, random_state, parents, hooks: x

rescale_augmenter = iaa.Lambda(
    func_images=rescale_images,
    func_segmentation_maps=void_fun,
    func_heatmaps=void_fun,
    func_keypoints=void_fun)


center_augmenter = iaa.Lambda(
    func_images=center_rescaled_images,
    func_heatmaps=void_fun,
    func_keypoints=void_fun)
