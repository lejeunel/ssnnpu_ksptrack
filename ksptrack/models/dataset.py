import os
from .my_augmenters import rescale_augmenter
from . import im_utils as utls
import torch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from collections import namedtuple
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(
            self,
            im_paths,
            in_shape=None,
            truth_paths=None,
            locs2d=None,
            sig_prior=0.1,
            augmentations=None,
            seed=0):

        self.im_paths = im_paths
        self.augmentations = augmentations

        self.truth_paths = truth_paths
        self.locs2d = locs2d
        self.sig_prior = sig_prior
        self.in_shape = in_shape
        self.seed = 0

        self.normalize = rescale_augmenter

        if(isinstance(self.in_shape, int)):
            self.in_shape = (self.in_shape, self.in_shape)
            
    def add_imgs(self, im_paths):
        self.im_paths += im_paths

    def add_truths(self, truth_paths):
        self.truth_paths += truth_paths

    def __len__(self):
        return len(self.im_paths)

    def sample_uniform(self):

        random_sample_idx = np.random.choice(
            np.arange(len(self.im_paths)))

        return self.__getitem__(random_sample_idx)

    def __getitem__(self, idx):

        im_path = self.im_paths[idx]

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        im = utls.imread(im_path, scale=False)

        if(self.in_shape is None):
            self.in_shape = im.shape[:2]

        im_orig = im.copy()
        im_shape = im.shape[0:2]

        truth = None
        obj_prior = None

        if(self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            truth = utls.imread(truth_path, scale=False)[..., :1]
            truth = truth > 0

            if (len(truth.shape) == 3):
                truth = np.mean(truth, axis=-1)

            truth = truth.astype(np.uint8)
        else:
            truth = np.zeros(im.shape[:2]).astype(np.uint8)

        # Apply data augmentation
        if(self.augmentations is not None):
            aug_det = self.augmentations.to_deterministic()
        else:
            aug_det = iaa.Noop()

        if(self.locs2d is not None):
            locs = self.locs2d[self.locs2d['frame'] == idx]
            locs = [utls.coord2Pixel(l['x'], l['y'], self.in_shape[1], self.in_shape[0])
                    for _, l in locs.iterrows()]

            keypoints = ia.KeypointsOnImage(
                [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
                shape=(self.in_shape[0], self.in_shape[1]))
            keypoints = aug_det.augment_keypoints([keypoints])[0]

            if (len(locs) > 0):
                obj_prior = [
                    utls.make_2d_gauss((self.in_shape[0], self.in_shape[1]),
                                       self.sig_prior * max(self.in_shape),
                                       (kp.y, kp.x)) for kp in keypoints.keypoints
                ]
                obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
                obj_prior += obj_prior.min()
                obj_prior /= obj_prior.max()
            else:
                obj_prior = (
                    np.ones((self.in_shape[0], self.in_shape[1])))[..., None]
        else:
            obj_prior = np.zeros((self.in_shape[0], self.in_shape[1]))[..., None]


        im = aug_det.augment_images([im])[0]

        truth = ia.SegmentationMapsOnImage(truth,
                                          shape=truth.shape)
        truth = aug_det.augment_segmentation_maps(
            [truth])[0].get_arr()[..., np.newaxis]

        return {'image': im,
                'prior': obj_prior,
                'frame_name': os.path.split(self.im_paths[idx])[-1],
                'label/segmentation': truth,
                'original image': im_orig}

    @staticmethod
    def collate_fn(data):
        
        im = [np.rollaxis(d['image'], -1) for d in data]
        im = torch.stack([torch.from_numpy(i).float()
                          for i in im])

        obj_prior = [np.rollaxis(d['prior'], -1) for d in data]
        obj_prior = torch.stack([torch.from_numpy(i).float()
                                 for i in obj_prior])

        truth = [np.rollaxis(d['label/segmentation'], -1) for d in data]
        truth = torch.stack([torch.from_numpy(i) for i in truth]).float()

        im_orig = [np.rollaxis(d['original image'], -1) for d in data]

        return{'image': im,
               'prior': obj_prior,
               'label/segmentation': truth,
               'original image': im_orig}
