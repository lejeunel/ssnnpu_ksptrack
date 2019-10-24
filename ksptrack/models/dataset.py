import os
from ksptrack.models.my_augmenters import rescale_augmenter
from ksptrack.models import im_utils as utls
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
            in_shape,
            im_paths=[],
            truth_paths=None,
            locs2d=None,
            sig_prior=0.1,
            cuda=False,
            augmentations=None,
            seed=0):

        self.im_paths = im_paths
        self.augmentations = augmentations

        self.truth_paths = truth_paths
        self.locs2d = locs2d
        self.sig_prior = sig_prior
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.in_shape = in_shape
        self.seed = 0

        self.normalize = rescale_augmenter
            
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

        im_orig = im.copy()
        im_shape = im.shape[0:2]

        truth = None
        obj_prior = None

        if(self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            truth = utls.imread(truth_path, scale=False)
            truth = truth / 255

            if (len(truth.shape) == 3):
                truth = np.mean(truth, axis=-1)

            truth = truth.astype(np.uint8)
        else:
            truth = np.zeros(im.shape[:2]).astype(np.uint8)

        # Apply data augmentation
        aug_det = self.augmentations.to_deterministic()
        if(self.locs2d is not None):
            locs = self.locs2d[self.locs2d['frame'] == idx]
            locs = [utls.coord2Pixel(l['x'], l['y'], self.in_shape, self.in_shape)
                    for _, l in locs.iterrows()]

            keypoints = ia.KeypointsOnImage(
                [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
                shape=(self.in_shape, self.in_shape))
            keypoints = aug_det.augment_keypoints([keypoints])[0]

            if (len(locs) > 0):
                obj_prior = [
                    utls.make_2d_gauss((self.in_shape, self.in_shape),
                                       self.sig_prior * self.in_shape,
                                       (kp.y, kp.x)) for kp in keypoints.keypoints
                ]
                obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
                obj_prior += obj_prior.min()
                obj_prior /= obj_prior.max()
            else:
                obj_prior = (
                    np.ones((self.in_shape, self.in_shape)))[..., None]
        else:
            obj_prior = np.zeros(im_shape[:2])


        im = aug_det.augment_images([im])[0]

        truth = ia.SegmentationMapOnImage(truth,
                                          shape=truth.shape,
                                          nb_classes=1 + 1)
        truth = aug_det.augment_segmentation_maps(
            [truth])[0].get_arr_int()[..., np.newaxis]

        return {'image': im,
                'prior': obj_prior,
                'label/segmentation': truth,
                'original image': im_orig}

    @staticmethod
    def collate_fn(data):
        
        im = [np.rollaxis(d['image'], -1) for d in data]
        im = torch.stack([torch.from_numpy(i).type(torch.float)
                          for i in im])

        obj_prior = [np.rollaxis(d['prior'], -1) for d in data]
        obj_prior = torch.stack([torch.from_numpy(i).type(torch.float)
                                 for i in obj_prior])

        truth = [np.rollaxis(d['label/segmentation'], -1) for d in data]
        truth = torch.stack([torch.from_numpy(i) for i in truth])

        im_orig = [np.rollaxis(d['original image'], -1) for d in data]
        truth = torch.stack([torch.from_numpy(i) for i in im_orig])

        return{'image': im,
               'prior': obj_prior,
               'label/segmentation': truth,
               'original image': im_orig}
