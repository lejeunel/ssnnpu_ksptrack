import os
import numpy as np
from skimage import io
import imgaug as ia
from imgaug import augmenters as iaa
from os.path import join as pjoin
import glob
from torch.utils import data
import torch
from ksptrack.models.my_augmenters import Rescale, Normalize
from ksptrack.models.my_augmenters import Normalize, rescale_augmenter

np.random.bit_generator = np.random._bit_generator


def imread(path, scale=True):
    im = io.imread(path)

    if (im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if (len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im


class BaseDataset(data.Dataset):
    """
    Loads and augments images and ground truths
    """
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=iaa.Noop(),
                 resize_shape=None,
                 got_labels=True):

        self.root_path = root_path

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'input-frames', e))))
        truth_paths = []
        for e in exts:
            truth_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'ground_truth-frames', e))))
        self.truth_paths = truth_paths
        self.img_paths = img_paths

        self.truths = [io.imread(f).astype('bool') for f in self.truth_paths]
        self.truths = [
            t if (len(t.shape) < 3) else t[..., 0] for t in self.truths
        ]
        self.imgs = [imread(f, scale=False) for f in self.img_paths]

        if (got_labels):
            self.labels = np.load(
                pjoin(root_path, 'precomp_desc', 'sp_labels.npz'))['sp_labels']
        else:
            self.labels = np.zeros(
                (self.imgs[0].shape[0], self.imgs[0].shape[1],
                 len(self.imgs))).astype(int)

        self.augmentations = augmentations

        if(isinstance(normalization, iaa.Augmenter)):
            self.normalization = normalization
        elif(isinstance(normalization, str)):
            # check for right parameters in list
            if normalization not in ['rescale', 'std']:
                raise ValueError("Invalid value for 'normalization': %s "
                                "'normalization' should be in "
                                "['rescale', 'std']"
                                    % normalization)
            if(normalization == 'std'):

                im_flat = np.array(self.imgs).reshape((-1, 3)) / 255
                mean = im_flat.mean(axis=0)
                std = im_flat.std(axis=0)
                self.normalization = iaa.Sequential([rescale_augmenter,
                                                    Normalize(mean=mean, std=std)])
            else:
                im_flat = np.array(self.imgs).reshape((-1, 3)) / 255
                min_ = im_flat.min(axis=0)
                max_ = im_flat.max(axis=0)
                self.normalization = iaa.Sequential([Rescale(min_, max_)])

        self.reshaper_img = iaa.Noop()
        self.reshaper_seg = iaa.Noop()

        if(resize_shape is not None):
            self.reshaper_img = iaa.size.Resize(resize_shape)
            self.reshaper_seg = iaa.size.Resize(resize_shape, 'nearest')

    def __len__(self):
        return len(self.imgs)

    def sample_uniform(self):

        random_sample_idx = np.random.choice(np.arange(len(self.imgs)))

        return self.__getitem__(random_sample_idx)

    def __getitem__(self, idx):

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        im = self.imgs[idx]
        shape = im.shape

        truth = None

        if (self.truth_paths is not None):
            truth_path = self.truth_paths[idx]
            truth = imread(truth_path, scale=False)[..., :1]
            truth = truth > 0

            if (len(truth.shape) == 3):
                truth = np.mean(truth, axis=-1)

            truth = truth.astype(np.uint8)
        else:
            truth = np.zeros(im.shape[:2]).astype(np.uint8)

        # Apply data augmentation
        if (self.augmentations is not None):
            aug_det = self.augmentations.to_deterministic()
        else:
            aug_det = iaa.Noop()

        im = aug_det.augment_images([im])[0]
        im_unnormal = im.copy()

        truth = ia.SegmentationMapsOnImage(truth, shape=truth.shape)
        labels = ia.SegmentationMapsOnImage(self.labels[..., idx].astype(
            np.int16),
                                            shape=self.labels[..., idx].shape)

        if (self.normalization is not None):
            im = self.normalization.augment_image(im)

        truth = truth.get_arr()[..., np.newaxis]
        labels = labels.get_arr()[..., np.newaxis]

        im = self.reshaper_img.augment_image(im)
        im_unnormal = self.reshaper_img.augment_image(im_unnormal)
        truth = self.reshaper_seg.augment_image(truth)
        labels = self.reshaper_seg.augment_image(labels)

        return {
            'image': im,
            'image_unnormal': im_unnormal,
            'frame_name': os.path.split(self.img_paths[idx])[-1],
            'labels': labels,
            'frame_idx': idx,
            'label/segmentation': truth
        }

    @staticmethod
    def collate_fn(data):

        im = [np.rollaxis(d['image'], -1) for d in data]
        im = torch.stack([torch.from_numpy(i).float() for i in im])

        im_unnormal = [np.rollaxis(d['image_unnormal'], -1) for d in data]
        im_unnormal = torch.stack(
            [torch.from_numpy(i).float() for i in im_unnormal])

        truth = [np.rollaxis(d['label/segmentation'], -1) for d in data]
        truth = torch.stack([torch.from_numpy(i) for i in truth]).float()

        labels = [np.rollaxis(d['labels'], -1) for d in data]
        labels = torch.stack([torch.from_numpy(i) for i in labels]).float()

        return {
            'image': im,
            'image_unnormal': im_unnormal,
            'frame_name': [d['frame_name'] for d in data],
            'frame_idx': [d['frame_idx'] for d in data],
            'label/segmentation': truth,
            'labels': labels
        }
