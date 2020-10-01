from ksptrack.utils.base_dataset import BaseDataset, make_normalizer, base_apply_augs
from ksptrack.utils.kps_label import KeypointsOnLabelMap
from torch.utils.data import DataLoader, Dataset
from os.path import join as pjoin
import os
import pandas as pd
import numpy as np
import imgaug as ia
from torch.utils import data
import torch
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage import segmentation
from scipy import ndimage as nd
import networkx as nx


def relabel(labels):

    sorted_labels = np.asarray(sorted(np.unique(labels).ravel()))
    if (np.any((sorted_labels[1:] - sorted_labels[0:-1]) > 1)):
        mapping = sorted_labels[..., None]
        mapping = np.concatenate((np.unique(labels)[..., None], mapping),
                                 axis=1)
        _, ind = np.unique(labels, return_inverse=True)
        shape = labels.shape
        labels = mapping[ind, 1:].reshape((shape[0], shape[1]))

    return labels


def loc_apply_augs(im, labels, truth, keypoints, aug):

    im, labels, truth = base_apply_augs(im, labels, truth, aug)

    keypoints = aug.augment_keypoints(keypoints)
    return im, labels, truth, keypoints


def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)


def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


def readCsv(csvName, seqStart=None, seqEnd=None):

    out = np.loadtxt(open(csvName, "rb"), delimiter=";",
                     skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)

    return pd.DataFrame(data=out,
                        columns=['frame', 'time', 'visible', 'x', 'y'])


def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j


class LocPriorDataset(BaseDataset):
    """
    Adds objectness prior using 2d locations
    """
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=iaa.Noop(),
                 resize_shape=None,
                 sp_labels_fname='sp_labels.npy',
                 csv_fname='video1.csv',
                 sig_prior=0.04):
        super().__init__(root_path=root_path, sp_labels_fname=sp_labels_fname)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, 'gaze-measurements', csv_fname)
        if (os.path.exists(locs2d_path)):
            self.locs2d = readCsv(locs2d_path)
        else:
            raise Exception('couldnt find 2d locs file {}'.format(locs2d_path))

        self.__normalization = make_normalizer(normalization,
                                               map(lambda s: s['image'], self))

        self.__reshaper = iaa.Noop()
        self.__augmentations = iaa.Noop()

        if (augmentations is not None):
            self.__augmentations = augmentations

        if (resize_shape is not None):
            self.__reshaper = iaa.size.Resize(resize_shape)

    def __getitem__(self, idx):

        sample = super(LocPriorDataset, self).__getitem__(idx)

        orig_shape = sample['image'].shape[:2]

        locs = self.locs2d[self.locs2d['frame'] == idx]
        locs = [
            coord2Pixel(l['x'], l['y'], orig_shape[1], orig_shape[0])
            for _, l in locs.iterrows()
        ]

        keypoints = ia.KeypointsOnImage(
            [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
            shape=sample['labels'].squeeze().shape)

        aug = iaa.Sequential(
            [self.__reshaper, self.__augmentations, self.__normalization])
        aug_det = aug.to_deterministic()

        sample['image'], sample['labels'], sample[
            'label/segmentation'], keypoints = loc_apply_augs(
                sample['image'], sample['labels'],
                sample['label/segmentation'], keypoints, aug_det)

        sample['labels'] = relabel(sample['labels'])

        shape = sample['image'].shape[:2]
        keypoints = ia.KeypointsOnImage([
            ia.Keypoint(np.clip(k.x, a_min=0, a_max=shape[1] - 1),
                        np.clip(k.y, a_min=0, a_max=shape[0] - 1))
            for k in keypoints
        ], shape)
        keypoints.labels = [
            np.squeeze(sample['labels'])[k.y_int, k.x_int]
            for k in keypoints.keypoints
        ]

        if len(keypoints.labels) > 0:
            pos_labels = pd.DataFrame([{
                'frame':
                idx,
                'label':
                int(l),
                'n_labels':
                np.unique(sample['labels']).shape[0]
            } for l in keypoints.labels])
        else:
            pos_labels = pd.DataFrame([{
                'frame':
                idx,
                'label':
                np.nan,
                'n_labels':
                np.unique(sample['labels']).shape[0]
            }])

        sample['loc_keypoints'] = keypoints
        sample['pos_labels'] = pos_labels

        return sample

    @staticmethod
    def collate_fn(data):

        out = super(LocPriorDataset, LocPriorDataset).collate_fn(data)

        out['loc_keypoints'] = [d['loc_keypoints'] for d in data]
        out['pos_labels'] = pd.concat(out['pos_labels'])

        return out


if __name__ == "__main__":

    transf = iaa.Sequential([
        iaa.OneOf([
            iaa.BilateralBlur(d=8,
                              sigma_color=(100, 150),
                              sigma_space=(100, 150)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.06 * 255)),
            iaa.GammaContrast((1.0, 2.0))
        ])
        # iaa.Flipud(p=0.5),
        # iaa.Fliplr(p=.5),
        # iaa.Rot90((1, 3))
    ])

    dset = LocPriorDataset(root_path=pjoin(
        '/home/ubelix/lejeune/data/medical-labeling/Dataset10'),
                           normalization='rescale',
                           augmentations=transf,
                           resize_shape=512)
    dl = DataLoader(dset, collate_fn=dset.collate_fn)
    n = 50
    f = 97

    for _ in range(n):

        im = dset[f]['image']
        plt.imshow(im)
        plt.show()
