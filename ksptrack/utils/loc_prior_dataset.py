import math
import os
from os.path import join as pjoin

import imgaug as ia
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from ksptrack.utils.base_dataset import (BaseDataset, base_apply_augs,
                                         make_normalizer)
from skimage import draw, segmentation
from torch.utils.data import DataLoader


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


def loc_apply_augs(sample, aug):

    sample = base_apply_augs(sample, aug)

    shape = (sample['annotations'].iloc[0].h, sample['annotations'].iloc[0].w)
    keypoints = ia.KeypointsOnImage([
        ia.Keypoint(np.clip(r.x, a_min=0, a_max=r.w - 1),
                    np.clip(r.y, a_min=0, a_max=r.h - 1))
        for _, r in sample['annotations'].iterrows()
    ], shape)

    keypoints = aug.augment_keypoints(keypoints)
    annotations = pd.DataFrame([{
        'frame':
        r.frame,
        'label':
        math.nan if math.isnan(k.y) else int(r.label),
        'n_labels':
        int(r.n_labels),
        'h':
        keypoints.shape[0],
        'w':
        keypoints.shape[1],
        'x':
        math.nan if math.isnan(k.x) else k.x_int,
        'y':
        math.nan if math.isnan(k.y) else k.y_int
    } for (_, r), k in zip(sample['annotations'].iterrows(), keypoints)])

    loc_prior = ia.HeatmapsOnImage(sample['loc_prior'],
                                   shape=sample['loc_prior'].shape)
    sample['loc_prior'] = aug.augment_heatmaps(loc_prior).get_arr()
    sample['annotations'] = annotations
    return sample


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

    return g / g.max()


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


def draw_gt_contour(im, truth, color=(255, 0, 0)):
    truth_ct = segmentation.find_boundaries(truth, mode='thick')
    im_out = np.copy(im)
    im_out[truth_ct, ...] = (255, 0, 0)

    return im_out


def draw_2d_loc(im, i, j, radius=7, color=(0, 255, 0)):
    im_out = np.copy(im)
    rr, cc = draw.disk((i, j), radius, shape=im_out.shape)
    im_out[rr, cc, :] = color

    return im_out


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
                 locs_dir='.',
                 locs_fname='2dlocs.csv',
                 sig_prior=0.20):
        super().__init__(root_path=root_path, sp_labels_fname=sp_labels_fname)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, locs_dir, locs_fname)
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

        aug = iaa.Sequential([
            self._LocPriorDataset__reshaper,
            self._LocPriorDataset__augmentations,
            self._LocPriorDataset__normalization
        ])
        aug_det = aug.to_deterministic()

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
            loc_prior = [
                make_2d_gauss(sample['labels'].shape,
                              self.sig_prior * sample['labels'].shape[0],
                              (k.y_int, k.x_int)) for k in keypoints.keypoints
            ]
            sample['loc_prior'] = np.stack(loc_prior).sum(axis=0).astype(
                np.float32)
            annotations = pd.DataFrame([{
                'frame':
                idx,
                'label':
                int(l),
                'n_labels':
                np.unique(sample['labels']).shape[0],
                'h':
                sample['labels'].shape[0],
                'w':
                sample['labels'].shape[1],
                'x':
                k.x,
                'y':
                k.y
            } for k, l in zip(keypoints, keypoints.labels)])
        else:
            annotations = pd.DataFrame([{
                'frame':
                idx,
                'label':
                np.nan,
                'n_labels':
                np.unique(sample['labels']).shape[0],
                'h':
                sample['labels'].shape[0],
                'w':
                sample['labels'].shape[1],
                'x':
                np.nan,
                'y':
                np.nan
            }])
            sample['loc_prior'] = np.ones(sample['labels'].shape,
                                          dtype=np.float32) * 0.5

        sample['annotations'] = annotations

        sample = loc_apply_augs(sample, aug_det)

        return sample

    @staticmethod
    def collate_fn(data):

        out = super(LocPriorDataset, LocPriorDataset).collate_fn(data)

        out['annotations'] = pd.concat(out['annotations'])
        out['loc_prior'] = torch.stack([
            torch.from_numpy(np.rollaxis(d['loc_prior'], -1)).float()
            for d in data
        ])

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
        '/home/ubelix/lejeune/data/medical-labeling/Dataset12'),
                           normalization='rescale',
                           augmentations=transf,
                           resize_shape=512)
    dl = DataLoader(dset, collate_fn=dset.collate_fn, batch_size=2)

    # freqs = [
    #     (s['label/segmentation'] >= 0.5).sum() / s['label/segmentation'].size
    #     for s in dset
    # ]

    # plt.plot(freqs)
    # plt.grid()
    # plt.show()

    for s in dl:

        print(s['annotations'])
        # im = s['image']
        # prior = s['label/segmentation']
        # plt.subplot(121)
        # plt.imshow(im)
        # plt.subplot(122)
        # plt.imshow(prior)
        # plt.show()
