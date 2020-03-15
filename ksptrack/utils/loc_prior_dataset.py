from ksptrack.utils.base_dataset import BaseDataset
from os.path import join as pjoin
import os
import pandas as pd
import numpy as np
import imgaug as ia
from torch.utils import data
import torch
import matplotlib.pyplot as plt


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


class LocPriorDataset(BaseDataset, data.Dataset):
    """
    Adds objectness prior using 2d locations
    """
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=None,
                 csv_fname='video1.csv',
                 sig_prior=0.1):
        super().__init__(root_path, augmentations, normalization)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, 'gaze-measurements', csv_fname)
        if (os.path.exists(locs2d_path)):
            self.locs2d = readCsv(locs2d_path)
        else:
            raise Exception('couldnt find 2d locs file {}'.format(locs2d_path))

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        aug_det = sample['aug_det']

        shape = sample['image'].shape

        locs = self.locs2d[self.locs2d['frame'] == idx]
        locs = [
            coord2Pixel(l['x'], l['y'], shape[1], shape[0])
            for _, l in locs.iterrows()
        ]

        keypoints = ia.KeypointsOnImage(
            [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
            shape=(shape[0], shape[1]))
        keypoints = aug_det.augment_keypoints([keypoints])[0]

        if (len(locs) > 0):
            obj_prior = [
                make_2d_gauss((shape[0], shape[1]),
                              self.sig_prior * max(shape), (kp.y, kp.x))
                for kp in keypoints.keypoints
            ]
            obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
            offset = np.ones_like(obj_prior) * 0.5
            obj_prior -= obj_prior.min()
            obj_prior /= obj_prior.max()
            obj_prior *= 0.5
            obj_prior += offset
        else:
            obj_prior = (np.ones((shape[0], shape[1])))[..., None]

        sample['prior'] = obj_prior
        sample['loc_keypoints'] = keypoints
        rounded_kps = [(np.clip(kp.x_int, a_min=0, a_max=shape[1] - 1),
                        np.clip(kp.y_int, a_min=0, a_max=shape[0] - 1))
                       for kp in keypoints.keypoints]

        coords = np.array([(np.round(kp.y).astype(int), np.round_(kp.x).astype(int))
                           for kp in keypoints.keypoints])
        if(coords.shape[0] > 0):
            coords[:, 0] = np.clip(coords[:, 0], a_min=0, a_max=shape[0]-1)
            coords[:, 1] = np.clip(coords[:, 1], a_min=0, a_max=shape[1]-1)

        sample['labels_clicked'] = [self.labels[i, j, idx] for i, j in coords]


        return sample

    @staticmethod
    def collate_fn(data):

        out = super(LocPriorDataset, LocPriorDataset).collate_fn(data)

        obj_prior = [np.rollaxis(d['prior'], -1) for d in data]
        obj_prior = torch.stack(
            [torch.from_numpy(i).float() for i in obj_prior])

        out['prior'] = obj_prior
        out['loc_keypoints'] = [d['loc_keypoints'] for d in data]
        out['labels_clicked'] = [s['labels_clicked'] for s in data]

        return out


if __name__ == "__main__":

    dl = LocPriorDataset(
        '/home/ubelix/lejeune/data/medical-labeling/Dataset00')
    # sample = dl[10]

    for sample in dl:
        print(sample['label_keypoints'])
        plt.subplot(221)
        plt.imshow(sample['image_unnormal'])
        plt.subplot(222)
        plt.imshow(sample['prior'][..., 0])
        plt.subplot(223)
        plt.imshow(sample['labels'][..., 0])
        plt.subplot(224)
        plt.imshow(sample['labels'][..., 0] == sample['label_keypoints'][0])
        plt.show()
