from ksptrack.utils.base_dataset import BaseDataset, make_normalizer
from ksptrack.utils.kps_label import KeypointsOnLabelMap
from torch.utils.data import DataLoader
from os.path import join as pjoin
import os
import pandas as pd
import numpy as np
import imgaug as ia
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage import segmentation
from scipy import ndimage as nd
import networkx as nx


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
                 normalization=iaa.Noop(),
                 resize_shape=None,
                 csv_fname='video1.csv',
                 sig_prior=0.04):
        super().__init__(root_path=root_path)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, 'gaze-measurements', csv_fname)
        if (os.path.exists(locs2d_path)):
            self.locs2d = readCsv(locs2d_path)
        else:
            raise Exception('couldnt find 2d locs file {}'.format(locs2d_path))

        self.__normalization = make_normalizer(normalization, self.imgs)

        self.__reshaper = iaa.Noop()
        self.__augmentations = iaa.Noop()

        if (augmentations is not None):
            self.__augmentations = augmentations

        if (resize_shape is not None):
            self.__reshaper = iaa.size.Resize(resize_shape)

    def __getitem__(self, idx):

        sample = super(LocPriorDataset, self).__getitem__(idx)

        orig_shape = self.imgs[0].shape[:2]

        locs = self.locs2d[self.locs2d['frame'] == idx]
        locs = [
            coord2Pixel(l['x'], l['y'], orig_shape[1], orig_shape[0])
            for _, l in locs.iterrows()
        ]

        keypoints = ia.KeypointsOnImage(
            [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
            shape=sample['labels'].squeeze().shape)
        truth = ia.SegmentationMapsOnImage(
            sample['label/segmentation'].squeeze(),
            shape=sample['label/segmentation'].shape[:2])
        labels = ia.SegmentationMapsOnImage(sample['labels'].squeeze(),
                                            shape=sample['labels'].shape[:2])

        aug = iaa.Sequential(
            [self.__reshaper, self.__augmentations, self.__normalization])
        aug_det = aug.to_deterministic()

        im = aug_det(image=sample['image'])
        truth = aug_det(segmentation_maps=truth).get_arr()[..., None]
        labels = aug_det(segmentation_maps=labels).get_arr()[..., None]
        keypoints = aug_det.augment_keypoints(keypoints)
        shape = im.shape[:2]
        keypoints = ia.KeypointsOnImage([
            ia.Keypoint(np.clip(k.x, a_min=0, a_max=shape[1] - 1),
                        np.clip(k.y, a_min=0, a_max=shape[0] - 1))
            for k in keypoints
        ], shape)
        keypoints.labels = [
            labels[k.y_int, k.x_int, 0] for k in keypoints.keypoints
        ]
        # keypoints.update_labels([sample['labels'].squeeze()])

        sample['labels'] = labels
        sample['label/segmentation'] = truth
        sample['image'] = im
        sample['loc_keypoints'] = keypoints

        return sample

    @staticmethod
    def collate_fn(data):

        out = super(LocPriorDataset, LocPriorDataset).collate_fn(data)

        out['loc_keypoints'] = [d['loc_keypoints'] for d in data]

        return out


if __name__ == "__main__":
    transf = iaa.Sequential([
        iaa.Affine(scale={
            "x": (1 - 0.3, 1 + 0.3),
            "y": (1 - 0.3, 1 + 0.3)
        },
                   rotate=(-10, 10),
                   shear=(-10, 10)),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
    ])

    dset = LocPriorDataset(root_path=pjoin(
        '/home/ubelix/lejeune/data/medical-labeling/Dataset00'),
                           normalization='rescale',
                           augmentations=transf,
                           resize_shape=512)
    dl = DataLoader(dset, collate_fn=dset.collate_fn)

    cmap = plt.get_cmap('viridis')
    for data in dl:

        import pdb
        pdb.set_trace()  ## DEBUG ##
        im = (255 * np.rollaxis(data['image'].squeeze().detach().cpu().numpy(),
                                0, 3)).astype(np.uint8)
        labels = data['labels'].squeeze().detach().cpu().numpy()
        # labels = (cmap((labels.astype(float) / labels.max() * 255).astype(
        #     np.uint8))[..., :3] * 255).astype(np.uint8)
        print(data['loc_keypoints'])
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(labels)
        plt.show()
