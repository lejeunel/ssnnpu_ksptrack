from ksptrack.utils.base_dataset import BaseDataset
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
        super().__init__(root_path=root_path,
                         augmentations=augmentations,
                         normalization=normalization,
                         resize_shape=resize_shape)
        self.sig_prior = sig_prior

        locs2d_path = pjoin(self.root_path, 'gaze-measurements', csv_fname)
        if (os.path.exists(locs2d_path)):
            self.locs2d = readCsv(locs2d_path)
        else:
            raise Exception('couldnt find 2d locs file {}'.format(locs2d_path))

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        orig_shape = self.imgs[0].shape[:2]
        new_shape = sample['image'].shape

        locs = self.locs2d[self.locs2d['frame'] == idx]
        locs = [
            coord2Pixel(l['x'], l['y'], orig_shape[1], orig_shape[0])
            for _, l in locs.iterrows()
        ]

        keypoints = ia.KeypointsOnImage(
            [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
            shape=(orig_shape[0], orig_shape[1]))

        keypoints = self.reshaper_seg.augment_keypoints(keypoints)

        if (len(locs) > 0):
            obj_prior = [
                make_2d_gauss((new_shape[0], new_shape[1]),
                              self.sig_prior * max(new_shape), (kp.y, kp.x))
                for kp in keypoints.keypoints
            ]
            obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
            # offset = np.ones_like(obj_prior) * 0.5
            obj_prior -= obj_prior.min()
            obj_prior /= obj_prior.max()
            # obj_prior *= 0.5
            # obj_prior += offset
        else:
            obj_prior = (np.ones((new_shape[0], new_shape[1])))[..., None]

        sample['prior'] = obj_prior
        sample['loc_keypoints'] = keypoints

        coords = np.array([(np.round(kp.y).astype(int),
                            np.round_(kp.x).astype(int))
                           for kp in keypoints.keypoints])
        if (coords.shape[0] > 0):
            coords[:, 0] = np.clip(coords[:, 0],
                                   a_min=0,
                                   a_max=keypoints.shape[0] - 1)
            coords[:, 1] = np.clip(coords[:, 1],
                                   a_min=0,
                                   a_max=keypoints.shape[1] - 1)

        sample['labels_clicked'] = [
            sample['labels'][i, j, 0] for i, j in coords
        ]

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


def _add_edge_filter(values, g):
    """Add an edge between first element in `values` and
    all other elements of `values` in the graph `g`.
    `values[0]` is expected to be the central value of
    the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    g : RAG
        The graph to add edges in.

    Returns
    -------
    0.0 : float
        Always returns 0.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        g.add_edge(current, value)
    return 0.0


if __name__ == "__main__":

    from ilastikrag import rag
    import vigra

    dset = LocPriorDataset(root_path=pjoin(
        '/home/ubelix/lejeune/data/medical-labeling/Dataset00'),
                           normalization='rescale',
                           resize_shape=512)
    frames = [96, 97]
    labels_comp = [
        311, 312, 324, 332, 355, 361, 362, 370, 390, 397, 405, 407, 439, 461,
        469, 1376, 1398, 1399, 1440, 1450, 1470, 1472, 1483, 1509, 1515, 1546,
        1548, 1550
    ]
    all_labels = []
    rags = []
    max_node = 0
    for f in frames:
        labels = dset[f]['labels'] + max_node
        all_labels.append(labels.squeeze())
        max_node += labels.max() + 1

    # all_labels = np.concatenate(all_labels, axis=-1)
    # all_labels = vigra.Volume(all_labels, dtype=np.uint32)
    # full_rag = rag.Rag(all_labels).edge_ids.T.astype(np.int32)

    labels_on = [np.zeros_like(all_labels[0]) for l in all_labels]
    for i in range(len(all_labels)):
        for l in labels_comp:
            labels_on[i][all_labels[i] == l] = True

    plt.subplot(221)
    plt.imshow(labels_on[0])
    plt.subplot(222)
    plt.imshow(all_labels[0])
    plt.subplot(223)
    plt.imshow(labels_on[1])
    plt.subplot(224)
    plt.imshow(all_labels[1])
    plt.show()

# frames = [10, 11]
# label_stack = []
# max_node = 0
# for f in frames:
#     labels = dset[f]['labels']
#     label_stack.append(labels + max_node)
#     max_node += labels.max() + 1

# labels_stack = np.concatenate(label_stack, axis=-1)

# g = nx.Graph()

# # run the add-edge filter on the regions
# nd.generic_filter(labels_stack,
#                   function=_add_edge_filter,
#                   footprint=fp,
#                   mode='nearest',
#                   extra_arguments=(g, ))
