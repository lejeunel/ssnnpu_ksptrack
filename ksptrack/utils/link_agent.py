import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from itertools import count
from ksptrack.utils import csv_utils as csv
import glob
from skimage.draw import circle
from skimage import io
import warnings
import torch
from abc import ABC, abstractmethod
from ksptrack.utils.loc_prior_dataset import LocPriorDataset


class LinkAgent(ABC):
    def __init__(self,
                 csv_path,
                 data_path,
                 thr_entrance=0.5,
                 sp_labels_fname='sp_labels.npy'):

        super().__init__()

        self.dset = LocPriorDataset(data_path,
                                    normalization='rescale',
                                    sp_labels_fname=sp_labels_fname)

        self.labels_ = np.squeeze(np.array([s['labels'] for s in self.dset]))
        self.shape = self.labels.shape[1:]
        self.trans_transform = None
        self.thr_clip = 0.001
        self.locs = csv.readCsv(csv_path, as_pandas=True)
        self.thr_entrance = thr_entrance

    @property
    def labels(self):

        return self.labels_

    def get_i_j(self, loc):
        i, j = csv.coord2Pixel(loc['x'], loc['y'], self.shape[1],
                               self.shape[0])
        return i, j

    def get_all_entrance_sps(self, sp_desc_df):

        sps = []

        for f in range(self.labels.shape[0]):
            mask = self.make_entrance_mask(f) > self.thr_entrance
            labels_ = np.unique(self.labels[f, mask])
            descs_ = sp_desc_df[sp_desc_df['frame'] == f]
            descs_ = descs_.loc[descs_['label'].isin(labels_)]
            sps += [(f, row['label']) for _, row in descs_.iterrows()
                    if (mask[self.get_i_j(row)])]

        return sps

    @abstractmethod
    def make_entrance_mask(self, frame):
        pass

    def is_entrance(self, frame, label):
        """
        """

        mask = self.make_entrance_mask(frame)
        return np.mean(mask[self.labels[frame] == label]) > self.thr_entrance

    def get_closest_label(self, sp):
        """
        find in label maps the label whose centroid is the closest to 
        """
        loc_compare = self.locs.loc[self.locs['frame'] == sp['frame']]
        if (loc_compare.shape[0] == 0):
            return None

        dists = [
            np.linalg.norm(np.array((sp['x'], sp['y'])) - np.array(r_compare))
            for r_compare in [(
                x, y) for x, y in zip(loc_compare['x'], loc_compare['y'])]
        ]
        loc_min = loc_compare.iloc[np.argmin(dists)]
        i_min, j_min = csv.coord2Pixel(loc_min['x'], loc_min['y'],
                                       self.shape[1], self.shape[0])
        return self.labels[sp['frame'], i_min, j_min]
