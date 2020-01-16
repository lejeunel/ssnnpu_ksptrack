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
from ksptrack.utils.lfda import myLFDA
from sklearn.decomposition import PCA
import warnings
import torch
from ksptrack.utils.link_agent_radius import LinkAgentRadius


class LinkAgentMask(LinkAgentRadius):
    def __init__(self,
                csv_path,
                labels,
                 thr_entrance,
                mask_path=None,
                sigma=0.07):

        super().__init__(csv_path, labels, thr_entrance=thr_entrance,
                         sigma=sigma)

        self.mask_path = mask_path
        self.sigma = sigma
        self.labels = labels

        self.mask_paths = sorted(glob.glob(pjoin(mask_path, '*.png')))
        self.mask_paths += sorted(glob.glob(pjoin(mask_path, '*.jpg')))
        self.masks = {
            f: io.imread(p, as_gray=True) / 255
            for f, p in zip(self.locs['frame'], self.mask_paths)
        }


    def make_entrance_mask(self, frame):
        mask = np.zeros(self.labels[..., 0].shape).astype(bool)
        if (frame in self.masks.keys()):
            mask = self.masks[frame]

        return mask


    def get_proba_entrance(self, tl, tl_loc, sp_desc, labels):

        # compute average probability on mask on label occupied by tl
        if (tl.get_in_frame() in self.masks.keys()):
            label_mask = labels[..., tl.get_in_frame()] == tl.get_in_label()

            proba = np.mean(self.masks[tl.get_in_frame()][label_mask])

            proba = np.clip(proba,
                            a_min=self.thr_clip,
                            a_max=1 - self.thr_clip)
            return proba
        return self.thr_clip
