import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import logging
from ksptrack.utils import my_utils as utls
import torch
from pytorch_utils.loader import Loader
from dqnksp.episode import Episode
from dqnksp import utils as dqn_utls
from dqnksp.dqnshape import DQNShape
from imgaug import augmenters as iaa
from itertools import count
from ksptrack.utils import csv_utils as csv
import glob
from skimage.draw import circle


class EntranceAgent:
    def __init__(self, shape, mask_path=None, entrance_radius=None):

        if((entrance_radius is None) and (mask_path is None)):
            raise Exception('entrance_radius and mask_path cannot be both None')

        if((entrance_radius is not None) and (mask_path is not None)):
            raise Exception('entrance_radius and mask_path cannot be both specified')

        if((entrance_radius is not None) and (shape is None)):
            raise Exception('when using entrance_radius you must specify input shape')

        self.entrance_radius = entrance_radius
        self.shape = shape
        self.mask_path = mask_path

        if(mask_path is not None):
            self.mask_paths = sorted(glob.glob(pjoin(mask_path, '*.png')))
            self.masks = [plt.imread(p) for p in self.mask_paths]


    def is_entrance(self,
                    loc_2d,
                    idx=None):

        # idx is row index of gaze file, and also entrance masks

        i_gaze, j_gaze = csv.coord2Pixel(loc_2d['x'],
                                         loc_2d['y'],
                                         self.shape[1],
                                         self.shape[0])

        if(self.mask_path is not None):
            return self.masks[idx].astype(bool)[i_gaze, j_gaze]
        else:
            mask = np.zeros(self.shape, dtype=bool)
            rr, cc = circle(
                i_gaze,
                j_gaze,
                self.shape[0] * self.entrance_radius,
                shape=self.shape)
            mask[rr, cc] = True
            return mask[i_gaze, j_gaze]
