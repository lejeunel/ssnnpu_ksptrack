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


class EntranceAgent:
    def __init__(self, mask_path):

        self.mask_paths = sorted(glob.glob(pjoin(mask_path, '*.png')))
        self.masks = [plt.imread(p) for p in self.mask_paths]

    def is_entrance(self,
                    centroids,
                    loc_2d,
                    tracklet,
                    labels):

        centroid_sp = centroids.loc[
            (centroids['frame'] == tracklet.get_in_frame())
            & (centroids['sp_label'] == tracklet.get_in_label())]
        i_gaze, j_gaze = csv.coord2Pixel(loc_2d[0], loc_2d[1],
                                         labels[..., 0].shape[1],
                                         labels[..., 0].shape[0])

        return self.masks[tracklet.get_in_frame()][i_gaze, j_gaze]
