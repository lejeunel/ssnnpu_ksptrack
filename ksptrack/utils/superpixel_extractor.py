import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import progressbar
from skimage import (color, io, segmentation)
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import (ndimage)
import glob, itertools
import logging
# from SLICsupervoxels import libsvx
from ksptrack.utils import my_utils as utls
from ksptrack.utils import superpixel_utils as spix_utls


def relabel(labels):

    sorted_labels = np.asarray(sorted(np.unique(labels).ravel()))
    non_contiguous = np.any((sorted_labels[1:] - sorted_labels[0:-1]) > 1)
    not_start_zero = sorted_labels[0] != 0
    if (non_contiguous or not_start_zero):
        map_dict = {sorted_labels[i]: i for i in range(sorted_labels.shape[0])}

        shape = labels.shape
        labels = labels.ravel()
        new_labels = np.copy(labels)
        for k, v in map_dict.items():
            new_labels[labels == k] = v

        return new_labels.reshape(shape)
    else:
        return labels


class SuperpixelExtractor:
    def __init__(self):

        self.logger = logging.getLogger('SuperpixelExtractor')

        # Create supervoxel object
        # self.my_svx = libsvx.svx.create()

    def extract(self,
                im_paths,
                save_dir,
                fname,
                slic_compactness=10.0,
                slic_n_sp=1000):

        self.logger.info('Got {} images'.format(len(im_paths)))

        if(len(im_paths) < 5):
            im_paths += [im_paths[-1] for i in range(5 - len(im_paths))]

        ims_list = [utls.imread(im) for im in im_paths]
        self.logger.info('Running SLIC on {} images with {} labels'.format(len(im_paths),
                                                                           slic_n_sp))
        labels = np.array([segmentation.slic(im, n_segments = slic_n_sp, compactness=slic_compactness)
                  for im in ims_list])

        # labels, numlabels = self.my_svx.run(ims_arr,
        #                                     slic_n_sp,
        #                                     slic_compactness)

        # labels = np.array([relabel(labels[..., i].copy()) for i in range(labels.shape[-1])])
        labels = np.rollaxis(labels, 0, 3)

        # self.logger.info('Num. of labels: {}' \
        #                     .format(numlabels))

        # return labels, numlabels
        return labels
