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
from SLICsupervoxels import libsvx
from ksptrack.utils import my_utils as utls
from ksptrack.utils import superpixel_utils as spix_utls


class SuperpixelExtractor:
    def __init__(self):

        self.logger = logging.getLogger('SuperpixelExtractor')

        # Create supervoxel object
        self.my_svx = libsvx.svx.create()

    def extract(self,
                im_paths,
                save_dir,
                fname,
                slic_compactness=10.0,
                slic_n_sp=1000):

        self.logger.info('Got {} images'.format(len(im_paths)))

        if (len(im_paths) < 5):
            im_paths += [im_paths[-1] for i in range(5 - len(im_paths))]

        ims_list = [utls.imread(im) for im in im_paths]
        ims_arr = np.asarray(ims_list).transpose(1, 2, 3, 0)
        dims = ims_arr.shape

        labels, numlabels = self.my_svx.run(ims_arr, slic_n_sp,
                                            slic_compactness)

        self.logger.info('Num. of labels: {}' \
                            .format(numlabels))

        return labels, numlabels
