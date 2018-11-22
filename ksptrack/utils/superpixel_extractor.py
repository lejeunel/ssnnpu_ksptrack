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


class SuperpixelExtractor:
    def __init__(self):

        self.logger = logging.getLogger('SuperpixelExtractor')

        # Create supervoxel object
        self.my_svx = libsvx.svx.create()

    def extract(self,
                im_paths,
                save_path,
                compactness=10.0,
                numreqdsupervoxels=1000,
                save=False):

        self.logger.info('Got {} images'.format(len(im_paths)))

        if (len(im_paths) < 5):
            im_paths += [im_paths[-1] for i in range(5 - len(im_paths))]

        ims = [io.imread(im) for im in im_paths]
        ims = np.asarray(ims).transpose(1, 2, 3, 0)
        dims = ims.shape

        #numrequiredsupervoxels = int(
        #    reqdsupervoxelsize_relative**2 * dims[0] * dims[1])

        labels, numlabels = self.my_svx.run(ims,
                                            numreqdsupervoxels,
                                            compactness)

        self.logger.info('Num. of labels: {}' \
                            .format(numlabels))

        if (save):
            self.logger.info('Saving labels to {}'.format(save_path))
            np.savez(save_path, **{
                'sp_labels': labels,
                'numlabels': numlabels
            })

        return labels
