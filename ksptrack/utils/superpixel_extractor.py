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
                slic_n_sp=1000,
                save_labels=False,
                save_previews=False):

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

        if (save_labels):
            self.logger.info('Saving labels to {}'.format(save_dir))
            np.savez(
                os.path.join(save_dir, fname), **{
                    'sp_labels': labels,
                    'numlabels': numlabels
                })

        if (save_previews):
            previews_dir = os.path.join(save_dir, 'spix_previews')
            if (not os.path.exists(previews_dir)):
                os.makedirs(previews_dir)
            for i, im in enumerate(ims_list):
                fname = os.path.join(previews_dir,
                                     'frame_{0:04d}.png'.format(i))

                im = spix_utls.drawLabelContourMask(im, labels[..., i])
                io.imsave(fname, im)

        return labels
