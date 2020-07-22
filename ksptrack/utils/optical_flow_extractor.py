import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from skimage import (color, io, segmentation)
import glob
import logging
import pyflow
from ksptrack.utils.base_dataset import BaseDataset


class OpticalFlowExtractor:
    def __init__(self,
                 alpha=0.012,
                 ratio=0.75,
                 minWidth=50.,
                 nOuterFPIterations=7.,
                 nInnerFPIterations=1.,
                 nSORIterations=30.):

        self.logger = logging.getLogger('OpticalFlowExtractor')

        self.alpha = alpha
        self.ratio = ratio
        self.minWidth = minWidth
        self.nOuterFPIterations = nOuterFPIterations
        self.nInnerFPIterations = nInnerFPIterations
        self.nSORIterations = nSORIterations

    def extract(self, root_path, save_path):

        flows_bvx = []
        flows_bvy = []
        flows_fvx = []
        flows_fvy = []

        paths = [
            os.path.join(save_path, 'flows_{}.npy'.format(f))
            for f in ['fvx', 'fvy', 'bvx', 'bvy']
        ]
        exists = [os.path.exists(p) for p in paths]

        if (np.sum(exists) == 4):
            self.logger.info("Flows are already computed.")
        else:
            dset = BaseDataset(root_path)
            self.logger.info('Precomputing the optical flows...')
            for f in np.arange(1, len(dset)):
                self.logger.info('{}/{}'.format(f, len(dset)))
                im1 = dset[f - 1]['image'] / 255.
                im2 = dset[f]['image'] / 255.
                fvx, fvy, _ = pyflow.coarse2fine_flow(im1, im2, self.alpha,
                                                      self.ratio,
                                                      self.minWidth,
                                                      self.nOuterFPIterations,
                                                      self.nInnerFPIterations,
                                                      self.nSORIterations, 0)
                bvx, bvy, _ = pyflow.coarse2fine_flow(im2, im1, self.alpha,
                                                      self.ratio,
                                                      self.minWidth,
                                                      self.nOuterFPIterations,
                                                      self.nInnerFPIterations,
                                                      self.nSORIterations, 0)
                flows_bvx.append(bvx.astype(np.float32))
                flows_bvy.append(bvy.astype(np.float32))
                flows_fvx.append(fvx.astype(np.float32))
                flows_fvy.append(fvy.astype(np.float32))

            bvx = np.asarray(bvx).transpose(1, 2, 0)
            bvy = np.asarray(bvy).transpose(1, 2, 0)
            fvx = np.asarray(fvx).transpose(1, 2, 0)
            fvy = np.asarray(fvy).transpose(1, 2, 0)
            self.logger.info('Optical flow calculations done')

            self.logger.info('Saving optical flows to {}'.format(save_path))

            np.save(os.path.join(save_path, 'flows_fvx.npy'), fvx)
            np.save(os.path.join(save_path, 'flows_fvy.npy'), fvy)
            np.save(os.path.join(save_path, 'flows_bvx.npy'), bvx)
            np.save(os.path.join(save_path, 'flows_bvy.npy'), bvy)

            self.logger.info('Done.')
