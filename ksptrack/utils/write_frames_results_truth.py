from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import progressbar
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, segmentation, io)
import logging
import pandas as pd
import glob
from ksptrack.utils import my_utils as utls
from ksptrack.utils import learning_dataset
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import data_manager as ds
from ksptrack.exps import results_dirs as rd
from ksptrack.cfgs import cfg
from ksptrack.tr import Tracklet


def main(conf, logger=None):

    logger = logging.getLogger('plot_results_ksp')

    logger.info('--------')
    logger.info('Writing result frames to: ' + conf.dataOutDir)
    logger.info('--------')


    res = np.load(
        os.path.join(conf.dataOutDir, 'results.npz'))

    frame_dir = os.path.join(conf.dataOutDir, 'results')
    if(not os.path.exists(frame_dir)):
        logger.info('Creating output frame dir: {}'.format(frame_dir))
        os.makedirs(frame_dir)

    scores = (res['ksp_scores_mat'].astype('uint8'))*255
    imgs = [io.imread(f) for f in conf.frameFileNames]
    truth_dir = os.path.join(conf.dataInRoot, conf.dataSetDir,
                             conf.gtFrameDir)
    gts = [io.imread(f) for f in sorted(glob.glob(os.path.join(
        truth_dir, '*.png')))]

    locs2d = csv.readCsv(os.path.join(conf.dataInRoot,
                                      conf.dataSetDir,
                                      conf.gazeDir,
                                      conf.csvFileName_fg))

    for f in range(scores.shape[-1]):
        logger.info('{}/{}'.format(f+1,scores.shape[-1]))
        cont_gt = segmentation.find_boundaries(
            gts[f], mode='thick')
        idx_cont_gt = np.where(cont_gt)

        im =  csv.draw2DPoint(locs2d,
                              f,
                              imgs[f],
                              radius=7)

        im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
        score_ = np.repeat(scores[..., f][..., np.newaxis], 3, axis=2)
        im_ = np.concatenate((im, score_), axis=1)

        io.imsave(os.path.join(frame_dir, 'im_{0:04d}.png'.format(f)),
                  im_)

if __name__ == "__main__":
    main(sys.argv)

#dir_ = os.path.join(rd.root_dir,
#                    'Dataset30/results/2017-11-07_14-49-56_exp')
