from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import progressbar
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, segmentation, io)
import logging
import pandas as pd
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

    scores = res['ksp_scores_mat']

    for i in range(scores.shape[-1]):
        logger.info('{}/{}'.format(i+1,scores.shape[-1]))
        io.imsave(os.path.join(frame_dir, 'im_{}.png'.format(i)),
                  scores[..., i])

if __name__ == "__main__":
    main(sys.argv)

#dir_ = os.path.join(rd.root_dir,
#                    'Dataset30/results/2017-11-07_14-49-56_exp')
