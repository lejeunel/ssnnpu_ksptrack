from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, segmentation, io)
import logging
from ksptrack.utils import my_utils as utls
from ksptrack.cfgs import params
from ksptrack.tr import Tracklet


def main(cfg, out_path, logger=None):

    logger = logging.getLogger('plot_results_ksp')

    logger.info('Writing result frames to: ' + out_path)

    res = np.load(
        os.path.join(out_path, 'results.npz'))

    frame_dir = os.path.join(out_path, 'results')
    if(not os.path.exists(frame_dir)):
        logger.info('Creating output frame dir: {}'.format(frame_dir))
        os.makedirs(frame_dir)

    scores = (res['ksp_scores_mat'].astype('uint8'))*255

    for i in range(scores.shape[-1]):
        logger.info('{}/{}'.format(i+1,scores.shape[-1]))
        io.imsave(os.path.join(frame_dir, 'im_{:04d}.png'.format(i)),
                  scores[..., i])

    if('pm_scores_mat' in res.keys()):
        scores_pm = (res['pm_scores_mat']*255.).astype('uint8')
        for i in range(scores.shape[-1]):
            logger.info('{}/{}'.format(i+1,scores.shape[-1]))
            io.imsave(os.path.join(frame_dir, 'im_pb_{}.png'.format(i)),
                    scores_pm[..., i])

if __name__ == "__main__":
    p = params.get_params()

    p.add('--out-path', required=True)

    cfg = p.parse_args()
    main(cfg)

#dir_ = os.path.join(rd.root_dir,
#                    'Dataset30/results/2017-11-07_14-49-56_exp')
