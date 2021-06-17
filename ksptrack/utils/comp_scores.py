import logging
import os
from os.path import join as pjoin

import numpy as np
import pandas as pd
from ksptrack import params
from ksptrack.utils.base_dataset import BaseDataset
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


def main(cfg):

    logger = logging.getLogger('comp_ksp')

    out_path = pjoin(cfg.out_path, cfg.exp_name)
    logger.info('Writing scores to: ' + out_path)

    res = np.load(os.path.join(out_path, 'results.npz'))

    dset = BaseDataset(cfg.in_path)

    truths = np.array([s['label/segmentation'] for s in dset])

    fpr, tpr, _ = roc_curve(truths.ravel(), res['ksp_scores_mat'].ravel())
    precision, recall, _ = precision_recall_curve(
        truths.ravel(), res['ksp_scores_mat'].ravel())
    f1 = f1_score(truths.ravel(), res['ksp_scores_mat'].ravel())

    data = {
        'f1_ksp': f1,
        'fpr_ksp': fpr[1],
        'tpr_ksp': tpr[1],
        'pr_ksp': precision[1],
        'rc_ksp': recall[1]
    }

    precision, recall, _ = precision_recall_curve(truths.ravel(),
                                                  res['pm_scores_mat'].ravel())
    f1 = (2 * (precision * recall) / (precision + recall)).max()
    auc_ = auc(fpr, tpr)

    data.update({'f1_pm': f1, 'auc_pm': auc_})

    df = pd.Series(data)
    df.to_csv(pjoin(out_path, 'scores.csv'))

    data = {'pr_pm': precision, 'rc_pm': recall, 'tpr_pm': tpr, 'fpr_pm': fpr}

    np.savez(pjoin(out_path, 'scores_curves.npz'), **data)


if __name__ == "__main__":
    p = params.get_params()

    p.add('--out-path', required=True)
    p.add('--in-path', required=True)

    cfg = p.parse_args()
    main(cfg)

#dir_ = os.path.join(rd.root_dir,
#                    'Dataset30/results/2017-11-07_14-49-56_exp')
