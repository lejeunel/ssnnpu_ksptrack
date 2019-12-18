from ksptrack import iterative_ksp
from ksptrack.gc_optimize import do_graph_cut
from ksptrack.cfgs import params
from ksptrack.models.dataset import Dataset
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
from skimage import io
from os.path import join as pjoin
from ksptrack.utils import my_utils as utls
from os.path import join as pjoin
import os
import numpy as np
import maxflow
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import copy


def refine_all(ims, seg, pm, gamma, lambda_, sigma):

    # do refinement
    bar = tqdm.tqdm(total=seg.shape[-1])
    gcs = np.zeros_like(seg)
    for i in range(seg.shape[-1]):
        gcs[..., i] = do_graph_cut(ims[i].mean(axis=-1) / 255,
                                   seg[..., i],
                                   pm[..., i],
                                   gamma,
                                   lambda_,
                                   sigma,
                                   return_sigma=False)
        bar.update(1)
    bar.close()

    return gcs


def main(run_dir, gamma, sigma, lambda_, data_dir, frame_dir, truth_dir):

    path_res = pjoin(run_dir, 'gc_results')
    path_frames = pjoin(path_res, 'results')
    file_gc = pjoin(path_res, 'results_gc.npz')
    file_scores = pjoin(path_res, 'scores.csv')

    if(not os.path.exists(path_res)):
        os.makedirs(path_res)
    if(not os.path.exists(path_frames)):
        os.makedirs(path_frames)

    if(os.path.exists(file_gc) and os.path.exists(path_res)):
        print('{} already exists'.format(path_res))
        return None

    # get segmentation results and inputs
    res = np.load(pjoin(run_dir, 'results.npz'))
    seg = res['ksp_scores_mat']
    pm = res['ksp_scores_mat']

    frames = utls.get_images(pjoin(data_dir, frame_dir))
    gt_frames = utls.get_images(pjoin(data_dir, truth_dir))
    loader = Dataset(im_paths=frames, truth_paths=gt_frames)

    # run refinements
    gcs = refine_all([s['image'] for s in loader], seg, pm, gamma,
                     lambda_, sigma)

    # compute scores with gc refined
    truths = np.rollaxis(
        np.array([s['label/segmentation'] > 0 for s in loader]), 0,
        -1).squeeze()
    fpr, tpr, _ = roc_curve(truths.ravel(), gcs.ravel())
    pr, rc, _ = precision_recall_curve(truths.ravel(), gcs.ravel())
    f1 = (2 * (pr * rc) / (pr + rc)).max()
    auc_ = auc(fpr, tpr)
    csv = pd.Series(
        (f1, auc_, fpr[1], tpr[1], pr[1], rc[1]),
        index=('f1', 'auc', 'fpr', 'tpr', 'pr', 'rc'))

    print('Saving gc refined to {}'.format(file_scores))
    csv.to_csv(file_scores)

    print('Writing gc refined frames to {}'.format(path_frames))
    if(not os.path.exists(path_frames)):
        os.makedirs(path_frames)

    for i, s in enumerate(loader):
        io.imsave(pjoin(path_frames, s['frame_name']),
                  gcs[..., i].astype(np.uint8) * 255,
                  check_contrast=False)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--run-dir', required=True)
    p.add('--data-dir', required=True)

    p.add('--gamma', type=float, default=None)
    p.add('--lambda_', type=float, default=None)
    p.add('--sigma', type=float, default=None)
    p.add('--params-path', type=str, default=None)

    cfg = p.parse_args()

    if (cfg.params_path is None):
        assert ((cfg.gamma is not None) or (cfg.lambda_ is not None)
                ), 'when params-path is None specify gamma and lambda'
    else:
        params = pd.read_csv(cfg.params_path,
                             header=None,
                             squeeze=True,
                             index_col=0)
        cfg.gamma = params['gamma']
        cfg.sigma = params['sigma']
        cfg.lambda_ = params['lambda']

    main(cfg.run_dir, cfg.gamma, cfg.sigma, cfg.lambda_, cfg.data_dir,
         cfg.frame_dir, cfg.truth_dir)
