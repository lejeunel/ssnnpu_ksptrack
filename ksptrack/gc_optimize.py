from .cfgs import params
from .models.dataset import Dataset
from sklearn.metrics import f1_score
from os.path import join as pjoin
from .utils import my_utils as utls

from os.path import join as pjoin
import os
import numpy as np
import maxflow
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import copy


def do_graph_cut(im, seg, pm, gamma, lambda_, sigma=None, return_sigma=True):
    
    dx_im = im - np.roll(im, -1, axis=1)
    dy_im = im - np.roll(im, 1, axis=0)

    if(sigma is None):
        # estimate sigma from input image
        sigma = np.mean(np.hstack([(dx_im**2).ravel(), (dy_im**2).ravel()]))

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(seg.shape)
        
    eps = np.finfo(np.float).eps
    # Edges pointing right/left
    structure = np.zeros((3,3))
    structure[1,2] = 1
    weights = np.exp(-(dx_im**2)/(2*sigma**2))
    g.add_grid_edges(nodeids,
                        structure=structure,
                        weights=lambda_*weights,
                        symmetric=True)

    # Edges pointing up/down
    structure = np.zeros((3,3))
    structure[0,1] = 1
    weights = np.exp(-(dy_im**2)/(2*sigma**2))
    g.add_grid_edges(nodeids,
                        structure=structure,
                        weights=lambda_*weights,
                        symmetric=True)

    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    pos_unary = 0.5 * np.ones(pm.shape)
    pos_unary[seg] = 1
    pos_unary = -np.log(pos_unary + eps) / -np.log(eps)
    pos_unary -= gamma * np.log(pm + eps) / -np.log(eps)

    neg_unary = 0.5 * np.ones(pm.shape)
    neg_unary[pm == 0] = 1
    neg_unary = -np.log(neg_unary + eps) / -np.log(eps)
    neg_unary -= gamma * np.log(1 - pm + eps) / -np.log(eps)

    g.add_grid_tedges(nodeids, pos_unary, neg_unary)

    g.maxflow()
    out = g.get_grid_segments(nodeids)

    if(return_sigma):
        return out, sigma
    else:
        return out

def optimize_params(val_dir,
                    val_run_dir,
                    frame_dir,
                    truth_dir,
                    val_frames,
                    gamma_range,
                    lambda_range,
                    sigma):

    path_gc = pjoin(val_run_dir, 'gc_results')
    file_gc = pjoin(path_gc, 'results_gc.npz')
    path_params = pjoin(val_run_dir, 'gc_results')
    file_params = pjoin(path_params, 'best_gc_params.csv')

    if(not os.path.exists(path_gc)):
        os.makedirs(path_gc)
    if(not os.path.exists(path_params)):
        os.makedirs(path_params)

    if(os.path.exists(file_gc) and os.path.exists(file_params)):
        print('{} already exists, loading.'.format(file_params))
        csv = pd.read_csv(file_params, index_col=0, header=None, squeeze=True)
        return csv['gamma'], csv['lambda'], csv['sigma']

    # Make validation frame loader
    frames = utls.get_images(
        pjoin(val_dir, frame_dir))
    gt_frames = utls.get_images(pjoin(val_dir, truth_dir))
    loader = Dataset(im_paths=frames,
                     truth_paths=gt_frames)
    samples = [loader[f] for f in val_frames]

    # get validation binary frame to refine
    results = np.load(pjoin(val_run_dir, 'results.npz'))
    val_ksp = [results['ksp_scores_mat'][..., f] for f in val_frames]
    val_pm = [results['pm_scores_mat'][..., f] for f in val_frames]
    val_imgs = [loader[f]['image'] for f in val_frames]
    val_fnames = [loader[f]['frame_name'] for f in val_frames]
    val_truths = [loader[f]['label/segmentation'] for f in val_frames]

    gamma, lambda_ = np.meshgrid(gamma_range,
                                 lambda_range)
    params = np.vstack((gamma.ravel(), lambda_.ravel())).T

    res_1_frame = [{'lambda': None,
                    'gamma': None,
                    'sigma': None,
                    'gc': None,
                    'f1':None}
                   for _ in range(params.shape[0])]
    res = [{'fname': fname,
            'im': im,
            'truth': truth,
            'pm': pm,
            'seg': seg,
            'res': copy.deepcopy(res_1_frame)}
           for fname, im, truth, pm, seg, res in zip(val_fnames,
                                                     val_imgs,
                                                     val_truths,
                                                     val_pm,
                                                     val_ksp,
                                                     res_1_frame)]

    bar = tqdm.tqdm(total=len(val_imgs) * params.shape[0])
    for i, (seg, pm, im, truth) in enumerate(zip(val_ksp, val_pm, val_imgs, val_truths)):
        res[i]['im'] = im
        res[i]['truth'] = truth
        res[i]['seg'] = seg
        res[i]['pm'] = pm
        for j, (gamma, lambda_) in enumerate(params):
            gc, sigma = do_graph_cut(im.mean(axis=-1) / 255,
                                     seg, pm, gamma, lambda_,
                                     sigma=sigma)
            res[i]['res'][j]['lambda'] = lambda_
            res[i]['res'][j]['gamma'] = gamma
            res[i]['res'][j]['sigma'] = sigma
            res[i]['res'][j]['gc'] = gc
            res[i]['res'][j]['f1'] = f1_score(truth[..., 0].ravel(), gc.ravel())

            bar.update(1)

    bar.close()

    print('saving results to {}'.format(file_gc))
    np.savez(file_gc, res)

    # for all images and all parameter set, stack f1 score
    dfs = []
    for r in res:
        r.pop('im')
        r.pop('truth')
        r.pop('pm')
        r.pop('seg')
        for r_ in r['res']:
            r_.pop('gc')

        df = pd.DataFrame.from_dict(r['res'])
        dfs.append(df)

    res_df = pd.concat(dfs, names=['frame'],
                       keys=[r['fname'] for r in res], axis=0)

    # get best params
    idx_max_f1 = res_df['f1'].mean(level=1).idxmax()
    best_gamma, best_lambda = params[idx_max_f1, :]
    best_sigma = res_df['sigma'].mean(level=1)[idx_max_f1]
    best = pd.Series((best_gamma, best_lambda, best_sigma), index=('gamma', 'lambda', 'sigma'))

    print('saving best params to {}'.format(file_params))
    best.to_csv(file_params, header=False)

    return best_gamma, best_lambda, best_sigma

if __name__ == "__main__":

    p = params.get_params()

    p.add('--val-run-dir', required=True)
    p.add('--val-dir', required=True)
    p.add('--val-frames', nargs='+', type=int, required=True)
    p.add('--gamma-range', nargs='+', type=float, required=True)
    p.add('--gamma-step', type=float, required=True)
    p.add('--lambda-range', nargs='+', type=float, required=True)
    p.add('--lambda-step', type=float, required=True)
    p.add('--sigma', type=float, default=None)

    cfg = p.parse_args()

    assert(len(cfg.gamma_range) == 2), 'gamma-range must be two values'
    assert(len(cfg.lambda_range) == 2), 'lambda-range must be two values'

    gamma_range = np.arange(cfg.gamma_range[0],
                            cfg.gamma_range[1],
                            cfg.gamma_step)
    lambda_range = np.arange(cfg.lambda_range[0],
                             cfg.lambda_range[1],
                             cfg.lambda_step)

    optimize_params(cfg.val_dir,
                    cfg.val_run_dir,
                    'input-frames',
                    'ground_truth-frames',
                    cfg.val_frames,
                    gamma_range,
                    lambda_range,
                    cfg.sigma)
