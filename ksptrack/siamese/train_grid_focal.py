import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import (params, train_autoencoder, train_init_clst,
                              train_siam, train_all, train_focal)
from sklearn.model_selection import ParameterGrid

if __name__ == "__main__":

    p = params.get_params()
    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--run-dirs', nargs='+', required=True)
    cfg = p.parse_args()
    cfg.siamese = 'none'
    cfg.epochs_dist = 20

    for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):
        cfg.run_dir = run_dir
        cfg.train_dir = train_dir

        param_grid = {
            'neg_mode': ['rand_weighted', 'rand_uniform', 'all'],
            'alpha': [0.5, 0.75, 0.9],
            'gamma': [0, 0.5, 1, 1.5, 2]
        }
        for params in ParameterGrid(param_grid):
            cfg.neg_mode = params['neg_mode']
            cfg.alpha = params['alpha']
            cfg.gamma = params['gamma']

            # use focal loss foreground model with GMM
            cfg.exp_name = 'focal_mode_{}_alpha_{}_gamma_{}'.format(
                cfg.neg_mode, cfg.alpha, cfg.gamma)
            print('------------')
            print(cfg.run_dir)
            print(cfg.exp_name)
            print('------------')
            train_all.main(cfg)
