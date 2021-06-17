#!/usr/bin/env python3

import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from ksptrack.pu import params, train_all
from sklearn.model_selection import ParameterGrid


def phase_bool_to_str(ph):
    if ph:
        return 'mdl'
    return 'bag'


if __name__ == "__main__":

    param_grid = {
        'unlabeled_ratio': [0.12],
        'pi_ovrs': [1.4, 0.8, 1.0, 1.8, 1.6, 1.2]
    }
    param_grid = ParameterGrid(param_grid)

    p = params.get_params()
    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--run-dirs', nargs='+', required=True)
    cfg = p.parse_args()

    cfg.cuda = True

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--out-path')
    p_ksp.add('--in-path')
    p_ksp.add('--model-path', default='')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    cfg.nnpu_ascent = True
    cfg.pi_filt = True
    cfg.aug_in_neg = False
    cfg.coordconv = False
    cfg_ksp.coordconv = False

    cfg.loc_prior = False

    for param in param_grid:
        for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):

            cfg.true_prior = ''

            cfg.pred_init_dir = ''

            cfg.run_dir = run_dir
            cfg.train_dir = train_dir
            cfg_ksp.cuda = True
            cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
            cfg_ksp.out_path = pjoin(
                os.path.split(cfg.out_root)[0], 'ksptrack', cfg.run_dir)

            cfg.pi_overspec_ratio = param['pi_ovrs']
            unlabeled_ratio = float(param['unlabeled_ratio'])

            cfg_ksp.trans_path = pjoin(cfg.out_root, 'autoenc', 'cp.pth.tar')
            cfg_ksp.trans_path = pjoin(cfg.out_root, cfg.run_dir, 'autoenc',
                                       'cp.pth.tar')

            cfg.true_prior = False
            cfg.phase = 0
            cfg.nnpu_ascent = True

            if cfg.pi_overspec_ratio >= 1.2:
                print('------------------------')
                print('nnPU (ph0)')
                cfg.exp_name = 'pu_piovrs_{}_ph0'.format(cfg.pi_overspec_ratio)
                print('exp_name: {}'.format(cfg.exp_name))
                print('in_path: {}'.format(cfg_ksp.in_path))
                print('------------------------')
                cfg.unlabeled_ratio = 0.
                cfg.loss_obj_pred = 'pu'
                cfg.aug_method = 'none'
                cfg.pred_init_dir = ''
                train_all.main(cfg)

                cfg.nnpu_ascent = True
                cfg.true_prior = False
                cfg.phase = 1
                cfg.pred_init_dir = cfg.exp_name

                print('------------------------')
                print('nnPU (ph1)')
                cfg.exp_name = 'pu_piovrs_{}_ph1'.format(cfg.pi_overspec_ratio)
                print('exp_name: {}'.format(cfg.exp_name))
                print('in_path: {}'.format(cfg_ksp.in_path))
                print('------------------------')
                cfg.unlabeled_ratio = 0.
                cfg.loss_obj_pred = 'pu'
                cfg.aug_method = 'none'
                train_all.main(cfg)

                cfg.pred_init_dir = cfg.exp_name

                cfg.phase = 2
                print('------------------------')
                print('nnPU (ph2)')
                cfg.exp_name = 'pu_piovrs_{}_ph2'.format(cfg.pi_overspec_ratio)
                print('exp_name: {}'.format(cfg.exp_name))
                print('in_path: {}'.format(cfg_ksp.in_path))
                print('------------------------')
                cfg.unlabeled_ratio = 0.
                cfg.loss_obj_pred = 'pu'
                cfg.aug_method = 'none'
                cfg_ksp.exp_name = cfg.exp_name
                cfg_ksp.use_model_pred = True
                cfg_ksp.trans = 'lfda'
                train_all.main(cfg)
                cfg_ksp.model_path = pjoin(cfg.out_root, cfg.run_dir,
                                           cfg_ksp.exp_name, 'cps')
                iterative_ksp.main(cfg_ksp)

            if cfg.pi_overspec_ratio <= 1.4:
                cfg.phase = 2
                print('------------------------')
                print('nnPU (mean true prior cst.)')
                cfg.exp_name = 'pu_meantrue_cst_piovrs_{}'.format(
                    cfg.pi_overspec_ratio)
                print('exp_name: {}'.format(cfg.exp_name))
                print('in_path: {}'.format(cfg_ksp.in_path))
                print('------------------------')
                cfg.unlabeled_ratio = 0.
                cfg.loss_obj_pred = 'pu'
                cfg.true_prior = 'mean'
                cfg.aug_method = 'none'
                cfg_ksp.exp_name = cfg.exp_name
                cfg_ksp.use_model_pred = True
                cfg_ksp.trans = 'lfda'
                train_all.main(cfg)
                cfg_ksp.model_path = pjoin(cfg.out_root, cfg.run_dir,
                                           cfg_ksp.exp_name, 'cps')
                iterative_ksp.main(cfg_ksp)

            cfg.phase = 2
            print('------------------------')
            print('nnPU (true prior per frame)')
            cfg.exp_name = 'pu_true'
            print('exp_name: {}'.format(cfg.exp_name))
            print('in_path: {}'.format(cfg_ksp.in_path))
            print('------------------------')
            cfg.unlabeled_ratio = 0.
            cfg.loss_obj_pred = 'pu'
            cfg.aug_method = 'none'
            cfg.true_prior = 'true'
            cfg_ksp.exp_name = cfg.exp_name
            cfg_ksp.use_model_pred = True
            cfg_ksp.trans = 'lfda'
            train_all.main(cfg)
            cfg_ksp.model_path = pjoin(cfg.out_root, cfg.run_dir,
                                       cfg_ksp.exp_name, 'cps')
            iterative_ksp.main(cfg_ksp)
