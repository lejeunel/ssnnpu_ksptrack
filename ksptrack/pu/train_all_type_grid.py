#!/usr/bin/env python3

import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from ksptrack.pu import (params, train_all)
from sklearn.model_selection import ParameterGrid
import numpy as np


def phase_bool_to_str(ph):
    if ph:
        return 'mdl'
    return 'bag'


if __name__ == "__main__":

    param_grid = {
        # 'unlabeled_ratio': [0.2, 0.1, 0.3],
        'unlabeled_ratio': [0.2],
        # 'pi_mul': [1., 0.9, 1.1],
        'pi_mul': [1.]
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
    p_ksp.add('--trans-path', default='')
    p_ksp.add('--use-model-pred', default=False, action='store_true')
    p_ksp.add('--siam-trans', default='lfda', type=str)
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    cfg.nnpu_ascent = True
    cfg.aug_reset = True

    for param in param_grid:
        for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):
            cfg.sec_phase = False
            cfg.pred_init_fname = ''

            cfg.run_dir = run_dir
            cfg.train_dir = train_dir
            cfg_ksp.cuda = True
            cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
            cfg_ksp.out_path = pjoin(
                os.path.split(cfg.out_root)[0], 'ksptrack', cfg.run_dir)

            pi_mul = float(param['pi_mul'])
            unlabeled_ratio = float(param['unlabeled_ratio'])

            cfg_ksp.trans_path = pjoin(cfg.out_root, cfg.run_dir,
                                       'checkpoints', 'cp_autoenc.pth.tar')
            print('------------------------')
            print('nnPU + LFDA')
            cfg.exp_name = 'pu_pimul_{}_pr_{}'.format(
                pi_mul, phase_bool_to_str(cfg.sec_phase))
            print('exp_name: {}'.format(cfg.exp_name))
            print('in_path: {}'.format(cfg_ksp.in_path))
            print('------------------------')
            cfg.unlabeled_ratio = 0.
            cfg.pi_mul = pi_mul
            cfg.loss_obj_pred = 'pu'
            cfg.aug_method = 'none'
            cfg_ksp.exp_name = cfg.exp_name
            cfg_ksp.use_model_pred = True
            cfg_ksp.trans = 'lfda'
            train_all.main(cfg)
            cfg_ksp.model_path = pjoin(
                cfg.out_root, cfg.run_dir, 'checkpoints',
                'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
            # iterative_ksp.main(cfg_ksp)

            cfg.pred_init_fname = 'cp_{}.pth.tar'.format(cfg_ksp.exp_name)
            cfg.sec_phase = True

            print('------------------------')
            print('nnPU + LFDA')
            cfg.exp_name = 'pu_pimul_{}_pr_{}'.format(
                pi_mul, phase_bool_to_str(cfg.sec_phase))
            print('exp_name: {}'.format(cfg.exp_name))
            print('in_path: {}'.format(cfg_ksp.in_path))
            print('------------------------')
            cfg.unlabeled_ratio = 0.
            cfg.pi_mul = pi_mul
            cfg.loss_obj_pred = 'pu'
            cfg.aug_method = 'none'
            cfg_ksp.exp_name = cfg.exp_name
            cfg_ksp.use_model_pred = True
            cfg_ksp.trans = 'lfda'
            train_all.main(cfg)
            cfg_ksp.model_path = pjoin(
                cfg.out_root, cfg.run_dir, 'checkpoints',
                'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
            iterative_ksp.main(cfg_ksp)

            print('------------------------')
            print('aaPU + LFDA')
            cfg.exp_name = 'aapu_pimul_{}_ur_{}_pr_{}'.format(
                pi_mul, unlabeled_ratio, phase_bool_to_str(cfg.sec_phase))
            print('exp_name: {}'.format(cfg.exp_name))
            print('in_path: {}'.format(cfg_ksp.in_path))
            print('------------------------')
            cfg.unlabeled_ratio = unlabeled_ratio
            cfg.pi_mul = pi_mul
            cfg.loss_obj_pred = 'pu'
            cfg.aug_method = 'none'
            cfg_ksp.exp_name = cfg.exp_name
            cfg_ksp.use_model_pred = True
            cfg_ksp.trans = 'lfda'
            train_all.main(cfg)
            cfg_ksp.model_path = pjoin(
                cfg.out_root, cfg.run_dir, 'checkpoints',
                'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
            iterative_ksp.main(cfg_ksp)

            print('------------------------')
            print('tree aaPU + LFDA')
            cfg.exp_name = 'treeaapu_pimul_{}_ur_{}_pr_{}'.format(
                pi_mul, unlabeled_ratio, phase_bool_to_str(cfg.sec_phase))
            print('exp_name: {}'.format(cfg.exp_name))
            print('in_path: {}'.format(cfg_ksp.in_path))
            print('------------------------')
            cfg.unlabeled_ratio = unlabeled_ratio
            cfg.pi_mul = pi_mul
            cfg.loss_obj_pred = 'pu'
            cfg.aug_method = 'tree'
            cfg_ksp.exp_name = cfg.exp_name
            cfg_ksp.use_model_pred = True
            cfg_ksp.trans = 'lfda'
            train_all.main(cfg)
            cfg_ksp.model_path = pjoin(
                cfg.out_root, cfg.run_dir, 'checkpoints',
                'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
            iterative_ksp.main(cfg_ksp)

            # print('------------------------')
            # print('BCE + LFDA')
            # cfg.exp_name = 'bce_pimul_{}_pr_{}'.format(pi_mul, prior_method)
            # print('exp_name: {}'.format(cfg.exp_name))
            # print('in_path: {}'.format(cfg_ksp.in_path))
            # print('------------------------')
            # cfg.siamese = 'none'
            # cfg.pi_mul = pi_mul
            # cfg.prior_method = prior_method
            # cfg.unlabeled_ratio = 0.
            # cfg.loss_obj_pred = 'bce'
            # cfg.aug_method = 'none'
            # cfg_ksp.exp_name = cfg.exp_name
            # cfg_ksp.use_model_pred = True
            # cfg_ksp.trans = 'lfda'
            # train_all.main(cfg)
            # cfg_ksp.model_path = pjoin(
            #     cfg.out_root, cfg.run_dir, 'checkpoints',
            #     'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
            # iterative_ksp.main(cfg_ksp)
