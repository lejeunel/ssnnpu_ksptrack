#!/usr/bin/env python3
import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.pu import params, train_obj_pred
from ksptrack.utils.superpixel_extractor import SuperpixelExtractor

if __name__ == "__main__":

    p = params.get_params()
    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    p.add('--exp-name', required=True)
    cfg = p.parse_args()

    spext = SuperpixelExtractor(cfg.in_path,
                                compactness=cfg.slic_compactness,
                                n_segments=cfg.slic_n_sp)
    spext.run()

    cfg.nnpu_ascent = True
    cfg.pi_filt = True
    cfg.true_prior = True

    cfg.phase = 0
    print('------------------------')
    print('nnPU (ph0)')
    cfg.exp_name = 'pu_piovrs_{}_ph0'.format(cfg.pi_overspec_ratio)
    print('exp_name: {}'.format(cfg.exp_name))
    print('in_path: {}'.format(cfg.in_path))
    print('------------------------')
    cfg.loss_obj_pred = 'pu'
    if not os.path.exists(pjoin(cfg.out_path, cfg.exp_name)):
        train_obj_pred.main(cfg)

    cfg.phase = 2
    print('------------------------')
    print('nnPU (true prior per frame)')
    cfg.exp_name = 'pu_true'
    print('exp_name: {}'.format(cfg.exp_name))
    print('in_path: {}'.format(cfg.in_path))
    print('------------------------')
    cfg.true_prior = 'true'
    if not os.path.exists(pjoin(cfg.out_path, cfg.exp_name)):
        train_obj_pred.main(cfg)

    cfg.model_path = pjoin(cfg.out_path, cfg.exp_name, 'cps')
    iterative_ksp.main(cfg)
