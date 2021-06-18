#!/usr/bin/env python3
import os
from os.path import join as pjoin

from ksptrack.tracking import ksp_segmentation
from ksptrack.pu import train_obj_pred
from ksptrack import params
from ksptrack.utils.superpixel_extractor import SuperpixelExtractor

if __name__ == "__main__":

    p = params.get_params()
    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    cfg = p.parse_args()

    spext = SuperpixelExtractor(cfg.in_path,
                                compactness=cfg.slic_compactness,
                                n_segments=cfg.slic_n_sp)
    spext.run()

    # do gradient ascent
    cfg.nnpu_ascent = True
    # apply frame-wise filtering to priors
    cfg.pi_filt = True
    # use ground truth to compute priors
    cfg.true_prior = False

    # pre-train using pi_overspec_ratio as prior on all frames
    cfg.phase = 0
    print('-' * 10)
    print('SSnnPU (warm-up phase)')
    cfg.exp_name = 'pu_piovrs_{}_ph0'.format(cfg.pi_overspec_ratio)
    print('exp_name: {}'.format(cfg.exp_name))
    print('in_path: {}'.format(cfg.in_path))
    print('-' * 10)
    if not os.path.exists(pjoin(cfg.out_path, cfg.exp_name)):
        train_obj_pred.main(cfg)

    # train using class-prior estimation method
    cfg.phase = 1
    cfg.pred_init_dir = cfg.exp_name
    cfg.exp_name = 'pu_piovrs_{}_ph1'.format(cfg.pi_overspec_ratio)
    print('-' * 10)
    print('SSnnPU (prior estimation phase)')
    print('exp_name: {}'.format(cfg.exp_name))
    print('in_path: {}'.format(cfg.in_path))
    print('-' * 10)
    if not os.path.exists(pjoin(cfg.out_path, cfg.exp_name)):
        train_obj_pred.main(cfg)

    # train a couple more epochs with class-priors as estimated above
    cfg.phase = 2
    cfg.last_phase_path = pjoin(cfg.out_path, cfg.exp_name)
    cfg.exp_name = 'pu_piovrs_{}_ph2'.format(cfg.pi_overspec_ratio)
    print('-' * 10)
    print('SSnnPU (prior estimation phase)')
    print('exp_name: {}'.format(cfg.exp_name))
    print('in_path: {}'.format(cfg.in_path))
    print('-' * 10)
    if not os.path.exists(pjoin(cfg.out_path, cfg.exp_name)):
        train_obj_pred.main(cfg)

    # run spatio-temporal regularization step
    if cfg.ksp:
        cfg.model_path = pjoin(cfg.out_path, cfg.exp_name, 'cps')
        ksp_segmentation.main(cfg)
