import numpy as np
from ksptrack import iterative_ksp
from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils import my_utils as utls
from ksptrack import gc_optimize
from ksptrack import gc_refinement
from itertools import product


masks_paths_unet = {
    ('Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04', 'Dataset05'):
    'Dataset00',
    ('Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'):
    'Dataset10',
    ('Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25'):
    'Dataset20',
    ('Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34', 'Dataset35'):
    'Dataset30'
}
masks_paths_darnet = {
    ('Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04', 'Dataset05'):
    'Dataset00',
    ('Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'):
    'Dataset10',
    ('Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25'):
    'Dataset20',
    ('Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34', 'Dataset35'):
    'Dataset30'
}

if __name__ == "__main__":

    p = params.get_params('../cfgs')

    p.add('--out-path', required=True)
    p.add('--root-path', required=True)
    p.add('--sets', nargs='+', required=True)
    p.add('--set-labeled', type=str, required=True)
    p.add('--labeled-frames', nargs='+', type=int, required=True)

    cfg = p.parse_args()

    gamma_range = np.arange(cfg.gc_gamma_range[0], cfg.gc_gamma_range[1],
                            cfg.gc_gamma_step)
    lambda_range = np.arange(cfg.gc_lambda_range[0], cfg.gc_lambda_range[1],
                             cfg.gc_lambda_step)

    assert (cfg.set_labeled == cfg.sets[0]
            ), print('make labeled set first in sets to process')

    cfg.sets = ['Dataset{}'.format(set_) for set_ in cfg.sets]
    cfg.set_feat_refine = 'Dataset{}'.format(cfg.set_labeled)

    cfg.refine_gtFileNames = utls.get_images(
        pjoin(cfg.root_path, 'data', 'medical-labeling',
              'Dataset' + cfg.set_labeled, cfg.truth_dir))
    cfg.refine_gtFileNames = [
        f for i, f in enumerate(cfg.refine_gtFileNames)
        if (i in cfg.labeled_frames)
    ]

    cfg.refine_frameFileNames = utls.get_images(
        pjoin(cfg.root_path, 'data', 'medical-labeling',
              'Dataset' + cfg.set_labeled, cfg.frame_dir))
    cfg.refine_frameFileNames = [
        f for i, f in enumerate(cfg.refine_frameFileNames)
        if (i in cfg.labeled_frames)
    ]

    gc_params = {}

    sigmas = [0.2, 0.25, 0.3]
    trans_mode = ['overlap', 'radius']
    combs = list(product(sigmas, trans_mode))

    for i, dset in enumerate(cfg.sets):
        cfg.in_path = pjoin(cfg.root_path, 'data/medical-labeling', dset)
        cfg.out_path = pjoin(cfg.root_path, 'runs/ksptrack', dset)

        # constant radius ---------------------------------------------------
        cfg.feats_mode = 'autoenc'
        cfg.entrance_masks_path = None
        cfg.model_path = None

        for sig_, m_ in zip(combs):
            cfg.exp_name = 'sigma_{}_trans_{}'.format(sig_, m_)
            cfg.ml_sigma = sig_
            cfg.trans_init_mode = m_
            
            iterative_ksp.main(cfg)

        # constant radius ---------------------------------------------------
        # 0 frames
        # cfg.feats_mode = 'autoenc'
        # cfg.entrance_masks_path = None
        # cfg.model_path = None
        # cfg.exp_name = '{}_feats_disk_entr'.format(cfg.feats_mode)
        # iterative_ksp.main(cfg)

        # 1 frame
        # cfg.feats_mode = 'pred'
        # cfg.exp_name = '{}_feats_disk_entr'.format(cfg.feats_mode)
        # cfg = iterative_ksp.main(cfg)

        # optimize graph-cut
        # if (i == 0):
        #     cfg.val_run_dir = cfg.run_dir
        #     cfg.val_dir = cfg.in_path
        #     gc_gamma, gc_lambda, gc_sigma = gc_optimize.optimize_params(
        #         cfg.in_path, cfg.run_dir, 'input-frames',
        #         'ground_truth-frames', cfg.labeled_frames, gamma_range,
        #         lambda_range, cfg.gc_sigma)
        #     gc_params['disk'] = {
        #         'gamma': gc_gamma,
        #         'lambda': gc_lambda,
        #         'sigma': gc_sigma
        #     }

        # 1 frame + GC
        # gc_refinement.main(cfg.run_dir, gc_params['disk']['gamma'],
        #                    gc_params['disk']['sigma'],
        #                    gc_params['disk']['lambda'], cfg.in_path,
        #                    cfg.frame_dir, cfg.truth_dir)
        # ------------------------------------------------------------------

        # selective search -------------------------------------------------
        # cfg.entrance_masks_path = pjoin(cfg.root_path, 'runs',
        #                                 'selective_search', dset,
        #                                 cfg.csv_fname)
        # 0 frames
        # cfg.feats_mode = 'autoenc'
        # cfg.exp_name = '{}_feats_ss_entr'.format(cfg.feats_mode)
        # iterative_ksp.main(cfg)

        # 1 frame
        # cfg.feats_mode = 'pred'
        # cfg.exp_name = '{}_feats_ss_entr'.format(cfg.feats_mode)
        # cfg = iterative_ksp.main(cfg)

        # optimize graph-cut
        # if (i == 0):
        #     cfg.val_run_dir = cfg.run_dir
        #     cfg.val_dir = cfg.in_path
        #     gc_gamma, gc_lambda, gc_sigma = gc_optimize.optimize_params(
        #         cfg.in_path, cfg.run_dir, 'input-frames',
        #         'ground_truth-frames', cfg.labeled_frames, gamma_range,
        #         lambda_range, cfg.gc_sigma)
        #     gc_params['ss'] = {
        #         'gamma': gc_gamma,
        #         'lambda': gc_lambda,
        #         'sigma': gc_sigma
        #     }

        # 1 frame + GC
        # gc_refinement.main(cfg.run_dir, gc_gamma, gc_sigma, gc_lambda,
        #                    cfg.in_path, cfg.frame_dir, cfg.truth_dir)
        # gc_refinement.main(cfg.run_dir, gc_params['ss']['gamma'],
        #                    gc_params['ss']['sigma'], gc_params['ss']['lambda'],
        #                    cfg.in_path, cfg.frame_dir, cfg.truth_dir)
        # ------------------------------------------------------------------

        # U-net patch -------------------------------------------------
        # mask_path = [v for k, v in masks_paths_unet.items() if (dset in k)][0]
        # cfg.entrance_masks_path = pjoin(cfg.root_path, 'runs', 'unet_region',
        #                                 mask_path, dset, 'entrance_masks',
        #                                 'proba')
        # 0 frames
        # cfg.feats_mode = 'autoenc'
        # cfg.exp_name = '{}_feats_unet_patch_entr'.format(cfg.feats_mode)
        # iterative_ksp.main(cfg)

        # 1 frame
        # cfg.feats_mode = 'pred'
        # cfg.exp_name = '{}_feats_unet_patch_entr'.format(cfg.feats_mode)
        # cfg = iterative_ksp.main(cfg)

        # optimize graph-cut
        # if (i == 0):
        #     cfg.val_run_dir = cfg.run_dir
        #     cfg.val_dir = cfg.in_path
        #     gc_gamma, gc_lambda, gc_sigma = gc_optimize.optimize_params(
        #         cfg.in_path, cfg.run_dir, 'input-frames',
        #         'ground_truth-frames', cfg.labeled_frames, gamma_range,
        #         lambda_range, cfg.gc_sigma)
        #     gc_params['unet_patch'] = {
        #         'gamma': gc_gamma,
        #         'lambda': gc_lambda,
        #         'sigma': gc_sigma
        #     }

        # 1 frame + GC
        # gc_refinement.main(cfg.run_dir, gc_params['unet_patch']['gamma'],
        #                    gc_params['unet_patch']['sigma'],
        #                    gc_params['unet_patch']['lambda'], cfg.in_path,
        #                    cfg.frame_dir, cfg.truth_dir)
        # ------------------------------------------------------------------

        # DARNet patch -------------------------------------------------
        # mask_path = [v for k, v in masks_paths_darnet.items()
        #              if (dset in k)][0]
        # cfg.entrance_masks_path = pjoin(cfg.root_path, 'runs', 'darnet',
        #                                 mask_path, dset, 'entrance_masks',
        #                                 'pred')

        # 0 frames
        # cfg.feats_mode = 'autoenc'
        # cfg.exp_name = '{}_feats_darnet_entr'.format(cfg.feats_mode)
        # iterative_ksp.main(cfg)

        # 1 frame
        # cfg.feats_mode = 'pred'
        # cfg.exp_name = '{}_feats_darnet_entr'.format(cfg.feats_mode)
        # cfg = iterative_ksp.main(cfg)

        # optimize graph-cut
        # if (i == 0):
        #     cfg.val_run_dir = cfg.run_dir
        #     cfg.val_dir = cfg.in_path
        #     gc_gamma, gc_lambda, gc_sigma = gc_optimize.optimize_params(
        #         cfg.in_path, cfg.run_dir, 'input-frames',
        #         'ground_truth-frames', cfg.labeled_frames, gamma_range,
        #         lambda_range, cfg.gc_sigma)
        #     gc_params['darnet'] = {
        #         'gamma': gc_gamma,
        #         'lambda': gc_lambda,
        #         'sigma': gc_sigma
        #     }

        # 1 frame + GC
        # gc_refinement.main(cfg.run_dir, gc_params['darnet']['gamma'],
        #                    gc_params['darnet']['sigma'],
        #                    gc_params['darnet']['lambda'], cfg.in_path,
        #                    cfg.frame_dir, cfg.truth_dir)
        # ------------------------------------------------------------------
