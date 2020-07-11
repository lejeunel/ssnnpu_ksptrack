import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import (params, train_autoencoder, train_init_clst,
                              train_siam, train_all)

if __name__ == "__main__":

    p = params.get_params()
    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dirs', nargs='+', required=True)
    p.add('--run-dirs', nargs='+', required=True)
    cfg = p.parse_args()

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--out-path')
    p_ksp.add('--in-path')
    p_ksp.add('--siam-path', default='')
    p_ksp.add('--use-siam-pred', default=False, action='store_true')
    p_ksp.add('--use-siam-trans', default=False, action='store_true')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):
        cfg.run_dir = run_dir
        cfg.train_dir = train_dir
        cfg_ksp.cuda = True
        cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
        cfg_ksp.out_path = pjoin(
            os.path.split(cfg.out_root)[0], 'ksptrack', cfg.run_dir)

        # use focal loss foreground model with GMM
        cfg.exp_name = 'pu'
        cfg.siamese = 'none'
        cfg_ksp.exp_name = 'pu'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.use_siam_trans = False
        print('------------------------')
        print('focal loss + GMM')
        print('------------------------')
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_pu.pth.tar')
        iterative_ksp.main(cfg_ksp)

        # use focal loss foreground model with U-net siamese
        cfg.exp_name = 'unet'
        cfg.siamese = 'unet'
        cfg_ksp.exp_name = 'unet'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.use_siam_trans = True
        print('------------------------')
        print('focal loss + U-net siamese')
        print('------------------------')
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_unet.pth.tar')
        iterative_ksp.main(cfg_ksp)
