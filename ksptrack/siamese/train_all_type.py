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
        cfg.exp_name = 'focal'
        cfg.siamese = 'none'
        cfg_ksp.exp_name = 'focal'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.use_siam_trans = False
        print('------------------------')
        print('focal loss + GMM')
        print('------------------------')
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_focal.pth.tar')
        iterative_ksp.main(cfg_ksp)

        # use focal loss foreground model with GCN
        cfg.exp_name = 'gcn'
        cfg.siamese = 'gcn'
        cfg_ksp.exp_name = 'gcn'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.use_siam_trans = True
        print('------------------------')
        print('focal loss + GCN')
        print('------------------------')
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_gcn.pth.tar')
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

        # ksp/gmm
        # train_init_clst.main(cfg)
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'init_dec.pth.tar')
        # cfg_ksp.exp_name = 'gmm'
        # iterative_ksp.main(cfg_ksp)

        # train predictor only and retrain k-means
        # cfg.clf = True
        # cfg.fix_clst = True
        # cfg.exp_name = 'pred'
        # train_siam.main(cfg)
        # cp_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                 'cp_pred.pth.tar')
        # retrain_kmeans(cfg, cp_path, cp_path)
        # cfg_ksp.siam_path = cp_path
        # cfg_ksp.exp_name = cfg.exp_name
        # cfg_ksp.use_siam_pred = True
        # iterative_ksp.main(cfg_ksp)

        # run with DEC (bagging foreground)
        # cfg.clf = False
        # cfg.fix_clst = False
        # cfg.exp_name = 'dec'
        # train_siam.main(cfg)
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'cp_{}.pth.tar'.format(cfg.exp_name))
        # cfg_ksp.exp_name = cfg.exp_name
        # iterative_ksp.main(cfg_ksp)

        # run with DEC (DL foreground)
        # cfg.clf = True
        # cfg.fix_clst = False
        # cfg.exp_name = 'dec_pred'
        # train_siam.main(cfg)
        # cfg_ksp.use_siam_pred = True
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'cp_{}.pth.tar'.format(cfg.exp_name))
        # cfg_ksp.exp_name = cfg.exp_name
        # cfg_ksp.use_siam_pred = True
        # iterative_ksp.main(cfg_ksp)

        # run with DEC (DL foreground + reg)
        # cfg.clf = True
        # cfg.fix_clst = False
        # cfg.clf_reg = True
        # cfg.exp_name = 'dec_pred_reg'
        # train_siam.main(cfg)
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'cp_{}.pth.tar'.format(cfg.exp_name))
        # cfg_ksp.exp_name = cfg.exp_name
        # cfg_ksp.use_siam_pred = True
        # iterative_ksp.main(cfg_ksp)

        # run with gcn (DL foreground + reg)
        # cfg.clf = True
        # cfg.clf_reg = True
        # cfg.fix_clst = True
        # cfg.pw = True
        # cfg.exp_name = 'pw_pred_reg'
        # train_siam.main(cfg)
        # cfg_ksp.use_siam_pred = True
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'cp_{}.pth.tar'.format(cfg.exp_name))
        # cfg_ksp.exp_name = cfg.exp_name
        # cfg_ksp.use_siam_trans = True
        # iterative_ksp.main(cfg_ksp)
