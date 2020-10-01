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
    p_ksp.add('--siam-trans', default='lfda', type=str)
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    for run_dir, train_dir in zip(cfg.run_dirs, cfg.train_dirs):
        cfg.run_dir = run_dir
        cfg.train_dir = train_dir
        cfg_ksp.cuda = True
        cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
        cfg_ksp.out_path = pjoin(
            os.path.split(cfg.out_root)[0], 'ksptrack', cfg.run_dir)

        print('------------------------')
        print('aaPU + LFDA')
        print('------------------------')
        cfg.exp_name = 'aapu'
        cfg.siamese = 'none'
        cfg.unlabeled_ratio = 0.2
        cfg.loss_obj_pred = 'pu'
        cfg.aug_method = 'none'
        cfg_ksp.exp_name = 'aapu'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.siam_trans = 'lfda'
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
        iterative_ksp.main(cfg_ksp)

        print('------------------------')
        print('nnPU + LFDA')
        print('------------------------')
        cfg.exp_name = 'pu'
        cfg.siamese = 'none'
        cfg.unlabeled_ratio = 0.
        cfg.loss_obj_pred = 'pu'
        cfg.aug_method = 'none'
        cfg_ksp.exp_name = 'pu'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.siam_trans = 'lfda'
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
        iterative_ksp.main(cfg_ksp)

        print('------------------------')
        print('tree aaPU + LFDA')
        print('------------------------')
        cfg.exp_name = 'treeaapu'
        cfg.siamese = 'none'
        cfg.unlabeled_ratio = 0.2
        cfg.loss_obj_pred = 'pu'
        cfg.aug_method = 'tree'
        cfg_ksp.exp_name = 'treeaapu'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.siam_trans = 'lfda'
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
        iterative_ksp.main(cfg_ksp)

        print('------------------------')
        print('BCE + LFDA')
        print('------------------------')
        cfg.exp_name = 'bce'
        cfg.siamese = 'none'
        cfg.unlabeled_ratio = 0.
        cfg.loss_obj_pred = 'bce'
        cfg.aug_method = 'none'
        cfg_ksp.exp_name = 'bce'
        cfg_ksp.use_siam_pred = True
        cfg_ksp.siam_trans = 'lfda'
        train_all.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
        iterative_ksp.main(cfg_ksp)

        # print('------------------------')
        # print('treeaaPU + siam')
        # print('------------------------')
        # cfg.exp_name = 'treeaapusiam'
        # cfg.siamese = 'none'
        # cfg.unlabeled_ratio = 0.5
        # cfg.loss_obj_pred = 'pu'
        # cfg.aug_method = 'none'
        # cfg.init_cp_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                          'cp_treeaapu.pth.tar')
        # cfg_ksp.exp_name = 'aapu'
        # cfg_ksp.use_siam_pred = True
        # cfg_ksp.siam_trans = 'siam'
        # train_siam.main(cfg)
        # cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
        #                           'cp_{}.pth.tar'.format(cfg_ksp.exp_name))
        # iterative_ksp.main(cfg_ksp)
