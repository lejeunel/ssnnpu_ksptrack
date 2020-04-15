import os
from os.path import join as pjoin

from ksptrack import iterative_ksp
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import (params, train_autoencoder, train_init_clst,
                              train_siam)


def retrain_kmeans(cfg, in_cp_path, cp_path):
    import torch
    from torch.utils.data import DataLoader
    from ksptrack.utils.loc_prior_dataset import LocPriorDataset
    from ksptrack.siamese import utils as utls
    from ksptrack.siamese.modeling.siamese import Siamese

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    alpha=cfg.alpha,
                    backbone=cfg.backbone).to(device)
    if (cfg.clf):
        print('changing output of decoder to 1 channel')
        model.dec.autoencoder.to_predictor()

    print('loading checkpoint {}'.format(in_cp_path))
    state_dict = torch.load(in_cp_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    dl = LocPriorDataset(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                         resize_shape=cfg.in_shape,
                         normalization='rescale')
    dl = DataLoader(dl, collate_fn=dl.collate_fn, num_workers=cfg.n_workers)
    init_clusters, preds, L, feats, labels = train_init_clst.train_kmeans(
        model,
        dl,
        torch.device('cuda' if cfg.cuda else 'cpu'),
        cfg.n_clusters,
        embedded_dims=cfg.embedded_dims,
        reduc_method=cfg.reduc_method,
        bag_t=cfg.bag_t,
        bag_n_feats=cfg.bag_n_feats,
        bag_max_depth=cfg.bag_max_depth,
        up_thr=cfg.ml_up_thr,
        down_thr=cfg.ml_down_thr)

    L = torch.tensor(L).float().to(device)
    init_clusters = torch.tensor(init_clusters, dtype=torch.float).to(device)

    model.dec.set_clusters(init_clusters)
    model.dec.set_transform(L.T)

    print('saving re-initialized checkpoint {}'.format(cp_path))
    utls.save_checkpoint({
        'epoch': -1,
        'model': model
    },
                         False,
                         fname_cp=os.path.split(cp_path)[-1],
                         path=os.path.split(cp_path)[0])


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

        train_autoencoder.main(cfg)

        cfg_ksp.out_path = pjoin(
            os.path.split(cfg.out_root)[0], 'ksptrack', cfg.run_dir)
        cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
        cfg_ksp.cuda = True

        # ksp/gmm
        train_init_clst.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'init_dec.pth.tar')
        cfg_ksp.exp_name = 'gmm'
        iterative_ksp.main(cfg_ksp)

        # train predictor only and retrain k-means
        cfg.clf = True
        cfg.fix_clst = True
        cfg.exp_name = 'pred'
        train_siam.main(cfg)
        cp_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                        'cp_pred.pth.tar')
        retrain_kmeans(cfg, cp_path, cp_path)
        cfg_ksp.siam_path = cp_path
        cfg_ksp.exp_name = cfg.exp_name
        cfg_ksp.use_siam_pred = True
        iterative_ksp.main(cfg_ksp)

        # run with DEC (bagging foreground)
        cfg.clf = False
        cfg.fix_clst = False
        cfg.exp_name = 'dec'
        train_siam.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg.exp_name))
        cfg_ksp.exp_name = cfg.exp_name
        iterative_ksp.main(cfg_ksp)

        # run with DEC (DL foreground)
        cfg.clf = True
        cfg.fix_clst = False
        cfg.exp_name = 'dec_pred'
        train_siam.main(cfg)
        cfg_ksp.use_siam_pred = True
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg.exp_name))
        cfg_ksp.exp_name = cfg.exp_name
        cfg_ksp.use_siam_pred = True
        iterative_ksp.main(cfg_ksp)

        # run with DEC (DL foreground + reg)
        cfg.clf = True
        cfg.fix_clst = False
        cfg.clf_reg = True
        cfg.exp_name = 'dec_pred_reg'
        train_siam.main(cfg)
        cfg_ksp.siam_path = pjoin(cfg.out_root, cfg.run_dir, 'checkpoints',
                                  'cp_{}.pth.tar'.format(cfg.exp_name))
        cfg_ksp.exp_name = cfg.exp_name
        cfg_ksp.use_siam_pred = True
        iterative_ksp.main(cfg_ksp)

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
