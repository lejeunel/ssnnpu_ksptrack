import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import yaml
from imgaug import augmenters as iaa
from skimage import io
from tensorboardX import SummaryWriter
from torch import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

import clustering as clst
import params
from ksptrack import prev_trans_costs
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import train_init_clst
from ksptrack.siamese import utils as utls
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese.loader import StackLoader
from ksptrack.siamese.losses import ClusterPULoss, TripletCosineMarginLoss, DistanceWeightedMiner
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.utils.bagging import calc_bagging

from sklearn.utils.class_weight import compute_class_weight

import GPUtil

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def retrain_kmeans(model, dl, device, cfg):
    init_clusters, _, L, _, _ = train_init_clst.train_kmeans(
        model,
        dl,
        device,
        cfg.n_clusters,
        embedded_dims=cfg.embedded_dims,
        comp_probas='obj_pred')

    L = torch.tensor(L).float().to(device)
    init_clusters = torch.tensor(init_clusters, dtype=torch.float).to(device)

    model.dec.set_clusters(init_clusters)
    model.dec.set_transform(L.T)

    return model


def make_data_aug(cfg):
    transf = iaa.Sequential([
        iaa.OneOf([
            iaa.BilateralBlur(d=8,
                              sigma_color=(cfg.aug_blur_color_low,
                                           cfg.aug_blur_color_high),
                              sigma_space=(cfg.aug_blur_space_low,
                                           cfg.aug_blur_space_high)),
            iaa.AdditiveGaussianNoise(scale=(0, cfg.aug_noise * 255)),
            iaa.GammaContrast((0.5, 2.0))
        ]),
        iaa.Flipud(p=0.5),
        iaa.Fliplr(p=.5),
        iaa.Rot90((1, 3))
    ])

    return transf


def train_one_epoch(model, dataloaders, optimizers, device, lr_sch, cfg,
                    probas, nodes_list):

    model.train()

    running_loss = 0.0
    running_clst = 0.0
    running_reg = 0.0
    running_pw = 0.0
    running_obj_pred = 0.0

    criterion_pw = TripletCosineMarginLoss(margin=0.3)
    miner_pw = DistanceWeightedMiner(n_triplets_per_anchor=2)
    all_probas = torch.cat(probas)
    pos_freq = ((all_probas >= 0.5).sum().float() /
                all_probas.numel()).to(device)
    criterion_obj_pred = ClusterPULoss(pi=cfg.pi_mul * pos_freq,
                                       mode=cfg.neg_mode)

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            for k in optimizers.keys():
                optimizers[k].zero_grad()

            nodes_ = nodes_list[data['frame_idx'][0]].to(device).detach()
            res = model(data)

            loss = 0

            loss_obj_pred = criterion_obj_pred(res['rho_hat_pooled'].squeeze(),
                                               nodes_, data)
            loss += loss_obj_pred

            if (cfg.pw):
                miner_output = miner_pw(
                    res['siam_feats'], nodes_[1],
                    res['rho_hat_pooled'][..., None].sigmoid(), res['pos'])
                loss_pw = criterion_pw(res['siam_feats'],
                                       res['pos'],
                                       nodes_[1],
                                       sigma=model.cs_sigma.sigmoid(),
                                       indices_tuple=miner_output)
                loss += cfg.lambda_ * loss_pw
                # print(model.cs_sigma)

                # regularize norms to be ~1
                loss_reg = ((1 - res['siam_feats'].norm(dim=1))**2).mean()
                loss += cfg.delta * loss_reg

            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()

            optimizers['feats'].step()
            lr_sch['feats'].step()
            optimizers['pred'].step()
            lr_sch['pred'].step()
            if (cfg.pw):
                optimizers['gcns'].step()
                lr_sch['gcns'].step()
                optimizers['sigma'].step()
                lr_sch['sigma'].step()

        running_loss += loss.item()
        running_obj_pred += loss_obj_pred.item()
        if (cfg.pw):
            running_pw += loss_pw.item()
            running_reg += loss_reg.item()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        pbar.set_description('lss {:.6e}'.format(loss_))
        pbar.update(1)

        del res
        torch.cuda.empty_cache()

    pbar.close()

    # loss_recons = running_recons / (cfg.batch_size * len(dataloaders['train']))
    loss_pw = running_pw / (cfg.batch_size * len(dataloaders['train']))
    loss_reg = running_reg / (cfg.batch_size * len(dataloaders['train']))
    loss_clst = running_clst / (cfg.batch_size * len(dataloaders['train']))
    loss_obj_pred = running_obj_pred / (cfg.batch_size *
                                        len(dataloaders['train']))

    out = {
        'loss': loss_,
        'loss_pw': loss_pw,
        'loss_reg': loss_reg,
        'loss_clst': loss_clst,
        'loss_obj_pred': loss_obj_pred
    }

    return out


def train(cfg, model, device, dataloaders, run_path):

    cp_fname = 'cp_{}.pth.tar'.format(cfg.exp_name)
    rags_prevs_path = pjoin(run_path, 'prevs_{}'.format(cfg.exp_name))

    path_ = pjoin(run_path, 'checkpoints',
                  'init_dec_{}.pth.tar'.format(cfg.siamese))
    # path_ = pjoin(run_path, 'checkpoints', 'cp_{}.pth.tar'.format(cfg.siamese))
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    check_cp_exist = pjoin(run_path, 'checkpoints', cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        # return

    features, pos_masks = clst.get_features(model, dataloaders['init'], device)
    cat_features = np.concatenate(features['pooled_feats'])
    cat_pos_mask = np.concatenate(pos_masks)
    print('computing probability map')
    probas = calc_bagging(cat_features,
                          cat_pos_mask,
                          cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats,
                          n_jobs=1)
    probas = torch.from_numpy(probas)

    # setting objectness prior (https://leimao.github.io/blog/Focal-Loss-Explained/)
    # all conv layers must be initialized to zero mean
    pos_freq = ((probas >= 0.5).sum().float() / probas.numel()).item()
    # pos_freq = 0.1
    prior = -np.log((1 - pos_freq) / pos_freq)
    print('setting objectness prior to {}, p={}'.format(prior, pos_freq))
    model.rho_dec.output_convolution[0].bias.data.fill_(
        torch.tensor(prior).to(device))

    n_labels = [m.shape[0] for m in pos_masks]
    probas = torch.split(probas, n_labels)

    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--siam-path', default='')
    p_ksp.add('--in-path', default='')
    p_ksp.add('--do-all', default=False)
    p_ksp.add('--return-dict', default=False)
    p_ksp.add('--fin', nargs='+')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]
    cfg_ksp.bag_t = 300
    cfg_ksp.pm_thr = 0.5
    cfg_ksp.bag_n_feats = cfg.bag_n_feats
    cfg_ksp.bag_max_depth = cfg.bag_max_depth
    cfg_ksp.siam_path = pjoin(run_path, 'checkpoints',
                              'init_dec_{}.pth.tar'.format(cfg.siamese))
    cfg_ksp.use_siam_pred = False
    cfg_ksp.use_siam_trans = False
    cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = dataloaders['prev']

    # print('generating previews to {}'.format(rags_prevs_path))
    # prev_ims = prev_trans_costs.main(cfg_ksp)
    # io.imsave(pjoin(rags_prevs_path, 'ep_0000.png'), prev_ims)

    writer = SummaryWriter(run_path, flush_secs=1)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
        'feats':
        optim.Adam(model.dec.autoencoder.parameters(),
                   lr=1e-3,
                   weight_decay=1e-4),
        'sigma':
        optim.Adam([model.cs_sigma], lr=1e-3),
        'gcns':
        optim.Adam(model.locmotionapp.parameters(), lr=1e-3),
        'pred':
        optim.Adam(model.rho_dec.parameters(), lr=1e-3, weight_decay=1e-4)
    }

    lr_sch = {
        'feats':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'],
                                               cfg.lr_power),
        'pred':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['pred'],
                                               cfg.lr_power),
        'sigma':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['sigma'],
                                               cfg.lr_power),
        'gcns':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['gcns'],
                                               cfg.lr_power)
    }

    print('Generating connected components graphs')

    _, nodes_list = utls.make_edges_ccl(model,
                                        dataloaders['init'],
                                        device,
                                        return_signed=True,
                                        return_nodes=True,
                                        fully_connected=True)

    print('syncing siamese encoder...')
    model.locmotionapp.sync(model.dec.autoencoder.encoder)

    for epoch in range(cfg.epochs_dist):

        if epoch < cfg.epochs_pre_pred:
            mode = 'pred'
            cfg.pw = False
        else:
            if (epoch >= cfg.epochs_pre_pred):
                mode = 'siam'
                cfg.pw = True

        # save checkpoint
        if (epoch % cfg.cp_period == 0):
            path = pjoin(run_path, 'checkpoints', cp_fname)
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                }, path)
        # save previews
        if (epoch % cfg.prev_period == 0) and epoch > 0:
            out_path = rags_prevs_path

            print('generating previews to {}'.format(out_path))

            cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', cp_fname)
            cfg_ksp.siam_path_clst = pjoin(
                run_path, 'checkpoints',
                'init_dec_{}.pth.tar'.format(cfg.siamese))
            cfg_ksp.use_siam_pred = True
            cfg_ksp.use_siam_trans = True
            prev_ims = prev_trans_costs.main(cfg_ksp)
            io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)), prev_ims)

        print('epoch {}/{}. Mode: {}'.format(epoch, cfg.epochs_dist, mode))
        res = train_one_epoch(model,
                              dataloaders,
                              optimizers,
                              device,
                              lr_sch,
                              cfg,
                              probas=probas,
                              nodes_list=nodes_list)

        # write losses to tensorboard
        for k, v in res.items():
            writer.add_scalar(k, v, epoch)

    return model


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    path = 'cp_{}.pth.tar'.format(cfg.exp_name)
    if (os.path.exists(path)):
        print('checkpoint {} found. Skipping.'.format(path))
        return

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    backbone=cfg.backbone,
                    siamese=cfg.siamese).to(device)

    transf = make_data_aug(cfg)

    dl_train = StackLoader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                           depth=2,
                           normalization='rescale',
                           augmentations=transf,
                           resize_shape=cfg.in_shape)
    dl_clst = StackLoader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                          depth=2,
                          normalization='rescale',
                          resize_shape=cfg.in_shape)

    dl_init = StackLoader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                          depth=2,
                          normalization='rescale',
                          resize_shape=cfg.in_shape)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_train) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_init = DataLoader(dl_init, collate_fn=dl_init.collate_fn)
    dataloader_clst = DataLoader(dl_clst, collate_fn=dl_init.collate_fn)

    dataloader_train = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'init': dataloader_init,
        'clst': dataloader_clst,
        'prev': frames_tnsr_brd
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    model = train(cfg, model, device, dataloaders, run_path)

    return model


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
