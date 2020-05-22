import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import yaml
from skimage import io
from tensorboardX import SummaryWriter
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

import clustering as clst
import params
from ksptrack import prev_trans_costs
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import utils as utls
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese.loader import Loader, StackLoader, RandomBatchSampler
from ksptrack.siamese.losses import RAGTripletLoss, PointLoss
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.utils.bagging import calc_bagging
from ksptrack.siamese.train_init_clst import train_kmeans

from imgaug import augmenters as iaa


def make_data_aug(cfg):
    transf = iaa.OneOf([
        iaa.BilateralBlur(d=8,
                          sigma_color=(cfg.aug_blur_color_low,
                                       cfg.aug_blur_color_high),
                          sigma_space=(cfg.aug_blur_space_low,
                                       cfg.aug_blur_space_high)),
        iaa.AdditiveGaussianNoise(scale=(0, cfg.aug_noise * 255)),
        iaa.GammaContrast((0.5, 2.0))
    ])

    return transf


def train_one_epoch(model,
                    dataloaders,
                    optimizers,
                    device,
                    distrib_buff,
                    lr_sch,
                    cfg,
                    probas=None,
                    edges_list=None):

    model.train()

    running_loss = 0.0
    running_clst = 0.0
    running_recons = 0.0
    running_pw = 0.0
    running_obj_pred = 0.0

    criterion_clst = torch.nn.KLDivLoss(reduction='mean')
    criterion_pw = RAGTripletLoss()
    criterion_obj_pred = torch.nn.BCEWithLogitsLoss()
    criterion_recons = torch.nn.MSELoss()

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            for k in optimizers.keys():
                optimizers[k].zero_grad()

            _, targets = distrib_buff[data['frame_idx']]
            probas_ = torch.cat([probas[i] for i in data['frame_idx']])
            edges_ = edges_list[data['frame_idx'][0]].edge_index.to(
                probas_.device)
            with torch.autograd.detect_anomaly():
                res = model(data, edges_nn=edges_)

            loss = 0

            if (not cfg.fix_clst):
                loss_clst = criterion_clst(res['clusters'],
                                           targets.to(res['clusters']))
                loss += cfg.alpha * loss_clst

            if (cfg.clf):
                loss_obj_pred = criterion_obj_pred(
                    res['rho_hat_pooled'].squeeze(), (probas_ >= 0.5).float())
                loss += cfg.lambda_ * loss_obj_pred

            loss_recons = criterion_recons(sigmoid(res['output']),
                                           data['image_noaug'])
            loss += cfg.gamma * loss_recons

            if (cfg.pw):
                loss_pw = criterion_pw(res['siam_feats'], edges_)
                loss += cfg.beta * loss_pw

            with torch.autograd.detect_anomaly():
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            for k in optimizers.keys():
                optimizers[k].step()

            for k in lr_sch.keys():
                lr_sch[k].step()

        running_loss += loss.cpu().detach().numpy()
        running_recons += loss_recons.cpu().detach().numpy()
        if (not cfg.fix_clst):
            running_clst += loss_clst.cpu().detach().numpy()
        if (cfg.clf):
            running_obj_pred += loss_obj_pred.cpu().detach().numpy()
        if (cfg.pw):
            running_pw += loss_pw.cpu().detach().numpy()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        pbar.set_description('lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()

    # loss_recons = running_recons / (cfg.batch_size * len(dataloaders['train']))
    loss_pw = running_pw / (cfg.batch_size * len(dataloaders['train']))
    loss_clst = running_clst / (cfg.batch_size * len(dataloaders['train']))
    loss_obj_pred = running_obj_pred / (cfg.batch_size *
                                        len(dataloaders['train']))

    out = {
        'loss': loss_,
        'loss_pw': loss_pw,
        'loss_clst': loss_clst,
        'loss_obj_pred': loss_obj_pred,
        'loss_recons': loss_recons
    }

    return out


def train(cfg, model, device, dataloaders, run_path):

    cp_fname = 'cp_{}.pth.tar'.format(cfg.exp_name)
    best_cp_fname = 'best_{}.pth.tar'.format(cfg.exp_name)
    rags_prevs_path = pjoin(run_path, 'prevs_{}'.format(cfg.exp_name))

    path_ = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    model.load_partial(state_dict)
    # model.dec.autoencoder.to_predictor()

    check_cp_exist = pjoin(run_path, 'checkpoints', best_cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    features, pos_masks = clst.get_features(model, dataloaders['all_prev'],
                                            device)
    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)
    print('computing probability map')
    probas = calc_bagging(cat_features,
                          cat_pos_mask,
                          cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats,
                          n_jobs=1)
    probas = torch.from_numpy(probas).to(device)
    n_labels = [
        np.unique(s['labels']).size for s in dataloaders['all_prev'].dataset
    ]
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
    cfg_ksp.bag_n_feats = cfg.bag_n_feats
    cfg_ksp.bag_max_depth = cfg.bag_max_depth
    cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    cfg_ksp.use_siam_pred = False
    cfg_ksp.use_siam_trans = False
    cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = [s['frame_idx'] for s in dataloaders['prev'].dataset]

    print('generating previews to {}'.format(rags_prevs_path))
    # prev_ims = prev_trans_costs.main(cfg_ksp)
    # io.imsave(pjoin(rags_prevs_path, 'ep_0000.png'), prev_ims)

    writer = SummaryWriter(run_path)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
        'feats':
        optim.Adam(params=[{
            'params': model.dec.autoencoder.parameters(),
            'lr': 1e-3,
        }],
                   weight_decay=0),
        'gcns':
        optim.Adam(params=[{
            'params': model.locmotionapp.parameters(),
            'lr': 1e-1,
        }],
                   weight_decay=0),
        'pred':
        optim.Adam(params=[{
            'params': model.rho_dec.parameters(),
            'lr': 1e-3,
        }],
                   weight_decay=0),
        'assign':
        optim.Adam(params=[{
            'params': model.dec.assignment.parameters(),
            'lr': 1e-3,
        }],
                   weight_decay=0),
        'transform':
        optim.Adam(params=[{
            'params': model.dec.transform.parameters(),
            'lr': 1e-3,
        }],
                   weight_decay=0)
    }

    lr_sch = {
        'feats':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'],
                                               cfg.lr_power),
        'assign':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['assign'],
                                               cfg.lr_power),
        'pred':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['pred'],
                                               cfg.lr_power),
        'transform':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['transform'],
                                               cfg.lr_power),
        'gcns':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['gcns'],
                                               cfg.lr_power)
    }

    distrib_buff = DistribBuffer(cfg.tgt_update_period,
                                 thr_assign=cfg.thr_assign)
    distrib_buff.maybe_update(model, dataloaders['all_prev'], device)
    print('Generating connected components graphs')
    edges_list = utls.make_edges_ccl(model,
                                     dataloaders['edges'],
                                     device,
                                     return_signed=True,
                                     add_self_loops=True)

    for epoch in range(cfg.epochs_dist):

        if epoch < cfg.epochs_pre_pred:
            mode = 'pred'
            cfg.fix_clst = True
            cfg.clf = True
            cfg.clf_reg = False
            cfg.pw = False
        else:
            if (epoch >= cfg.epochs_pre_pred):
                if epoch == cfg.epochs_pre_pred:
                    print('training k-means')
                    init_clusters, preds, L, feats, labels = train_kmeans(
                        model,
                        dataloaders['all_prev'],
                        device,
                        cfg.n_clusters,
                        embedded_dims=cfg.embedded_dims,
                        reduc_method='pca')
                    L = torch.tensor(L).float().to(device)
                    init_clusters = torch.tensor(init_clusters,
                                                 dtype=torch.float).to(device)

                    print('Setting dim reduction and init. clusters')
                    model.dec.set_clusters(init_clusters)
                    model.dec.set_transform(L.T)
                mode = 'siam'
                cfg.fix_clst = False
                cfg.clf_reg = True
                cfg.pw = True
                if ((epoch - cfg.epochs_pre_pred) %
                        cfg.tgt_update_period == 0):
                    print('Generating connected components graphs')
                    edges_list = utls.make_edges_ccl(model,
                                                     dataloaders['edges'],
                                                     device,
                                                     return_signed=True)
                    print('Updating target distributions')
                    distrib_buff.do_update(model, dataloaders['all_prev'],
                                           device)

        # save checkpoint
        if (epoch % cfg.cp_period == 0):
            path = pjoin(run_path, 'checkpoints')
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                },
                False,
                fname_cp=cp_fname,
                fname_bm=best_cp_fname,
                path=path)
        # save previews
        if (epoch % cfg.prev_period == 0) and epoch > 0:
            out_path = rags_prevs_path

            print('generating previews to {}'.format(out_path))

            cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', cp_fname)
            cfg_ksp.use_siam_pred = cfg.clf
            cfg_ksp.use_siam_trans = cfg.pw
            prev_ims = prev_trans_costs.main(cfg_ksp)
            io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)), prev_ims)

        print('epoch {}/{}. Mode: {}'.format(epoch, cfg.epochs_dist, mode))
        res = train_one_epoch(model,
                              dataloaders,
                              optimizers,
                              device,
                              distrib_buff,
                              lr_sch,
                              cfg,
                              probas=probas,
                              edges_list=edges_list)

        # write losses to tensorboard
        for k, v in res.items():
            writer.add_scalar(k, v, epoch)

    return model


def main(cfg):

    if (cfg.clf_reg and not cfg.clf):
        raise ValueError('when clf_reg is true, clf must be true as well')

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    backbone=cfg.backbone).to(device)

    transf = make_data_aug(cfg)

    dl_train = StackLoader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                           depth=2,
                           normalization='rescale',
                           augmentations=transf,
                           resize_shape=cfg.in_shape)

    dl_prev = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                     normalization='rescale',
                     resize_shape=cfg.in_shape)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_train) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_prev, frames_tnsr_brd),
                                 collate_fn=dl_prev.collate_fn)

    dataloader_all_prev = DataLoader(dl_prev, collate_fn=dl_prev.collate_fn)

    dataloader_train = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloader_edges = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'all_prev': dataloader_all_prev,
        'edges': dataloader_edges,
        'prev': dataloader_prev
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
