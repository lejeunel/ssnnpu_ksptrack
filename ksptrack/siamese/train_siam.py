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
from ksptrack.siamese.loader import Loader
from ksptrack.siamese.losses import TripletLoss
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.utils.bagging import calc_bagging
from ksptrack.siamese.modeling.dil_unet import init_weights


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
    running_gcn = 0.0
    running_obj_pred = 0.0
    running_prob_dens = 0.0

    criterion_clst = torch.nn.KLDivLoss(reduction='mean')
    criterion_embd = TripletLoss()
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
            edges_list_ = [edges_list[i] for i in data['frame_idx']]
            res = model(data)

            loss = 0

            if (not cfg.fix_clst):
                loss_clst = criterion_clst(res['clusters'],
                                           targets.to(res['clusters']))
                loss += loss_clst

            if (cfg.clf):
                loss_obj_pred = criterion_obj_pred(res['obj_pred'].squeeze(),
                                                   (probas_ >= 0.5).float())
                loss += cfg.lambda_ * loss_obj_pred

                if (cfg.clf_reg):
                    loss_proba_density = (probas_[..., None] *
                                          res['clusters']).var(dim=0).mean()
                    loss += cfg.delta * loss_proba_density
            else:
                loss_recons = criterion_recons(sigmoid(res['output']),
                                               data['image'])
                loss += cfg.gamma * loss_recons

            if (cfg.pw):
                loss_gcn = criterion_embd(res['locmotionapp'], edges_list_)
                loss += cfg.beta * loss_gcn

            loss.backward()

            for k in optimizers.keys():
                optimizers[k].step()

            for k in lr_sch.keys():
                lr_sch[k].step()

        running_loss += loss.cpu().detach().numpy()
        if (not cfg.fix_clst):
            running_clst += loss_clst.cpu().detach().numpy()
        if (cfg.clf):
            running_obj_pred += loss_obj_pred.cpu().detach().numpy()
        if (cfg.clf_reg):
            running_prob_dens += loss_proba_density.cpu().detach().numpy()
        if (cfg.pw):
            running_gcn += loss_gcn.cpu().detach().numpy()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        pbar.set_description('lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()

    loss_recons = running_recons / (cfg.batch_size * len(dataloaders['train']))
    loss_gcn = running_gcn / (cfg.batch_size * len(dataloaders['train']))
    loss_clst = running_clst / (cfg.batch_size * len(dataloaders['train']))
    loss_obj_pred = running_obj_pred / (cfg.batch_size *
                                        len(dataloaders['train']))
    loss_prob_dens = running_prob_dens / (cfg.batch_size *
                                          len(dataloaders['train']))

    out = {
        'loss': loss_,
        'loss_gcn': loss_gcn,
        'loss_clst': loss_clst,
        'loss_obj_pred': loss_obj_pred,
        'loss_prob_dens': loss_prob_dens,
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

    if (cfg.clf):
        print('changing output of decoder to 1 channel')
        model.dec.autoencoder.to_predictor()
        # print('resetting parameters of decoder')
        # model.dec.autoencoder.decoder.apply(init_weights)

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
            'lr': cfg.lr_autoenc * 10,
        }]),
        # weight_decay=cfg.decay),
        'siam':
        optim.Adam(params=[{
            'params': model.siamese.parameters(),
            'lr': cfg.lr_dist,
        }]),
        # weight_decay=cfg.decay),
        'L':
        optim.Adam(params=[{
            'params': model.dec.transform.parameters(),
            'lr': cfg.lr_dist,
        }]),
        'assign':
        optim.Adam(params=[{
            'params': model.dec.assignment.parameters(),
            'lr': cfg.lr_dist,
        }])
    }

    lr_sch = {
        'feats':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'], 1.),
        'L': torch.optim.lr_scheduler.ExponentialLR(optimizers['L'], 1.),
        'assign':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['assign'], 1.),
        'siamese':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['siam'], 1.)
    }
    distrib_buff = DistribBuffer(cfg.tgt_update_period,
                                 thr_assign=cfg.thr_assign)
    distrib_buff.maybe_update(model, dataloaders['all_prev'], device)
    edges_list = None

    for epoch in range(1, cfg.epochs_dist + 1):
        # if (distrib_buff.converged):
        #     print('clustering assignment hit threshold. Ending training.')
        #     break

        print('epoch {}/{}'.format(epoch, cfg.epochs_dist))
        for phase in ['train', 'prev']:
            if phase == 'train':
                if (cfg.pw and ((epoch - 1) % cfg.tgt_update_period == 0)):
                    print('Generating connected components graphs')
                    edges_list = utls.make_edges_ccl(model,
                                                     dataloaders['all_prev'],
                                                     device, probas)

                res = train_one_epoch(model,
                                      dataloaders,
                                      optimizers,
                                      device,
                                      distrib_buff,
                                      lr_sch,
                                      cfg,
                                      probas=probas,
                                      edges_list=edges_list)

                if (epoch < cfg.epochs_dist):
                    distrib_buff.inc_epoch()

                if (not cfg.fix_clst):
                    distrib_buff.maybe_update(model, dataloaders['all_prev'],
                                              device)

                # write losses to tensorboard
                for k, v in res.items():
                    writer.add_scalar(k, v, epoch)

                if ((epoch % cfg.proba_update_period) == 0):
                    features, pos_masks = clst.get_features(
                        model, dataloaders['all_prev'], device)
                    cat_features = np.concatenate(features)
                    cat_pos_mask = np.concatenate(pos_masks)
                    print('computing probability map')
                    probas = calc_bagging(cat_features,
                                          cat_pos_mask,
                                          cfg.bag_t,
                                          bag_n_feats=cfg.bag_n_feats,
                                          bag_max_depth=cfg.bag_max_depth,
                                          n_jobs=1)
                    probas = torch.from_numpy(probas).to(device)
                    n_labels = [
                        np.unique(s['labels']).size
                        for s in dataloaders['all_prev'].dataset
                    ]
                    probas = torch.split(probas, n_labels)

                # save checkpoint
                if (epoch % cfg.cp_period == 0):
                    is_best = False
                    if (res['loss'] < best_loss):
                        is_best = True
                        best_loss = res['loss']
                    path = pjoin(run_path, 'checkpoints')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model': model,
                            'best_loss': best_loss,
                        },
                        is_best,
                        fname_cp=cp_fname,
                        fname_bm=best_cp_fname,
                        path=path)

            else:

                # save previews
                if (epoch % cfg.prev_period == 0):
                    out_path = rags_prevs_path

                    print('generating previews to {}'.format(out_path))

                    cfg_ksp.siam_path = pjoin(run_path, 'checkpoints',
                                              cp_fname)
                    cfg_ksp.use_siam_pred = cfg.clf
                    cfg_ksp.use_siam_trans = cfg.pw
                    prev_ims = prev_trans_costs.main(cfg_ksp)
                    io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)),
                              prev_ims)

                    # write previews to tensorboard
                    # prev_ims_pt = np.vstack([im for im in prev_ims.values()])
                    # writer.add_image('rags',
                    #                  prev_ims_pt,
                    #                  epoch,
                    #                  dataformats='HWC')

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
                    alpha=cfg.alpha,
                    backbone=cfg.backbone).to(device)

    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization='rescale',
                       resize_shape=cfg.in_shape)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_single) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_single, frames_tnsr_brd),
                                 collate_fn=dl_single.collate_fn)

    dataloader_all_prev = DataLoader(dl_single,
                                     collate_fn=dl_single.collate_fn)

    dataloader_train = DataLoader(dl_single,
                                  collate_fn=dl_single.collate_fn,
                                  batch_size=cfg.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'all_prev': dataloader_all_prev,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    model = train(cfg, model, device, dataloaders, run_path)

    # prev_ims = clst.do_prev_rags(model, device,
    #                              dataloaders['all_prev'],
    #                              couple_graphs, L)

    # save last clusterings to disk
    # last_rags_prev_path = pjoin(run_path, 'rags_prevs', 'last')
    # if (not os.path.exists(last_rags_prev_path)):
    #     os.makedirs(last_rags_prev_path)
    #     print('saving last rags previews...')
    #     for k, v in prev_ims.items():
    #         io.imsave(pjoin(last_rags_prev_path, k), v)

    return model


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
