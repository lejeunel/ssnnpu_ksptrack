import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
import yaml
from imgaug import augmenters as iaa
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import clustering as clst
import params
from ksptrack import prev_trans_costs
from ksptrack.siamese.loader import StackLoader
from ksptrack.cfgs import params as params_ksp
from ksptrack.siamese import train_init_clst
from ksptrack.siamese import utils as utls
from ksptrack.siamese.losses import (TripletMarginMiner, DistanceWeightedMiner,
                                     TripletCosineMarginLoss)
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.tree_set_explorer import TreeSetExplorer
from ksptrack.utils.bagging import calc_bagging

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


def train_one_epoch(model, dataloaders, optimizer, device, lr_sch, cfg, probas,
                    nodes_list, writer, epoch):

    model.train()

    # freeze the running stats of batchnorms and dropouts in encoder/decoder
    model.dec.eval()
    model.rho_dec.eval()

    running_loss = 0.0
    # running_reg = 0.0
    running_pw = 0.0

    criterion_pw = TripletCosineMarginLoss(margin=0.8)
    miner_pw = TripletMarginMiner(margin=0.8, type_of_triplets='hard')

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            nodes_ = nodes_list[data['frame_idx'][0]].to(device).detach()

            res = model(data)
            loss = 0

            # regularize norms to be ~1
            loss_reg = ((1 - res['siam_feats'].norm(dim=1))**2).mean()
            loss += cfg.delta * loss_reg

            # probas_pooled = res['rho_hat_pooled'][..., None].sigmoid().detach()
            # miner_output = miner_pw(res['pos'], nodes_[1])
            miner_output = miner_pw(res['siam_feats'], nodes_[1])
            # loss = criterion_pw(res['cosprod'], nodes_[2])
            loss += criterion_pw(res['siam_feats'],
                                 nodes_[1],
                                 indices_tuple=miner_output)
            # indices_tuple=miner_output)
            # loss += loss_pw

            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        # running_pw += loss_pw.item()
        # running_reg += loss_reg.item()
        loss = running_loss / ((i + 1) * cfg.batch_size)
        # loss_reg = running_reg / ((i + 1) * cfg.batch_size)

        niter = epoch * len(dataloaders['train']) + i
        writer.add_scalar('train/loss_{}_pw'.format(cfg.exp_name), loss, niter)
        # writer.add_scalar('train/loss_{}_reg'.format(cfg.exp_name), loss_reg,
        #                   niter)

        # pbar.set_description('lss_pw {:.4e}, lss_rg {:.4e}'.format(
        #     loss_pw, loss_reg))
        pbar.set_description('lss {:.4e}'.format(loss))

        pbar.update(1)

    pbar.close()

    lr_sch.step()

    # loss_recons = running_recons / (cfg.batch_size * len(dataloaders['train']))
    # loss_pw = running_pw / len(dataloaders['train'])
    # loss_reg = running_reg / len(dataloaders['train'])
    loss = running_loss / len(dataloaders['train'])

    # out = {'loss': loss, 'loss_pw': loss_pw, 'loss_reg': loss_reg}
    out = {'loss': loss}

    return out


def train(cfg, model, device, dataloaders, run_path):

    cp_fname = 'cp_{}.pth.tar'.format(cfg.exp_name)
    out_path = pjoin(run_path, 'prevs_{}'.format(cfg.exp_name))

    check_cp_exist = pjoin(run_path, 'checkpoints', cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    clusters = np.load(pjoin(run_path, 'init_clusters.npz'),
                       allow_pickle=True)['preds']
    res = clst.get_features(model, dataloaders['init'], device)

    probas = [torch.from_numpy(p) for p in res['obj_preds']]
    _, nodes_list = utls.make_edges_ccl(clusters,
                                        dataloaders['init'],
                                        return_signed=True,
                                        return_nodes=True,
                                        fully_connected=True)

    print('loading checkpoint {}'.format(cfg.init_cp_path))
    state_dict = torch.load(cfg.init_cp_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)

    if (not os.path.exists(out_path)):
        os.makedirs(out_path)

    p_ksp = params_ksp.get_params('../cfgs')
    p_ksp.add('--siam-path', default='')
    p_ksp.add('--in-path', default='')
    p_ksp.add('--do-all', default=False)
    p_ksp.add('--return-dict', default=False)
    p_ksp.add('--fin', nargs='+')
    cfg_ksp = p_ksp.parse_known_args(env_vars=None)[0]

    cfg_ksp.n_augs = 0
    cfg_ksp.do_scores = False
    cfg_ksp.cuda = True
    cfg_ksp.aug_method = 'none'
    cfg_ksp.siam_path = cfg.init_cp_path
    cfg_ksp.use_siam_pred = True
    cfg_ksp.siam_trans = 'siam'
    cfg_ksp.in_path = pjoin(cfg.in_root, 'Dataset' + cfg.train_dir)
    cfg_ksp.precomp_desc_path = pjoin(cfg_ksp.in_path, 'precomp_desc')
    cfg_ksp.fin = dataloaders['prev']

    print('generating previews to {}'.format(out_path))
    res = prev_trans_costs.main(cfg_ksp)

    prev_ims = np.concatenate([
        np.concatenate(
            (s['image'], s['pm'], s['pm_thr'], s['entrance']), axis=1)
        for s in res['images']
    ],
                              axis=0)
    io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(0)), prev_ims)

    writer = SummaryWriter(run_path, flush_secs=1)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizer = optim.Adam(model.locmotionapp.parameters(),
                           lr=1e-3,
                           weight_decay=1e-5)

    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.lr_power)

    for epoch in range(cfg.epochs_dist):
        # save checkpoint
        if (epoch % min(cfg.cp_period, cfg.prev_period) == 0 and epoch > 0):
            path = pjoin(run_path, 'checkpoints', cp_fname)
            utls.save_checkpoint(
                {
                    'epoch': epoch,
                    'model': model,
                    'best_loss': best_loss,
                }, path)
        # save previews
        if (epoch % cfg.prev_period == 0) and epoch > 0:

            print('generating previews to {}'.format(out_path))

            cfg_ksp.siam_path = pjoin(run_path, 'checkpoints', cp_fname)
            cfg_ksp.use_siam_pred = True
            cfg_ksp.siam_trans = 'siam'
            res = prev_trans_costs.main(cfg_ksp)
            prev_ims = np.concatenate([
                np.concatenate(
                    (s['image'], s['pm'], s['pm_thr'], s['entrance']), axis=1)
                for s in res['images']
            ],
                                      axis=0)
            io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)), prev_ims)

        print('epoch {}/{}, lr: {:.2e}'.format(epoch, cfg.epochs_dist,
                                               lr_sch.get_last_lr()[0]))
        res = train_one_epoch(model,
                              dataloaders,
                              optimizer,
                              device,
                              lr_sch,
                              cfg,
                              probas=probas,
                              nodes_list=nodes_list,
                              writer=writer,
                              epoch=epoch)

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
                    backbone=cfg.backbone).to(device)

    transf = make_data_aug(cfg)

    dl_train = StackLoader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                           depth=2,
                           normalization='rescale',
                           augmentations=transf,
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

    dataloader_train = DataLoader(dl_train,
                                  collate_fn=dl_train.collate_fn,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'init': dataloader_init,
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
    p.add('--init-cp-path', required=True)

    cfg = p.parse_args()

    main(cfg)
