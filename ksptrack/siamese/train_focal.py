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
from ksptrack.siamese.losses import ClusterFocalLoss, ClusterDiceLoss, TripletLoss
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.utils.bagging import calc_bagging
from ksptrack.siamese.train_siam import retrain_kmeans

from sklearn.utils.class_weight import compute_class_weight


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


def train_one_epoch(model,
                    dataloaders,
                    optimizers,
                    device,
                    lr_sch,
                    cfg,
                    epoch,
                    probas,
                    focal_loss_period=5,
                    edges_list=None):

    model.train()

    running_loss = 0.0
    running_obj_pred = 0.0

    all_probas = torch.cat(probas)
    pos_freq = (all_probas >= 0.5).sum().float() / all_probas.numel()
    criterion_obj_pred = ClusterFocalLoss(alpha=cfg.alpha,
                                          gamma=cfg.gamma,
                                          mode=cfg.neg_mode)

    # criterion_obj_pred = ClusterDiceLoss()

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            for k in optimizers.keys():
                optimizers[k].zero_grad()

            edges_ = edges_list[data['frame_idx'][0]].edge_index.to(
                device).detach()

            # with torch.autograd.set_detect_anomaly(True):
            res = model(data)

            loss = 0
            # probas_ = torch.cat([probas[f] for f in data['frame_idx']])
            # alpha = (probas_ >= 0.5).sum().float() / probas_.numel()

            loss_obj_pred = criterion_obj_pred(res['rho_hat_pooled'].squeeze(),
                                               edges_, data)
            loss += loss_obj_pred

            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()

            for k in optimizers.keys():
                optimizers[k].step()

            for k in lr_sch.keys():
                lr_sch[k].step()

        running_loss += loss.cpu().detach().numpy()
        # running_recons += loss_recons.cpu().detach().numpy()
        running_obj_pred += loss_obj_pred.cpu().detach().numpy()
        loss_ = running_loss / ((i + 1) * cfg.batch_size)
        pbar.set_description('lss {:.6e}'.format(loss_))
        pbar.update(1)

    pbar.close()

    loss_obj_pred = running_obj_pred / (cfg.batch_size *
                                        len(dataloaders['train']))

    out = {'loss': loss_, 'loss_obj_pred': loss_obj_pred}

    return out


def train(cfg, model, device, dataloaders, run_path):

    cp_fname = 'cp_{}.pth.tar'.format(cfg.exp_name)
    rags_prevs_path = pjoin(run_path, 'prevs_{}'.format(cfg.exp_name))

    path_ = pjoin(run_path, 'checkpoints',
                  'init_dec_{}.pth.tar'.format(cfg.siamese))
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    check_cp_exist = pjoin(run_path, 'checkpoints', cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

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
        'pred':
        optim.Adam(model.rho_dec.parameters(), lr=1e-3, weight_decay=1e-4),
    }

    lr_sch = {
        'feats':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'],
                                               cfg.lr_power),
        'pred':
        torch.optim.lr_scheduler.ExponentialLR(optimizers['pred'],
                                               cfg.lr_power)
    }

    print('Generating connected components graphs')
    edges_list = utls.make_edges_ccl(model,
                                     dataloaders['init'],
                                     device,
                                     return_signed=True,
                                     fully_connected=True)

    for epoch in range(cfg.epochs_dist):

        # save checkpoint
        if (epoch % cfg.cp_period == 0):
            # print('retraining k-means')
            # model = retrain_kmeans(model, dataloaders['clst'], device, cfg)
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
            cfg_ksp.use_siam_pred = True
            prev_ims = prev_trans_costs.main(cfg_ksp)
            io.imsave(pjoin(out_path, 'ep_{:04d}.png'.format(epoch)), prev_ims)

        print('epoch {}/{}. Mode: {}'.format(epoch, cfg.epochs_dist, 'pred'))
        res = train_one_epoch(model,
                              dataloaders,
                              optimizers,
                              device,
                              lr_sch,
                              cfg,
                              epoch=epoch,
                              focal_loss_period=5,
                              probas=probas,
                              edges_list=edges_list)

        # write losses to tensorboard
        for k, v in res.items():
            writer.add_scalar(k, v, epoch)

    return model


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    backbone=cfg.backbone,
                    siamese='none').to(device)

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
