from loader import Loader
import logging
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import torch.optim as optim
import torch.nn.functional as F
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese.modeling.dec import DEC
from ksptrack.siamese.cycle_scheduler import TrainCycleScheduler
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese.losses import PairwiseConstrainedClustering, SiameseLoss
from ksptrack.siamese import im_utils
import numpy as np
from skimage import color, io
import glob
import clustering as clst


def train_one_epoch(model, dataloaders, optimizers,
                    device,
                    train_sch,
                    distrib_buff,
                    cfg):

    criterion_clust = PairwiseConstrainedClustering(cfg.lambda_, cfg.n_edges)
    criterion_recons = torch.nn.MSELoss()
    criterion_siam = SiameseLoss(cfg.beta, cfg.n_edges, cfg.with_flow)

    mode = train_sch.get_cycle()
    curr_epoch = train_sch.curr_epoch

    model.train()

    if (mode == 'feats'):
        model.grad_linears(False)
        model.grad_dec(True)
    else:
        model.grad_linears(True)
        model.grad_dec(False)

    running_loss = 0.0
    running_loss_pur = 0.0
    running_loss_recons = 0.0

    pbar = tqdm.tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            # backward + optimize only if in training phase
            res = model(data)

            distrib_buff.maybe_update(model, dataloaders['prev'], device)
            distribs, targets = distrib_buff[data['frame_idx']]

            if (mode == 'feats'):
                optimizers['autoenc'].zero_grad()
                optimizers['assign'].zero_grad()

                loss_clust, _ = criterion_clust(data['graph'],
                                                res['feats'],
                                                res['clusters'],
                                                targets)

                loss_recons = criterion_recons(res['recons'], data['image'])
                loss = cfg.gamma * loss_clust + loss_recons
                loss.backward()
                optimizers['autoenc'].step()
                optimizers['assign'].step()
                running_loss += loss.cpu().detach().numpy()
                running_loss_recons += loss_recons.cpu().detach().numpy()
                running_loss_pur += loss_clust.cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * cfg.batch_size)
                loss_pur_ = running_loss_pur / ((i + 1) * cfg.batch_size)
                loss_recons_ = running_loss_recons / ((i + 1) * cfg.batch_size)
            else:
                optimizers['siam'].zero_grad()
                optimizers['autoenc'].zero_grad()
                sampled = criterion_siam(data['graph'], res['clusters'],
                                         targets,
                                         res['feats'], model.calc_probas)
                sampled['loss'].backward()
                optimizers['siam'].step()
                optimizers['autoenc'].step()
                running_loss += sampled['loss'].cpu().detach().numpy()
                loss_ = running_loss / ((i + 1) * cfg.batch_size)

        pbar.set_description('[{}] ep {}/{}, lss {:.4f}'.format(
            train_sch.get_cycle(), curr_epoch + 1, cfg.epochs_all, loss_))
        pbar.update(1)

    pbar.close()

    if (mode == 'feats'):
        losses = {
            'recons_pur': loss_,
            'recons': loss_recons_,
            'pur': loss_pur_
        }
    else:
        losses = {'siam': loss_}

    return losses



def train(cfg, model, device, dataloaders, run_path):
    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    rags_prevs_path = pjoin(run_path, 'rags_prevs')
    clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    frames_tnsr_brd = np.linspace(0,
                                  len(dataloaders['prev']) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    # train initial clustering
    if (not os.path.exists(init_clusters_path)):

        if (not cfg.with_agglo):
            init_clusters, preds = clst.train_kmeans(model,
                                                     dataloaders['prev'],
                                                     device, cfg.n_clusters,
                                                     cfg.with_pck)
        else:
            init_clusters, preds = clst.train_agglo(model, dataloaders['prev'],
                                                    device, cfg.n_clusters,
                                                    cfg.linkage)
        np.savez(init_clusters_path, **{
            'clusters': init_clusters,
            'preds': preds
        })

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    prev_ims = clst.do_prev_clusters_init(dataloaders['prev'], preds,
                                          frames=frames_tnsr_brd)

    # save initial clusterings to disk
    if (not os.path.exists(init_clusters_prev_path)):
        os.makedirs(init_clusters_prev_path)
        print('saving initial clustering previews...')
        for k, v in prev_ims.items():
            io.imsave(pjoin(init_clusters_prev_path, k), v)

    init_prev = np.vstack([
        prev_ims[k] for k in prev_ims.keys()
    ])

    writer = SummaryWriter(run_path)

    # can choose order of cycles here
    # train_sch = TrainCycleScheduler([cfg.epochs_dec, cfg.epochs_dist],
    #                                 cfg.epochs_all, ['feats', 'siam'])
    train_sch = TrainCycleScheduler([cfg.epochs_dec, cfg.epochs_dist],
                                    cfg.epochs_all, ['feats', 'siam'])
    distrib_buff = DistribBuffer(cfg.tgt_update_period)

    writer.add_image('clusters',
                     init_prev,
                     0,
                     dataformats='HWC')

    milestones = [cfg.epochs_dec]
    milestones = [milestones]
    init_clusters = torch.tensor(init_clusters,
                                 dtype=torch.float,
                                 requires_grad=True)
    if cfg.cuda:
        init_clusters = init_clusters.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()['assignment.cluster_centers'].copy_(init_clusters)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_all))

    model.to(device)

    optimizers = {
        'autoenc':
        optim.SGD(params=[{
            'params': model.autoencoder.parameters(),
            'lr': cfg.lr_autoenc
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay),
        'assign':
        optim.SGD(params=[{
            'params': model.assignment.parameters(),
            'lr': cfg.lr_assign
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay),
        'siam':
        optim.SGD(params=[{
            'params': model.linear1.parameters(),
            'lr': cfg.lr_dist
        }, {
            'params': model.linear2.parameters(),
            'lr': cfg.lr_dist
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay)
    }

    for epoch in range(1, cfg.epochs_all + 1):

        # if(distrib_buff.converged):
        #     print('labels assignments threshold hit!')
        #     break

        losses = train_one_epoch(model, dataloaders, optimizers,
                                 device, train_sch,
                                 distrib_buff,
                                 cfg)

        # write losses to tensorboard
        for k, v in losses.items():
            writer.add_scalar('loss_{}'.format(k), v, epoch)

        # save previews
        if (epoch % cfg.prev_period == 0):
            print('generating previews')
            prev_ims_clst = clst.do_prev_clusters(model, device,
                                                  dataloaders['prev'],
                                                  frames_tnsr_brd)
            prev_ims_rags = clst.do_prev_rags(model, device,
                                              dataloaders['prev'],
                                              frames_tnsr_brd)

            out_paths = [
                pjoin(p, 'epoch_{:04d}'.format(epoch))
                for p in [rags_prevs_path, clusters_prevs_path]
            ]
            types = ['rags', 'clusters']
            prev_ims = [prev_ims_rags, prev_ims_clst]

            for ims, out_path, type_ in zip(prev_ims, out_paths, types):
                print('saving previews to {}'.format(out_path))
                if (not os.path.exists(out_path)):
                    os.makedirs(out_path)
                for k, v in tqdm.tqdm(ims.items()):
                    io.imsave(pjoin(out_path, k), v)

                # write previews to tensorboard
                prev_ims_pt = np.vstack([
                    ims[k] for k in ims.keys()
                ])
                writer.add_image('{}'.format(type_),
                                 prev_ims_pt,
                                 epoch,
                                 dataformats='HWC')

        # save checkpoint
        if (epoch > 1 and (epoch % cfg.cp_period == 0)):
            is_best = False
            if ('siam' in losses.keys()):
                if(losses['siam'] < best_loss):
                    is_best = True
                    best_loss = losses['siam']
            path = pjoin(run_path, 'checkpoints')
            utls.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model': model,
                    'best_loss': best_loss,
                },
                is_best,
                fname_cp='checkpoint_dec_epoch_{:04d}.pth.tar'.format(epoch),
                fname_bm='best_dec.pth.tar',
                path=path)

        train_sch.step()


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DEC(cfg.n_clusters, roi_size=1, roi_scale=cfg.roi_spatial_scale,
                alpha=cfg.alpha)

    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.autoencoder.load_state_dict(state_dict)
        model.autoencoder
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    transf, transf_normal = im_utils.make_data_aug(cfg)

    dl_train = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                      normalization=transf_normal)

    dataloader_prev = DataLoader(dl_train,
                                 batch_size=1,
                                 collate_fn=dl_train.collate_fn)
    dataloader_train = DataLoader(dl_train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl_train.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train, 'prev': dataloader_prev}

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
