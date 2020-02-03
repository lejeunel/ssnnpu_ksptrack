from loader import Loader
import logging
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler, BatchSampler, SequentialSampler
import torch.optim as optim
import torch.nn.functional as F
import params
import torch
import datetime
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese.modeling.dec import DEC
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.cycle_scheduler import TrainCycleScheduler
from ksptrack.siamese.distrib_buffer import target_distribution
from ksptrack.siamese import utils as utls
from ksptrack.siamese.losses import SiameseLoss
from ksptrack.siamese import im_utils
import numpy as np
from skimage import color, io
import glob
import clustering as clst
import pandas as pd
import networkx as nx



def train_one_epoch(model, dataloaders,
                    couple_graphs,
                    optimizers,
                    device, cfg):

    criterion_siam = SiameseLoss()
    model.train()

    running_loss = 0.0

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):

            import pdb; pdb.set_trace() ## DEBUG ##
            res = model(data, couple_graphs[data['frame_idx'][0]])

            optimizers['siam'].zero_grad()
            loss = criterion_siam(res['probas_preds'], res['clusters'],
                                  couple_graphs[data['frame_idx'][0]])
            loss.backward()
            optimizers['siam'].step()
            running_loss += loss.cpu().detach().numpy()
            loss_ = running_loss / ((i + 1) * cfg.batch_size)

        pbar.set_description('lss {:.4f}'.format(loss_))
        pbar.update(1)

    pbar.close()

    out = {'loss_siam': loss_}

    return out


def train(cfg, model, device, dataloaders, run_path):
    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    rags_prevs_path = pjoin(run_path, 'rags_prevs')
    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)

    writer = SummaryWriter(run_path)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
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

    couple_graphs = utls.prepare_couple_graphs([s['graph']
                                                for s in dataloaders['buff'].dataset],
                                               0.1)

    for epoch in range(1, cfg.epochs_dist + 1):

        print('epoch {}/{}'.format(epoch, cfg.epochs_dist))
        for phase in ['train', 'prev']:
            if phase == 'train':
                res = train_one_epoch(model,
                                      dataloaders,
                                      couple_graphs,
                                      optimizers,
                                      device,
                                      cfg)

                # write losses to tensorboard
                for k, v in res.items():
                    writer.add_scalar(k, v, epoch)

                # save checkpoint
                if (epoch % cfg.cp_period == 0):
                    is_best = False
                    if (res['loss_siam'] < best_loss):
                        is_best = True
                        best_loss = res['loss_siam']
                    path = pjoin(run_path, 'checkpoints')
                    utls.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model': model,
                            'best_loss': best_loss,
                        },
                        is_best,
                        fname_cp='checkpoint_siam_epoch_{:04d}.pth.tar'.format(epoch),
                        fname_bm='best_siam.pth.tar',
                        path=path)
            else:

                # save previews
                if (epoch % cfg.prev_period == 0):
                    out_path = pjoin(rags_prevs_path, 'epoch_{:04d}'.format(epoch))
                    print('generating previews to {}'.format(out_path))

                    if (not os.path.exists(out_path)):
                        os.makedirs(out_path)

                    prev_ims = clst.do_prev_rags(model, device, dataloaders['prev'])

                    for k, v in prev_ims.items():
                        io.imsave(pjoin(out_path, k), v)


                    # write previews to tensorboard
                    prev_ims_pt = np.vstack([
                        im for im in prev_ims.values()
                    ])
                    writer.add_image('rags',
                                     prev_ims_pt,
                                     epoch,
                                     dataformats='HWC')


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    autoenc = DeepLabv3Plus(pretrained=True, embedded_dims=cfg.embedded_dims)
    dec = DEC(autoenc,
              cfg.n_clusters,
              roi_size=1,
              roi_scale=cfg.roi_spatial_scale,
              alpha=cfg.alpha)

    path_cp = sorted(
        glob.glob(pjoin(run_path, 'checkpoints',
                        'checkpoint_dec*.pth.tar')))[-1]
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        dec.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    model = Siamese(dec, cfg.embedded_dims)

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                normalization=transf_normal)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(dl, frames_tnsr_brd),
                                 batch_size=1,
                                 collate_fn=dl.collate_fn)

    dataloader_train = DataLoader(dl,
                                  batch_size=2,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    dataloader_buff = DataLoader(dl,
                                 collate_fn=dl.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'buff': dataloader_buff,
                   'prev': dataloader_prev}

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
