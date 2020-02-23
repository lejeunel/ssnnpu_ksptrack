from loader import Loader, StackLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import params
import torch
from torch.nn import functional as F
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese import utils as utls
from ksptrack.siamese import im_utils
from ksptrack.siamese.losses import TripletLoss
import numpy as np
from skimage import io
import clustering as clst


def train_one_epoch(model, dataloaders, couple_graphs, optimizers, device,
                    L, cfg):

    model.train()

    running_loss = 0.0
    criterion = TripletLoss()

    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):

            # get edge array of consecutive frames
            edges_nn = couple_graphs[data['frame_idx'][0]][data['frame_idx']
                                                           [1]]['edges_nn']
            # get cluster assignments of respective nodes
            clst = couple_graphs[data['frame_idx'][0]][data['frame_idx']
                                                       [1]]['clst']
            # no assignment... it has been done earlier
            res = model(data, edges_nn, do_assign=False)
            Y = (clst[edges_nn[:, 0]] == clst[edges_nn[:, 1]]).float()

            # optimizers['autoencoder'].zero_grad()
            optimizers['siam'].zero_grad()
            
            # compute positive/negative weights
            pos_weight = (Y == 0).sum().float() / (Y == 1).sum().float()
            loss = F.binary_cross_entropy_with_logits(res['probas_preds'],
                                                      Y,
                                                      pos_weight=pos_weight)

            loss.backward()
            optimizers['siam'].step()
            # optimizers['autoencoder'].step()
            running_loss += loss.cpu().detach().numpy()
            loss_ = running_loss / ((i + 1) * cfg.batch_size)

        pbar.set_description('lss {:.6f}'.format(loss_))
        pbar.update(1)

    pbar.close()

    out = {'loss_siam': loss_}

    return out


def train(cfg, model, device, dataloaders, run_path):

    path_clst = pjoin(run_path, 'clusters.npz')
    if(not os.path.exists(path_clst)):
        path_clst = pjoin(run_path, 'init_clusters.npz')
    print('loading clusters at {}'.format(path_clst))
    L = np.load(path_clst, allow_pickle=True)['L']
    L = torch.tensor(L).float().to(device)

    couple_graphs = utls.make_all_couple_graphs(model,
                                                device,
                                                dataloaders['train'],
                                                cfg.nn_radius,
                                                L,
                                                do_inter_frame=True)
    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_siam.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Loading and skipping.'.format(check_cp_exist))
        state_dict = torch.load(check_cp_exist,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        return couple_graphs

    rags_prevs_path = pjoin(run_path, 'rags_prevs')
    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)

    writer = SummaryWriter(run_path)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
        'autoencoder':
        optim.SGD(params=[{
            'params': model.dec.autoencoder.parameters(),
            'lr': cfg.lr_dist,
        }]),
        'siam':
        optim.SGD(params=[{
            'params': model.linear1.parameters(),
            'lr': cfg.lr_dist,
        }, {
            'params': model.linear2.parameters(),
            'lr': cfg.lr_dist,
        }]),
    }

    for epoch in range(1, cfg.epochs_dist + 1):

        print('epoch {}/{}'.format(epoch, cfg.epochs_dist))
        for phase in ['train', 'prev']:
            if phase == 'train':
                res = train_one_epoch(model, dataloaders, couple_graphs,
                                      optimizers, device, L, cfg)

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
                        fname_cp='checkpoint_siam.pth.tar'.format(epoch),
                        fname_bm='best_siam.pth.tar',
                        path=path)

            else:

                # save previews
                if (epoch % cfg.prev_period == 0):
                    out_path = pjoin(rags_prevs_path,
                                     'epoch_{:04d}'.format(epoch))
                    print('generating previews to {}'.format(out_path))

                    if (not os.path.exists(out_path)):
                        os.makedirs(out_path)

                    prev_ims = clst.do_prev_rags(model,
                                                 device,
                                                 dataloaders['prev'],
                                                 couple_graphs,
                                                 do_assign=True,
                                                 L=L)

                    for k, v in prev_ims.items():
                        io.imsave(pjoin(out_path, k), v)

                    # write previews to tensorboard
                    prev_ims_pt = np.vstack([im for im in prev_ims.values()])
                    writer.add_image('rags',
                                     prev_ims_pt,
                                     epoch,
                                     dataformats='HWC')
    return couple_graphs


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')
    model = Siamese(cfg.embedded_dims,
                    cfg.n_clusters,
                    roi_size=1,
                    roi_scale=cfg.roi_spatial_scale,
                    alpha=cfg.alpha).to(device)

    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.dec.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl_stack = StackLoader(2,
                           pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                           normalization=transf_normal)
    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization=transf_normal)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_single) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_single, frames_tnsr_brd),
                                 collate_fn=dl_single.collate_fn)

    dataloader_all_prev = DataLoader(dl_single,
                                     collate_fn=dl_single.collate_fn)

    dataloader_train = DataLoader(dl_stack,
                                  collate_fn=dl_stack.collate_fn,
                                  shuffle=True,
                                  num_workers=cfg.n_workers)

    dataloaders = {'train': dataloader_train,
                   'all_prev': dataloader_all_prev,
                   'prev': dataloader_prev}

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    couple_graphs = train(cfg, model, device, dataloaders, run_path)

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

if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
