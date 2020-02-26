from loader import Loader, StackLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese import utils as utls
from ksptrack.siamese import im_utils
from ksptrack.siamese.losses import LabelPairwiseLoss
import numpy as np
from skimage import io
import clustering as clst
from ksptrack.utils.bagging import calc_bagging


def make_constraints(edges_nn, clusters, pos_clusters):
    edges = {'sim': [], 'disim': []}
    # relabel clusters
    for c in pos_clusters:
        clusters[clusters == c] = min(pos_clusters)

    # get edges with both nodes on positive cluster
    edges_mask = (clusters[edges_nn[:, 0]] == min(pos_clusters))
    edges_mask *= (clusters[edges_nn[:, 1]] == min(pos_clusters))
    edges['sim'] = edges_nn[edges_mask, :]

    # get edges with a single positive nodes on positive cluster (negative set)
    edges_mask_0 = (clusters[edges_nn[:, 0]] == min(pos_clusters))
    edges_mask_0 *= (clusters[edges_nn[:, 1]] != min(pos_clusters))
    edges_mask_1 = (clusters[edges_nn[:, 1]] == min(pos_clusters))
    edges_mask_1 *= (clusters[edges_nn[:, 0]] != min(pos_clusters))
    edges['disim'] = edges_nn[edges_mask_0 + edges_mask_1, :]

    return edges


def train_one_epoch(model, dataloaders, couple_graphs, optimizers, device,
                    probas, lr_sch, cfg):

    model.train()

    running_loss = 0.0

    criterion = LabelPairwiseLoss(thrs=[cfg.ml_down_thr, cfg.ml_up_thr])
    criterion_recons = torch.nn.MSELoss()
    pbar = tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):

            # get edge array of consecutive frames
            edges_nn = couple_graphs[data['frame_idx'][0]][data['frame_idx']
                                                           [1]]['edges_nn']
            res = model(data)

            # get cluster assignments of respective nodes
            clst = couple_graphs[data['frame_idx'][0]][data['frame_idx']
                                                       [1]]['clst']
            optimizers['feats'].zero_grad()
            optimizers['transform'].zero_grad()
            probas_ = torch.cat([probas[i] for i in data['frame_idx']])
            loss = criterion(edges_nn,
                             probas_,
                             res['proj_pooled_aspp_feats'],
                             clusters=clst)
            loss += cfg.gamma * criterion_recons(res['output'], data['image'])

            loss.backward()
            optimizers['feats'].step()
            optimizers['transform'].step()
            running_loss += loss.cpu().detach().numpy()
            loss_ = running_loss / ((i + 1) * cfg.batch_size)
            lr_sch['feats'].step()
            lr_sch['transform'].step()

        pbar.set_description('lss {:.6f}'.format(loss_))
        pbar.update(1)

    pbar.close()

    out = {'loss_siam': loss_}

    return out


def train(cfg, model, device, dataloaders, run_path):

    features, pos_masks, _ = clst.get_features(model, dataloaders['all_prev'],
                                               device)
    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)
    print('computing probability map')
    probas = calc_bagging(cat_features,
                          cat_pos_mask,
                          cfg.bag_t,
                          bag_max_depth=64,
                          bag_n_feats=None,
                          bag_max_samples=500,
                          n_jobs=1)
    probas = torch.from_numpy(probas).to(device)
    n_labels = [
        np.unique(s['labels']).size for s in dataloaders['all_prev'].dataset
    ]
    probas = torch.split(probas, n_labels)

    if (cfg.skip_train_dec):
        cp_fname = 'checkpoint_siam.pth.tar'
        best_cp_fname = 'best_siam.pth.tar'
        path_ = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
        state_dict = torch.load(path_,
                                map_location=lambda storage, loc: storage)
        print('skipped clustering step.')
        print('loading checkpoint {}'.format(path_))
        model.load_state_dict(state_dict)
    else:
        cp_fname = 'checkpoint_siam_dec.pth.tar'
        best_cp_fname = 'best_siam_dec.pth.tar'
        path_ = pjoin(run_path, 'checkpoints', 'best_dec.pth.tar')
        print('did clustering step.')
        print('loading checkpoint {}'.format(path_))
        model.load_state_dict(state_dict)

    check_cp_exist = pjoin(run_path, 'checkpoints', best_cp_fname)
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    couple_graphs = utls.make_all_couple_graphs(model,
                                                device,
                                                dataloaders['train'],
                                                cfg.nn_radius,
                                                do_inter_frame=True)

    rags_prevs_path = pjoin(run_path, 'rags_prevs')
    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)

    writer = SummaryWriter(run_path)

    best_loss = float('inf')
    print('training for {} epochs'.format(cfg.epochs_dist))

    model.to(device)

    optimizers = {
        'feats':
        optim.SGD(params=[{
            'params': model.dec.autoencoder.parameters(),
            'lr': cfg.lr_dist / 100,
        }],
                  momentum=cfg.momentum,
                  weight_decay=cfg.decay),
        'transform':
        optim.SGD(params=[{
            'params': model.dec.transform.parameters(),
            'lr': cfg.lr_dist,
        }])
    }
    lr_sch = {'feats': torch.optim.lr_scheduler.ExponentialLR(optimizers['feats'],
                                                              cfg.lr_power),
              'transform': torch.optim.lr_scheduler.ExponentialLR(optimizers['transform'],
                                                                  cfg.lr_power)}

    for epoch in range(1, cfg.epochs_dist + 1):

        print('epoch {}/{}'.format(epoch, cfg.epochs_dist))
        for phase in ['train', 'prev']:
            if phase == 'train':
                res = train_one_epoch(model, dataloaders, couple_graphs,
                                      optimizers, device, probas, lr_sch, cfg)

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
                        fname_cp=cp_fname,
                        fname_bm=best_cp_fname,
                        path=path)

            else:

                # save previews
                if (epoch % cfg.prev_period == 0):
                    out_path = pjoin(rags_prevs_path,
                                     'epoch_{:04d}'.format(epoch))
                    print('generating previews to {}'.format(out_path))

                    if (not os.path.exists(out_path)):
                        os.makedirs(out_path)

                    prev_ims = clst.do_prev_rags(model, device,
                                                 dataloaders['prev'],
                                                 couple_graphs)

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

    dataloaders = {
        'train': dataloader_train,
        'all_prev': dataloader_all_prev,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)

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
