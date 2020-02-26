from loader import Loader, StackLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
import utils as utls
import tqdm
from ksptrack.siamese.modeling.dec import DEC
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.distrib_buffer import DistribBuffer
from ksptrack.siamese.train_init_clst import train_kmeans
from ksptrack.siamese.losses import LocationPairwiseLoss
from ksptrack.siamese import im_utils
import numpy as np
from skimage import io
import clustering as clst
from sklearn.metrics import f1_score
import pandas as pd
from ksptrack.utils.bagging import calc_bagging


def train_one_epoch(model, dataloaders, optimizers, device, distrib_buff,
                    all_edges_nn, probas, cfg):

    criterion_clust = LocationPairwiseLoss(thrs=[cfg.ml_down_thr,
                                                 cfg.ml_up_thr])
    criterion_recons = torch.nn.MSELoss()

    model.train()

    running_loss = 0.0
    running_loss_pur = 0.0
    running_loss_pw = 0.0
    running_loss_recons = 0.0

    pbar = tqdm.tqdm(total=len(dataloaders['train']))
    for i, data in enumerate(dataloaders['train']):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            res = model(data)

            distrib_buff.maybe_update(model, dataloaders['buff'], device)
            distribs, targets = distrib_buff[data['frame_idx']]

            optimizers['autoencoder'].zero_grad()
            optimizers['assign'].zero_grad()
            optimizers['transform'].zero_grad()

            # get edge array of consecutive frames
            # edges_nn = couple_graphs[data['frame_idx'][0]][data['frame_idx']
            #                                                [1]]['edges_nn']

            probas_ = torch.cat([probas[i] for i in data['frame_idx']])
            edges_nn = utls.combine_nn_edges(
                [all_edges_nn[i] for i in data['frame_idx']])
            loss_clust = criterion_clust(probas_, data, edges_nn,
                                         res['clusters'],
                                         targets)
            loss_recons = criterion_recons(res['output'], data['image'])

            loss = loss_clust['loss_clust']
            if(loss_clust['loss_pw'] is not None):
                loss += cfg.lambda_ * loss_clust['loss_pw']
            loss += cfg.gamma * loss_recons
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-10)
            optimizers['autoencoder'].step()
            optimizers['assign'].step()
            optimizers['transform'].step()

            running_loss += loss.cpu().detach().numpy()
            running_loss_pur += loss_clust['loss_clust'].cpu().detach(
            ).numpy()
            if(loss_clust['loss_pw'] is not None):
                running_loss_pw += loss_clust['loss_pw'].cpu().detach(
                ).numpy()
            running_loss_recons += loss_recons.cpu().detach().numpy()

            loss_ = running_loss / ((i + 1) * cfg.batch_size)
            loss_pur_ = running_loss_pur / ((i + 1) * cfg.batch_size)
            loss_pw_ = running_loss_pw / ((i + 1) * cfg.batch_size)
            loss_recons_ = running_loss_recons / ((i + 1) * cfg.batch_size)

        pbar.set_description('lss {:.6f}'.format(loss_))
        pbar.update(1)

    pbar.close()

    out = {
        'ratio_changed': distrib_buff.ratio_changed,
        'dec_loss_tot': loss_,
        'dec_loss_pur': loss_pur_,
        'dec_loss_recon': loss_recons_,
        'dec_loss_pw': loss_pw_
    }

    return out


def train(cfg, model, device, dataloaders, run_path):
    check_cp_exist = pjoin(run_path, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Loading.'.format(
            check_cp_exist))
        state_dict = torch.load(check_cp_exist,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    L = np.load(init_clusters_path, allow_pickle=True)['L']
    L = torch.tensor(L).float().to(device)
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    prev_ims = clst.do_prev_clusters_init(dataloaders['prev'], preds)

    # save initial clusterings to disk
    if (not os.path.exists(init_clusters_prev_path)):
        os.makedirs(init_clusters_prev_path)
        print('saving initial clustering previews...')
        for k, v in prev_ims.items():
            io.imsave(pjoin(init_clusters_prev_path, k), v)

    init_prev = np.vstack([prev_ims[k] for k in prev_ims.keys()])

    writer = SummaryWriter(run_path)

    # can choose order of cycles here
    distrib_buff = DistribBuffer(cfg.tgt_update_period,
                                 thr_assign=cfg.thr_assign)

    writer.add_image('clusters', init_prev, 0, dataformats='HWC')

    init_clusters = torch.tensor(init_clusters,
                                 dtype=torch.float)

    model.dec.set_clusters(init_clusters)
    model.dec.set_transform(L.T)

    if(cfg.skip_train_dec):
        print('skipping DEC training. ')
        path = pjoin(run_path, 'checkpoints')
        print('saving DEC with initial parameters to {}'.format(path))
        utls.save_checkpoint(
            {
                'epoch': -1,
                'model': model
            },
            False,
            fname_cp='init_dec.pth.tar',
            path=path)
        return

    else:
        best_loss = float('inf')
        print('training for {} epochs'.format(cfg.epochs_dec))

        model.to(device)

        optimizers = {
            'autoencoder':
            optim.SGD(params=[{
                'params': model.dec.autoencoder.parameters(),
                'lr': cfg.lr_assign / 10
            }]),
            'assign':
            optim.SGD(params=[{
                'params': model.dec.assignment.parameters(),
                'lr': cfg.lr_assign
            }]),
            'transform':
            optim.SGD(params=[{
                'params': model.dec.transform.parameters(),
                'lr': cfg.lr_assign / 10
            }])
        }

        lr_sch = {
            'autoencoder':
            torch.optim.lr_scheduler.ExponentialLR(optimizers['autoencoder'],
                                                   cfg.lr_power),
            'assign':
            torch.optim.lr_scheduler.ExponentialLR(optimizers['assign'],
                                                   cfg.lr_power),
            'transform':
            torch.optim.lr_scheduler.ExponentialLR(optimizers['transform'],
                                                   cfg.lr_power)
        }
        print('making NN graphs for all frames...')
        all_edges_nn = [
            utls.make_single_graph_nn_edges(s['graph'], device, cfg.nn_radius)
            # utls.make_single_graph_nn_edges(s['graph'], nn_radius=None)
            for s in dataloaders['buff'].dataset
        ]

# compute probability of all nodes
        
        features, pos_masks, _ = clst.get_features(model,
                                                   dataloaders['buff'],
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
        n_labels = [np.unique(s['labels']).size for s in dataloaders['buff'].dataset]
        probas = torch.split(probas, n_labels)

        for epoch in range(1, cfg.epochs_dec + 1):

            if (distrib_buff.converged):
                print('clustering assignment hit threshold. Ending training.')
                break

            print('epoch {}/{}'.format(epoch, cfg.epochs_dec))
            for phase in ['train', 'prev']:

                if phase == 'train':
                    if(epoch % cfg.proba_update_period == 0):
                        print('Updating foreground probabilities.')
                        features, pos_masks, _ = clst.get_features(model,
                                                                dataloaders['buff'],
                                                                device)
                        cat_features = np.concatenate(features)
                        cat_pos_mask = np.concatenate(pos_masks)
                        probas = calc_bagging(cat_features,
                                            cat_pos_mask,
                                            cfg.bag_t,
                                            bag_max_depth=64,
                                            bag_n_feats=None,
                                            bag_max_samples=500,
                                            n_jobs=1)
                        probas = torch.from_numpy(probas).to(device)
                        n_labels = [np.unique(s['labels']).size
                                    for s in dataloaders['buff'].dataset]
                        probas = torch.split(probas, n_labels)

                    res = train_one_epoch(model, dataloaders, optimizers,
                                          device, distrib_buff, all_edges_nn,
                                          probas, cfg)

                    lr_sch['autoencoder'].step()
                    lr_sch['assign'].step()
                    # write losses to tensorboard
                    for k, v in res.items():
                        writer.add_scalar(k, v, epoch)

                    # save checkpoint
                    if (epoch % cfg.cp_period == 0):
                        is_best = False
                        if (res['dec_loss_tot'] < best_loss):
                            is_best = True
                            best_loss = res['dec_loss_tot']
                        path = pjoin(run_path, 'checkpoints')
                        utls.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'model': model,
                                'best_loss': best_loss,
                            },
                            is_best,
                            fname_cp='checkpoint_dec.pth.tar',
                            fname_bm='best_dec.pth.tar',
                            path=path)
                else:
                    # save previews
                    if (epoch % cfg.prev_period == 0):
                        out_path = pjoin(clusters_prevs_path,
                                         'epoch_{:04d}'.format(epoch))
                        print('generating previews to {}'.format(out_path))
                        if (not os.path.exists(out_path)):
                            os.makedirs(out_path)
                        prev_ims = clst.do_prev_clusters(
                            model, device, dataloaders[phase])

                        for k, v in prev_ims.items():
                            io.imsave(pjoin(out_path, k), v)

                        # write previews to tensorboard
                        prev_ims_pt = np.vstack(
                            [im for im in prev_ims.values()])
                        writer.add_image('clusters',
                                         prev_ims_pt,
                                         epoch,
                                         dataformats='HWC')

    # save last clusterings to disk
    prev_ims = clst.do_prev_clusters(model, device, dataloaders['buff'])
    last_clusters_prev_path = pjoin(run_path, 'clusters_prevs', 'last')
    if (not os.path.exists(last_clusters_prev_path)):
        os.makedirs(last_clusters_prev_path)
        print('saving last clustering previews...')
        for k, v in prev_ims.items():
            io.imsave(pjoin(last_clusters_prev_path, k), v)


def eval(cfg, model, device, dataloader, run_path):
    """
    generate binary maps from clusters
    compute scores
    """

    check_cp_exist = pjoin(run_path, 'checkpoints', 'best_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Loading and skipping.'.format(
            check_cp_exist))
        state_dict = torch.load(check_cp_exist,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    prevs_path = pjoin(run_path, 'results')
    if (not os.path.exists(prevs_path)):
        os.makedirs(prevs_path)

    L = np.load(pjoin(run_path, 'init_clusters.npz'), allow_pickle=True)['L']
    L = torch.tensor(L).float().to(device)

    print('generating binary frames for evaluation')
    maps = {}
    truths = {}
    pbar = tqdm.tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)
        with torch.no_grad():
            res = model(data)
            labels = data['labels'].cpu().squeeze().numpy().astype(int)
            clusters = res['clusters']
            for l in data['label_keypoints'][0]:
                assigned_clst = torch.argmax(clusters,
                                             dim=1).cpu().squeeze().numpy()
                labels_in_assigned = [
                    i for i in np.unique(labels)
                    if (assigned_clst[i] == assigned_clst[l])
                ]
                map_ = np.array([
                    labels == lab_assigned
                    for lab_assigned in labels_in_assigned
                ]).sum(axis=0)
                maps[data['frame_name'][0]] = map_
                truths[data['frame_name'][0]] = data['label/segmentation'].cpu(
                ).squeeze().numpy()
        pbar.update(1)
    pbar.close()

    print('saving to {}'.format(prevs_path))
    for k, v in maps.items():
        io.imsave(pjoin(prevs_path, k), v)

    maps = np.array([maps[k] for k in sorted(maps.keys())])
    truths = np.array([truths[k] for k in sorted(truths.keys())])

    path = pjoin(run_path, 'scores.csv')
    print('computing scores to {}'.format(path))
    f1 = f1_score(truths.ravel(), maps.ravel())
    df = pd.Series({'f1': f1})
    df.to_csv(path)


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    roi_size=1,
                    roi_scale=cfg.roi_spatial_scale,
                    alpha=cfg.alpha)
    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.dec.autoencoder.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl_single = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                       normalization=transf_normal)
    # dl_stack = StackLoader(cfg.batch_size,
    #                        pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
    #                        normalization=transf_normal)
    # dataloader_train = DataLoader(dl_stack,
    #                               collate_fn=dl_stack.collate_fn,
    #                               shuffle=True,
    #                               num_workers=cfg.n_workers)
    dataloader_train = DataLoader(dl_single,
                                  batch_size=2,
                                  collate_fn=dl_single.collate_fn,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl_single) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(
        dl_single, frames_tnsr_brd),
                                 collate_fn=dl_single.collate_fn)
    dataloader_buff = DataLoader(dl_single,
                                 collate_fn=dl_single.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'buff': dataloader_buff,
        'prev': dataloader_prev
    }

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_path)
    # eval(cfg, model, device, dataloader_buff, run_path)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
