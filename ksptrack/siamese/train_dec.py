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
from siamese_sp.modeling.deeplab import DeepLabv3Plus
from siamese_sp.modeling.dec import DEC
from siamese_sp.cycle_scheduler import TrainCycleScheduler
from siamese_sp import im_utils
from sklearn.cluster import KMeans
from siamese_sp.my_kmeans import MyPCKMeans, MyKMeans
import numpy as np
from skimage import color, io
import glob


def do_prev_clusters_init(dataloader, init_clusters_path, out_path):

    prev_ims = {}
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)
        print('generating init clusters')
        predictions = np.load(init_clusters_path,
                              allow_pickle=True)['predictions']
        # form initial cluster centres
        pbar = tqdm.tqdm(total=len(dataloader.dataset))
        for data, preds in zip(dataloader.dataset, predictions):
            labels = data['labels']
            im = data['image_unnormal']
            all = im_utils.make_tiled_clusters(im, labels[..., 0], preds)
            prev_ims[data['frame_name']] = all
            io.imsave(pjoin(out_path, data['frame_name']), all)
            pbar.update(1)
        pbar.close()
    else:
        print('found init clusters at {}. loading images'.format(out_path))
        for f in sorted(glob.glob(pjoin(out_path, '*.png'))):
            prev_ims[f] = io.imread(f)

    return prev_ims


def do_prev_clusters(model, device, dataloader):

    model.eval()

    prevs = {}

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.no_grad():
            res = model(data)

        im = data['image_unnormal'].cpu().squeeze().numpy()
        im = np.rollaxis(im, 0, 3).astype(np.uint8)
        labels = data['labels'].cpu().squeeze().numpy()
        clusters = res['clusters'].cpu().squeeze().numpy()
        im = im_utils.make_tiled_clusters(im, labels, clusters)
        prevs[data['frame_name'][0]] = im

        pbar.update(1)
    pbar.close()

    return prevs


def do_prev_rags(model, device, dataloader):

    model.eval()

    prevs = {}

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.no_grad():
            res = model(data)

        probas = model.calc_all_probas(res['feats'], data['rag'])
        probas = [p.detach().cpu().squeeze().numpy() for p in probas][0]
        im = data['image_unnormal'].cpu().squeeze().numpy() / 255
        im = np.rollaxis(im, 0, 3)
        labels = data['labels'].cpu().squeeze().numpy()
        truth = data['label/segmentation'].cpu().squeeze().numpy()
        plot = im_utils.make_grid_rag(im,
                                      labels,
                                      data['rag'][0],
                                      probas,
                                      truth=truth)
        prevs[data['frame_name'][0]] = plot

        pbar.update(1)
    pbar.close()

    return prevs


def train_kmeans(model, dataloader, device, n_clusters, init_clusters_path,
                 use_locs):

    if (not os.path.exists(init_clusters_path)):

        print('forming {} initial clusters with kmeans'.format(cfg.n_clusters))

        if(cfg.with_pck):
            kmeans = MyPCKMeans(n_clusters=cfg.n_clusters)
        else:
            kmeans = MyKMeans(n_clusters=cfg.n_clusters)
            
        features = []
        labels_pos_mask = []

        model.eval()
        model.to(device)
        # form initial cluster centres
        pbar = tqdm.tqdm(total=len(dataloader))
        for index, data in enumerate(dataloader):
            data = utls.batch_to_device(data, device)
            with torch.no_grad():
                res = model(data)

            if (len(data['labels_clicked']) > 0):
                new_labels_pos = [
                    item for sublist in data['labels_clicked']
                    for item in sublist
                ]
                labels_pos_mask += [
                    True if l in new_labels_pos else False
                    for l in np.unique(data['labels'].cpu().numpy())
                ]
            feat = res['feats'].cpu().numpy()
            pbar.update(1)
            features.append(feat)
        pbar.close()

        # predictions = kmeans.fit_predict(features)
        print('fitting K-means...')
        cat_features = np.concatenate(features)
        kmeans.fit(cat_features, clicked_mask=labels_pos_mask)
        predictions = [
            utls.to_onehot(kmeans.predict(f), cfg.n_clusters) for f in features
        ]
        init_clusters = kmeans.cluster_centers_

        print('saving init clusters to {}'.format(init_clusters_path))
        np.savez(
            init_clusters_path, **{
                'init_clusters': init_clusters,
                'predictions': predictions
            })
    else:
        print('loading init clusters at {}'.format(init_clusters_path))
        init_clusters = np.load(init_clusters_path)['init_clusters']

    return init_clusters


def train_one_epoch(model, dataloader, optimizers, device, train_sch, cfg,
                    criterion_clust, criterion_recons, criterion_siam):

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

    pbar = tqdm.tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        data = utls.batch_to_device(data, device)

        # forward
        with torch.set_grad_enabled(True):
            # backward + optimize only if in training phase
            res = model(data)

            if (mode == 'feats'):
                optimizers['autoenc'].zero_grad()
                optimizers['assign'].zero_grad()
                target = target_distribution(res['clusters'])
                loss_clust = criterion_clust(res['clusters'].log(),
                                             target) / res['clusters'].shape[0]
                loss_recons = criterion_recons(res['recons'], data['image'])
                loss = cfg.gamma * loss_clust + (1 - cfg.gamma) * loss_recons
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
                X, Y = utls.sample_batch(data['rag'], res['clusters'],
                                         res['feats'], cfg.n_edges)
                Y_tilde = model.calc_probas(X)
                Y = Y.to(Y_tilde)
                loss = criterion_siam(Y_tilde, Y)
                loss.backward()
                optimizers['siam'].step()
                optimizers['autoenc'].step()
                running_loss += loss.cpu().detach().numpy()
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


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch**2) / torch.sum(batch, 0)
    weight = (weight.t() / torch.sum(weight, 1)).t()

    return weight


def train(cfg, model, device, dataloaders, run_dir):
    check_cp_exist = pjoin(run_dir, 'checkpoints', 'checkpoint_dec.pth.tar')
    if (os.path.exists(check_cp_exist)):
        print('found checkpoint at {}. Skipping.'.format(check_cp_exist))
        return

    rags_prevs_path = pjoin(run_dir, 'rags_prevs')
    clusters_prevs_path = pjoin(run_dir, 'clusters_prevs')
    if (not os.path.exists(rags_prevs_path)):
        os.makedirs(rags_prevs_path)
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    frames_tnsr_brd = np.linspace(0,
                                  len(dataloaders['prev']),
                                  num=cfg.n_ims_test,
                                  dtype=int)

    init_kmeans_path = pjoin(run_dir, 'init_clusters.npz')
    init_kmeans_prev_path = pjoin(clusters_prevs_path, 'init')
    init_clusters = train_kmeans(model, dataloaders['prev'], device,
                                 cfg.n_clusters, init_kmeans_path, True)

    prev_ims = do_prev_clusters_init(dataloaders['prev'], init_kmeans_path,
                                     init_kmeans_prev_path)

    init_prev = np.vstack([
        im for i, im in enumerate(prev_ims.values()) if (i in frames_tnsr_brd)
    ])

    writer = SummaryWriter(run_dir)

    train_sch = TrainCycleScheduler([cfg.epochs_dec, cfg.epochs_dist],
                                    cfg.epochs_all, ['feats', 'siam'])

    writer.add_image('{}'.format(train_sch.get_cycle()),
                     init_prev,
                     0,
                     dataformats='HWC')

    criterion_clust = torch.nn.KLDivLoss(reduction='sum')
    criterion_siam = torch.nn.BCEWithLogitsLoss()
    criterion_recons = torch.nn.MSELoss()

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

        losses = train_one_epoch(model, dataloaders['train'], optimizers,
                                 device, train_sch, cfg, criterion_clust,
                                 criterion_recons, criterion_siam)

        # write losses to tensorboard
        for k, v in losses.items():
            writer.add_scalar('loss_{}'.format(k), v, epoch)

        # save previews
        if (epoch > 1 and (epoch % cfg.prev_period == 0)):
            print('generating {} previews'.format(train_sch.get_cycle()))
            if (train_sch.get_cycle() == 'feats'):
                prev_ims = do_prev_clusters(model, device, dataloaders['prev'])
                out_path = clusters_prevs_path
            else:
                import pdb; pdb.set_trace() ## DEBUG ##
                prev_ims = do_prev_rags(model, device, dataloaders['prev'])
                out_path = rags_prevs_path

            out_path = pjoin(out_path, 'epoch_{:04d}'.format(epoch))
            os.makedirs(out_path)
            print('saving {} previews to {}'.format(train_sch.get_cycle(),
                                                    out_path))
            for k, v in tqdm.tqdm(prev_ims.items()):
                io.imsave(pjoin(out_path, k), v)

            # write previews to tensorboard
            prev_ims = np.vstack([
                im for i, im in enumerate(prev_ims.values())
                if (i in frames_tnsr_brd)
            ])
            writer.add_image('{}'.format(train_sch.get_cycle()),
                             prev_ims,
                             epoch,
                             dataformats='HWC')

        # save checkpoint
        if (train_sch.get_cycle() == 'siam'):
            is_best = False
            if (losses['siam'] < best_loss):
                is_best = True
                best_loss = losses['siam']
            path = pjoin(run_dir, 'checkpoints')
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

        train_sch.step()


def main(cfg):

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DEC(cfg.n_clusters, roi_size=1, roi_scale=cfg.roi_spatial_scale)

    path_cp = pjoin(cfg.out_dir, 'Dataset' + cfg.train_dir, 'checkpoints',
                    'checkpoint_autoenc.pth.tar')
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

    dl_train = Loader(
        pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
        # augmentation=transf,
        n_segments=cfg.n_segments_train,
        delta_segments=cfg.delta_segments_train,
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

    ds_dir = os.path.split('Dataset' + cfg.train_dir)[-1]

    run_dir = pjoin(cfg.out_dir, '{}'.format(ds_dir))

    if (not os.path.exists(run_dir)):
        os.makedirs(run_dir)

    # Save cfg
    with open(pjoin(run_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    train(cfg, model, device, dataloaders, run_dir)


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
