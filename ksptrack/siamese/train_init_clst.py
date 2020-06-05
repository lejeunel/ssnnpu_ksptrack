from torch.utils.data import DataLoader
from ksptrack.siamese.loader import Loader
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese import im_utils
from ksptrack.siamese import utils as utls
from ksptrack.utils.bagging import calc_bagging
import numpy as np
from skimage import io
import clustering as clst
from ksptrack.utils.lfda import myLFDA
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ksptrack.utils.my_utils import sample_features
from sklearn.cluster import KMeans


def get_pls_transform(features, probas, thrs, n_samps, embedded_dims):
    pls = PLSRegression(n_components=embedded_dims)
    X, y = sample_features(features, probas, thrs, n_samps)
    pls.fit(X, y)

    return pls.coef_.T


def get_pca_transform(features, embedded_dims):
    clf = PCA(n_components=embedded_dims)
    clf.fit(features)
    return clf.components_.T


def get_lfda_transform(features,
                       probas,
                       thrs,
                       n_samps,
                       embedded_dims,
                       nn_scale=None,
                       embedding_type='orthonormalized'):

    lfda = myLFDA(n_components=embedded_dims,
                  n_components_prestage=embedded_dims,
                  k=nn_scale,
                  embedding_type=embedding_type)

    print('fitting LFDA with thrs: {}'.format(thrs))
    lfda.fit(features, probas, thrs, n_samps)

    return lfda.components_.T


def train_kmeans(model,
                 dataloader,
                 device,
                 n_clusters,
                 embedded_dims=30,
                 bag_t=300,
                 bag_max_depth=64,
                 bag_n_feats=0.02,
                 up_thr=0.8,
                 down_thr=0.2,
                 reduc_method='pls'):

    import pdb
    pdb.set_trace()  ## DEBUG ##
    features, pos_masks = clst.get_features(model, dataloader, device)

    import pdb
    pdb.set_trace()  ## DEBUG ##
    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)

    if (reduc_method == 'pca'):
        print('PCA: computing transform matrix')
        L = get_pca_transform(cat_features, embedded_dims)
    elif ((reduc_method == 'lfda') or (reduc_method == 'pls')):
        print('PLS/LFDA: computing probability map')
        probas = calc_bagging(cat_features,
                              cat_pos_mask,
                              bag_t,
                              bag_max_depth,
                              bag_n_feats,
                              bag_max_samples=500,
                              n_jobs=1)
        if (reduc_method == 'lfda'):

            print('LFDA: computing transform matrix')
            L = get_lfda_transform(cat_features, probas, [down_thr, up_thr],
                                   6000, embedded_dims)
        else:
            print('PLS: computing transform matrix')
            L = get_pls_transform(cat_features, probas, [0.8, 0.5], 2000,
                                  embedded_dims)

    print('fitting {} clusters with kmeans'.format(n_clusters))

    clf = KMeans(n_clusters=n_clusters, n_init=30)
    preds = clf.fit_predict(np.dot(cat_features, L))
    # preds = clf.fit_predict(cat_features)
    init_clusters = clf.cluster_centers_

    predictions = [utls.to_onehot(p, n_clusters).ravel() for p in preds]

    # split predictions by frame
    predictions = np.split(predictions,
                           np.cumsum([len(f) for f in features]))[:-1]
    return init_clusters, predictions, L, cat_features, cat_pos_mask


def train(cfg, model, device, dataloaders, run_path=None):

    if (run_path is not None):
        clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
        if (not os.path.exists(clusters_prevs_path)):
            os.makedirs(clusters_prevs_path)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    # train initial clustering
    if (not os.path.exists(init_clusters_path)):

        init_clusters, preds, L, feats, labels = train_kmeans(
            model,
            dataloaders['buff'],
            device,
            cfg.n_clusters,
            embedded_dims=cfg.embedded_dims,
            reduc_method=cfg.reduc_method,
            bag_t=cfg.bag_t,
            bag_n_feats=cfg.bag_n_feats,
            bag_max_depth=cfg.bag_max_depth,
            up_thr=cfg.ml_up_thr,
            down_thr=cfg.ml_down_thr)
        data = {'clusters': init_clusters, 'preds': preds, 'L': L}

        np.savez(init_clusters_path, **data)

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
    init_clusters = np.load(init_clusters_path, allow_pickle=True)['clusters']
    L = np.load(init_clusters_path, allow_pickle=True)['L']
    prev_ims = clst.do_prev_clusters_init(dataloaders['all_prev'], preds)

    # save initial clusterings to disk
    if (not os.path.exists(init_clusters_prev_path)):
        os.makedirs(init_clusters_prev_path)
        print('saving initial clustering previews to {}'.format(
            init_clusters_prev_path))
        for k, v in prev_ims.items():
            io.imsave(pjoin(init_clusters_prev_path, k), v)

    init_prev = np.vstack([prev_ims[k] for k in prev_ims.keys()])

    writer = SummaryWriter(run_path)

    writer.add_image('clusters', init_prev, 0, dataformats='HWC')

    L = torch.tensor(L).float().to(device)
    init_clusters = torch.tensor(init_clusters, dtype=torch.float).to(device)

    model.dec.set_clusters(init_clusters)
    model.dec.set_transform(L.T)
    path = pjoin(run_path, 'checkpoints')
    print('saving DEC with initial parameters to {}'.format(path))
    utls.save_checkpoint({
        'epoch': -1,
        'model': model
    },
                         False,
                         fname_cp='init_dec.pth.tar',
                         path=path)

    return model


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = Siamese(embedded_dims=cfg.embedded_dims,
                    cluster_number=cfg.n_clusters,
                    backbone=cfg.backbone)
    path_cp = pjoin(run_path, 'checkpoints', 'cp_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.dec.autoencoder.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                resize_shape=cfg.in_shape,
                normalization='rescale')

    frames_tnsr_brd = np.linspace(0,
                                  len(dl) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(dl, frames_tnsr_brd),
                                 collate_fn=dl.collate_fn)
    dataloader_all_prev = DataLoader(dl, collate_fn=dl.collate_fn)
    dataloader_train = DataLoader(dl,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=dl.collate_fn,
                                  drop_last=True,
                                  num_workers=cfg.n_workers)
    dataloader_buff = DataLoader(dl,
                                 collate_fn=dl.collate_fn,
                                 num_workers=cfg.n_workers)

    dataloaders = {
        'train': dataloader_train,
        'buff': dataloader_buff,
        'prev': dataloader_prev,
        'all_prev': dataloader_all_prev
    }

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
