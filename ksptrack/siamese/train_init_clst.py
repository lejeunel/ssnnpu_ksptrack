from loader import Loader
from torch.utils.data import DataLoader
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from ksptrack.siamese.modeling.dec import DEC
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

    lfda.fit(features, probas, thrs, n_samps)

    return lfda.components_.T


def train_kmeans(model,
                 dataloader,
                 device,
                 n_clusters,
                 embedded_dims=30,
                 nn_scaling=None,
                 bag_t=300,
                 bag_max_depth=64,
                 bag_n_feats=0.013,
                 up_thr=0.8,
                 down_thr=0.2,
                 reduc_method='pls',
                 init_clusters=None):

    features, pos_masks, _ = clst.get_features(model, dataloader, device)

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
                              None,
                              bag_max_samples=500,
                              n_jobs=1)
        if (reduc_method == 'lfda'):

            print('LFDA: computing transform matrix')
            L = get_lfda_transform(cat_features, probas,
                                   [down_thr, up_thr], 6000,
                                   embedded_dims)
        else:
            print('PLS: computing transform matrix')
            L = get_pls_transform(cat_features, probas, [0.8, 0.5], 2000,
                                  embedded_dims)

    print('fitting {} clusters with kmeans'.format(n_clusters))

    kmeans = KMeans(n_clusters=n_clusters,
                    init=init_clusters if init_clusters is not None else 'k-means++')
    preds = kmeans.fit_predict(np.dot(cat_features, L))
    init_clusters = kmeans.cluster_centers_
    predictions = [utls.to_onehot(p, n_clusters).ravel() for p in preds]

    # split predictions by frame
    predictions = np.split(predictions,
                           np.cumsum([len(f) for f in features]))[:-1]
    return init_clusters, predictions, L


def train(cfg, model, device, dataloaders, run_path):

    clusters_prevs_path = pjoin(run_path, 'clusters_prevs')
    if (not os.path.exists(clusters_prevs_path)):
        os.makedirs(clusters_prevs_path)

    init_clusters_path = pjoin(run_path, 'init_clusters.npz')
    init_clusters_prev_path = pjoin(clusters_prevs_path, 'init')

    # train initial clustering
    if (not os.path.exists(init_clusters_path)):

        init_clusters, preds, L = train_kmeans(model,
                                               dataloaders['buff'],
                                               device,
                                               cfg.n_clusters,
                                               embedded_dims=cfg.embedded_dims,
                                               reduc_method=cfg.reduc_method,
                                               bag_t=cfg.bag_t,
                                               up_thr=cfg.ml_up_thr,
                                               down_thr=cfg.ml_down_thr)
        np.savez(init_clusters_path, **{
            'clusters': init_clusters,
            'preds': preds,
            'L': L
        })

    preds = np.load(init_clusters_path, allow_pickle=True)['preds']
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

    writer.add_image('clusters', init_prev, 0, dataformats='HWC')


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    model = DEC(embedding_dims=cfg.n_clusters,
                cluster_number=cfg.n_clusters,
                roi_size=cfg.roi_output_size,
                roi_scale=cfg.roi_spatial_scale,
                alpha=cfg.alpha)
    path_cp = pjoin(run_path, 'checkpoints', 'best_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.autoencoder.load_state_dict(state_dict)
    else:
        print(
            'checkpoint {} not found. Train autoencoder first'.format(path_cp))
        return

    _, transf_normal = im_utils.make_data_aug(cfg)

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                normalization=transf_normal)

    frames_tnsr_brd = np.linspace(0,
                                  len(dl) - 1,
                                  num=cfg.n_ims_test,
                                  dtype=int)

    dataloader_prev = DataLoader(torch.utils.data.Subset(dl, frames_tnsr_brd),
                                 collate_fn=dl.collate_fn)
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
        'prev': dataloader_prev
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
