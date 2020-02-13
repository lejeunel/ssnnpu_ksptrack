from loader import Loader
from torch.utils.data import DataLoader
import params
import torch
import os
from os.path import join as pjoin
import yaml
from tensorboardX import SummaryWriter
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese import im_utils
from ksptrack.siamese import utils as utls
from ksptrack.utils.bagging import calc_bagging
import numpy as np
from skimage import io
import clustering as clst
from ksptrack.siamese.my_kmeans import MyKMeans
from ksptrack.utils.lfda import myLFDA
import warnings


def get_lfda_transform(features,
                       probas,
                       thrs,
                       n_samps,
                       embedded_dims,
                       nn_scale,
                       embedding_type='weighted'):

    lfda = myLFDA(n_components=embedded_dims,
                  k=nn_scale,
                  embedding_type=embedding_type)
    if ((probas < thrs[0]).sum() < n_samps):
        sorted_probas = np.sort(probas)
        thrs[0] = sorted_probas[n_samps]
        warnings.warn('Not enough negatives. Setting thr to {}'.format(
            thrs[0]))
    if ((probas > thrs[1]).sum() < n_samps):
        sorted_probas = np.sort(probas)[::-1]
        thrs[1] = sorted_probas[n_samps]
        warnings.warn('Not enough positives. Setting thr to {}'.format(
            thrs[1]))
    lfda.fit(features, probas, thrs, n_samps)

    return lfda.components_.T


def train_kmeans(model,
                 dataloader,
                 device,
                 n_clusters,
                 embedded_dims=30,
                 nn_scaling=None,
                 bag_t=50,
                 bag_max_depth=5,
                 bag_n_feats=0.013):

    features, pos_masks = clst.get_features(model, dataloader, device)

    cat_features = np.concatenate(features)
    cat_pos_mask = np.concatenate(pos_masks)

    print('LFDA: computing probability map')
    probas = calc_bagging(cat_features,
                            cat_pos_mask,
                            bag_t,
                            bag_max_depth,
                            bag_n_feats,
                            bag_max_samples=500,
                            n_jobs=1)

    print('LFDA: computing transform matrix')
    L = get_lfda_transform(cat_features,
                           probas, [0.7, 0.3], 1000,
                           embedded_dims,
                           nn_scaling)

    print('forming {} initial clusters with kmeans'.format(n_clusters))

    kmeans = MyKMeans(n_clusters=n_clusters, use_locs=False)

    print('fitting...')
    preds, init_clusters = kmeans.fit_predict(np.dot(cat_features, L))
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
                                               embedded_dims=cfg.embedded_dims)
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

    model = DeepLabv3Plus(pretrained=True, num_classes=3)
    path_cp = pjoin(run_path, 'checkpoints', 'checkpoint_autoenc.pth.tar')
    if (os.path.exists(path_cp)):
        print('loading checkpoint {}'.format(path_cp))
        state_dict = torch.load(path_cp,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
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
