#!/usr/bin/env python3
from torch.utils.data import DataLoader
from ksptrack.siamese.loader import Loader
import params
import os
from os.path import join as pjoin
import yaml
import numpy as np
from skimage import io
import clustering as clst
from ksptrack.siamese.tice import tice_wrapper
# from ksptrack.siamese.Kernel_MPE_grad_threshold import wrapper as kmwrapper
from ksptrack.siamese.modeling.siamese import Siamese
import torch
import pandas as pd
import tqdm
from skimage.measure import regionprops


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir)
    out_path = pjoin(run_path, 'pi_estim.txt')

    if not os.path.exists(out_path):

        device = torch.device('cuda' if cfg.cuda else 'cpu')

        model = Siamese(embedded_dims=cfg.embedded_dims,
                        cluster_number=cfg.n_clusters,
                        backbone=cfg.backbone)
        path_cp = pjoin(run_path, 'checkpoints', 'cp_autoenc.pth.tar')
        if (os.path.exists(path_cp)):
            print('loading checkpoint {}'.format(path_cp))
            state_dict = torch.load(path_cp,
                                    map_location=lambda storage, loc: storage)
            model.dec.autoencoder.load_state_dict(state_dict, strict=False)
        else:
            print('checkpoint {} not found. Train autoencoder first'.format(
                path_cp))
            return

        dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                    resize_shape=cfg.in_shape,
                    normalization='rescale')

        dl = DataLoader(dl, collate_fn=dl.collate_fn)
        res = clst.get_features(model, dl, device)

        features = np.concatenate(res['feats'])
        pos_mask = np.concatenate(res['labels_pos_mask'])

        truths = []
        truths_sp = []
        pbar = tqdm.tqdm(total=len(dl))
        for s in dl.dataset:
            labels = s['labels']
            truth = s['label/segmentation']
            regions = regionprops(labels + 1, intensity_image=truth)
            pos = np.array([p['mean_intensity'] > 0.5 for p in regions])[...,
                                                                         None]
            mapping = np.concatenate((np.unique(labels)[..., None], pos),
                                     axis=1)

            _, ind = np.unique(labels, return_inverse=True)
            truth_sp = mapping[ind, 1:].reshape(labels.shape)
            truths_sp.append(truth_sp)
            pbar.update(1)
        pbar.close()

        truths_sp = np.concatenate(truths_sp)
        print('true class-prior: {}'.format(truths_sp.sum() / truths_sp.size))

        print('estimating class-prior...')
        pi_estim = tice_wrapper(features, pos_mask, n_folds=5, n_iter=2)
        print('done.')
        print('class prior: {}'.format(pi_estim))

        with open(pjoin(run_path, 'pi_estim.txt'), 'w') as f:
            f.write('%f' % pi_estim)

    else:
        print('found file {}'.format(out_path))

        with open(out_path, 'r') as f:
            pi_estim = float(f.readline())

        print('pi_estim: {}'.format(pi_estim))

    return pi_estim


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
