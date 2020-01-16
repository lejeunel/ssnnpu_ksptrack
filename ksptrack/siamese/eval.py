from os.path import join as pjoin
import os
import yaml
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from skimage import segmentation, draw, io, transform, measure
from skimage.measure import label
import configargparse
from imgaug import augmenters as iaa
from my_augmenters import rescale_augmenter, Normalize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import params
import numpy as np
from collections import defaultdict
from siamese_sp.siamese import Siamese
from siamese_sp.loader import Loader
import shutil


def main(cfg):

    cfg.batch_size = 1

    cp_path = pjoin(cfg.run_dir, 'checkpoints', 'best_model.pth.tar')

    cp = torch.load(cp_path, map_location=lambda storage, loc: storage)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    batch_to_device = lambda batch: {
        k: v.type(torch.float).to(device)
        if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    transf = iaa.Sequential([rescale_augmenter])
    transf_normal = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    model = Siamese()
    model.load_state_dict(cp)
    model = model.eval()

    eval_loaders = [
        Loader(pjoin(cfg.in_root, 'Dataset' + d),
               truth_type='hand',
               augmentation=transf,
               normalization=transf_normal) for d in cfg.sets
    ]

    eval_loaders = [
        DataLoader(l,
                   batch_size=cfg.batch_size,
                   collate_fn=l.collate_fn,
                   num_workers=cfg.n_workers) for l in eval_loaders
    ]

    for set_, l in zip(cfg.sets, eval_loaders):
        set_ = 'Dataset' + set_
        out_path = pjoin(cfg.run_dir, set_)
        out_fname = pjoin(out_path, 'sp_desc_siam.p')

        sp_labels = np.load(
            pjoin(cfg.in_root, set_, cfg.feats_path,
                  'sp_labels.npz'))['sp_labels']
        sp_labels = np.rollaxis(sp_labels, -1, 0)

        if (not os.path.exists(out_path)):
            os.makedirs(out_path)

        if (not os.path.exists(out_fname)):
            print('Generating features on {}'.format(set_))

            feats_sp = []
            pbar = tqdm.tqdm(total=len(l))
            for i, (data, labels) in enumerate(zip(l, sp_labels)):
                centroids = [(p['centroid'][1] / labels.shape[0],
                              p['centroid'][0] / labels.shape[1])
                             for p in measure.regionprops(labels)]

                data = batch_to_device(data)
                with torch.no_grad():
                    feat_ = model.feat_extr(data['image'])['out']
                for centroid, lab_ in zip(centroids,
                                          np.unique(labels)):
                    feats_sp.append(
                        (data['frame_idx'][0], int(lab_), centroid[0],
                         centroid[1],
                         model.sp_pool.pool(
                             feat_[0, :, labels == lab_],
                             dim=1).cpu().detach().numpy()[0, ...]))

                pbar.update(1)
            pbar.close()

            sp_desc = pd.DataFrame(
                feats_sp, columns=['frame', 'label', 'x', 'y', 'desc'])

            print('saving features to {}'.format(pjoin(out_path)))
            sp_desc.to_pickle(out_fname)

            out_path = pjoin(cfg.run_dir, set_, 'sp_labels.npz')
            in_path = pjoin(cfg.in_root, set_, cfg.feats_path, 'sp_labels.npz')
            print('copying sp labels to {}'.format(out_path))
            shutil.copyfile(in_path, out_path)

            out_path = pjoin(cfg.in_root, set_, cfg.feats_path)
            out_fname = pjoin(out_path, 'sp_desc_siam.p')
            print('copying features to {}'.format(pjoin(out_path)))
            sp_desc.to_pickle(out_fname)

        else:
            print('Features {} already exist'.format(out_fname))
            sp_desc = pd.read_pickle(out_fname)

        # make merge graphs
        out_path = pjoin(cfg.run_dir, set_, 'graphs')
        if(not os.path.exists(out_path)):
            os.makedirs(out_path)

        for i, (labels, fname) in enumerate(sp_labels, l.get_fnames()):
            graph_path = pjoin(out_path, os.path.splitext(fname)[0] + '.p')
            if(not os.path.exists(graph_path)):
                print('Computing merge graph {}/{}'.format(i+1, len(l)))
            else:
                print('Merge graph {} already exist'.format(graph_path))


if __name__ == "__main__":

    p = params.get_params()

    p.add('--run-dir', required=True)
    p.add('--in-root', required=True)
    p.add('--feats-path', default='precomp_desc')
    p.add('--sets', nargs='+', type=str, required=True)

    cfg = p.parse_args()

    main(cfg)
