import os
from os.path import join as pjoin

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import params
from ksptrack.pu.im_utils import get_features
from ksptrack.pu.loader import Loader
from ksptrack.pu.modeling.unet import UNet
from ksptrack.utils.bagging import calc_bagging
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import pandas as pd

torch.backends.cudnn.benchmark = True


def main(cfg):

    run_path = pjoin(cfg.out_root, cfg.run_dir, 'bagging')
    scores_path = pjoin(run_path, 'scores.csv')

    if os.path.exists(scores_path):
        print('found directory {}. Delete to re-run'.format(scores_path))
        return

    if (not os.path.exists(run_path)):
        os.makedirs(run_path)

    # Save cfg
    with open(pjoin(run_path, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    device = torch.device('cuda' if cfg.cuda else 'cpu')

    dl = Loader(pjoin(cfg.in_root, 'Dataset' + cfg.train_dir),
                normalization='rescale',
                resize_shape=cfg.in_shape,
                sp_labels_fname='sp_labels.npy')

    dl = DataLoader(dl, collate_fn=dl.collate_fn)

    model = UNet(depth=5, skip_mode='none')

    path_ = pjoin(os.path.split(run_path)[0], 'autoenc', 'cp.pth.tar')
    print('loading checkpoint {}'.format(path_))
    state_dict = torch.load(path_, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)

    res = get_features(model, dl, device)
    feats = np.concatenate(res['feats'])
    y = np.concatenate(res['labels_pos_mask'])
    y_true = np.concatenate(res['truths'])

    probas = calc_bagging(feats,
                          y,
                          T=cfg.bag_t,
                          bag_max_depth=cfg.bag_max_depth,
                          bag_n_feats=cfg.bag_n_feats)
    pred_labels = probas > 0

    fpr, tpr, _ = roc_curve(y_true, pred_labels)
    precision, recall, _ = precision_recall_curve(y_true, pred_labels)
    f1 = (2 * (precision * recall) / (precision + recall)).max()
    auc_ = auc(fpr, tpr)
    scores = dict()
    scores['f1'] = f1
    scores['auc'] = auc_
    scores['fpr'] = fpr[1]
    scores['tpr'] = tpr[1]
    df = pd.Series(scores)
    df.to_csv(scores_path)

    return model


if __name__ == "__main__":

    p = params.get_params()

    p.add('--out-root', required=True)
    p.add('--in-root', required=True)
    p.add('--train-dir', required=True)
    p.add('--run-dir', required=True)

    cfg = p.parse_args()

    main(cfg)
