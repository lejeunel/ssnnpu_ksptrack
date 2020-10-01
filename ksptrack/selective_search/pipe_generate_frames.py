import datetime
import os
import networkx as nx
from os.path import join as pjoin
import yaml
import tqdm
from skimage import io, draw, segmentation
import numpy as np
import configargparse
import selective_search
import glob
import pandas as pd
from unet_region import utils as utls
from unet_region.loader import Loader
import matplotlib.pyplot as plt
import selective_search
from unet_region.baselines.selective_search import generate_frames
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('-v', help='verbose', action='store_true')

    #Paths, dirs, names ...
    p.add('--root-data-dir', type=str, required=True)
    p.add('--root-run-dir', type=str, required=True)
    p.add('--csv-fname', type=str, required=True)
    p.add('--sets', nargs='+', required=True)
    p.add('--thr', type=float, default=0.9)

    cfg = p.parse_args()

    for s in cfg.sets:
        cfg.csv_path = pjoin(cfg.root_data_dir, 'Dataset'+s, 'gaze-measurements', cfg.csv_fname)
        cfg.data_dir = pjoin(cfg.root_data_dir, 'Dataset'+s)
        cfg.run_dir = pjoin(cfg.root_run_dir, 'Dataset'+s)

        cfg = generate_frames.main(cfg)

        pred_frames = sorted(glob.glob(pjoin(cfg.run_dir,
                                             cfg.csv_fname,
                                             '*.png')))

        # compute scores
        loader = Loader(cfg.data_dir, 'hand')
        truths = []
        preds = []
        for sample, p in zip(loader, pred_frames):
            truths.append(sample['label/segmentation'])
            preds.append(io.imread(p) / 255)

        truths = np.array(truths)
        preds = np.array(preds)

        fpr, tpr, _ = roc_curve(truths.ravel(), preds.ravel())
        precision, recall, _ = precision_recall_curve(truths.ravel(),
                                                      preds.ravel())
        f1 = (2 * (precision * recall) / (precision + recall))
        argmax = np.argmax(f1)
        auc_ = auc(fpr, tpr)

        data = {'f1': f1.max(),
                'auc': auc_,
                'fpr': fpr[argmax],
                'tpr': tpr[argmax],
                'pr': precision[argmax],
                'rc': recall[argmax]}

        df = pd.Series(data)
        df.to_csv(pjoin(cfg.run_dir, cfg.csv_fname, 'scores.csv'))
