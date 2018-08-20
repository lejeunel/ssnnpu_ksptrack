from sklearn.metrics import (f1_score, precision_recall_curve)
from skimage import (color, segmentation)
import os
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd
from labeling.cfgs import cfg

"""
Calculate F1 score for ratio of positive pixels in superpixels
"""
save_path = os.path.join(rd.root_dir, 'plots_results', 'sp_thr_f1s.npz')

ratios = [0.25, 0.5, 0.75, 1.0]
res = dict()

# Self-learning
for key in rd.types:
    res[key] = dict()
    for seq in rd.res_dirs_dict_ksp[key][0]:
        f1s = list()

        cfg_ = cfg.load_and_convert(os.path.join(rd.root_dir,
                                seq,
                                'cfg.yml'))

        dset = learning_dataset.LearningDataset(cfg_)
        gt = dset.gt
        sp_gt = dset.make_y_map_true(gt)

        for r in ratios:
            f1 = f1_score(gt.ravel(), (sp_gt >= r).ravel())
            f1s.append(f1)

        res[key][cfg_.dataSetDir] = f1s

data = {'res': res, 'ratios': ratios}
np.savez(save_path, **data)
