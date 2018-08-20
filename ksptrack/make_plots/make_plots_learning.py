from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
import my_utils as utls
import plot_results_ksp_simple_multigaze as pksp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learning_dataset as ld
from skimage import (color, segmentation, util, transform)
import gazeCsv as gaze
import results_dirs as rd
import shutil as sh

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

n_folds = 4
alpha = 0.3 #Transparency for std between curves
lw = 1
n_points = 2000

# Steps

# Make Inter-viewer frames
pr_my = []
rc_my = []
rc_true = []
pr_true = []
f1_my = []
f1_true = []
for key in rd.learning_dirs_dict.keys():
    for fold in range(n_folds):
        path_ = os.path.join(rd.root_dir,
                            'learning_exps',
                            rd.learning_dirs_dict[key],
                            'scores_' + str(fold) + '.npz')
        print('Loading: ' + path_)

        scores = np.load(path_)
        pr_my.append(scores['pr_my'])
        rc_my.append(scores['rc_my'])
        pr_true.append(scores['pr_true'])
        rc_true.append(scores['rc_true'])
        f1_my.append(np.max(scores['f1_my']))
        f1_true.append(np.max(scores['f1_true']))
        # Make images/gts/gaze-point

    rc_my_interp, pr_my_interp = utls.concat_interp(rc_my, pr_my, n_points)
    rc_true_interp, pr_true_interp = utls.concat_interp(rc_true, pr_true, n_points)
    conf = scores['conf'].item()

    #plt.subplot(121)

    plt.plot(
        rc_my_interp,
        pr_my_interp.mean(axis=0),
        'r-',
        lw=lw,
        label='KSP')
    plt.fill_between(rc_my_interp,
                    pr_my_interp.mean(axis=0)+pr_my_interp.std(axis=0),
                    pr_my_interp.mean(axis=0)-pr_my_interp.std(axis=0),
                    facecolor='r',
                    alpha=alpha)
    plt.hold(True)
    #plt.subplot(122)
    plt.plot(
        rc_true_interp,
        pr_true_interp.mean(axis=0),
        'b-',
        lw=lw,
        label='GT')
    plt.fill_between(rc_true_interp,
                    pr_true_interp.mean(axis=0)+pr_true_interp.std(axis=0),
                    pr_true_interp.mean(axis=0)-pr_true_interp.std(axis=0),
                    facecolor='b',
                    alpha=alpha)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(conf.seq_type)
    plt.legend()
    file_out = os.path.join(out_result_dir,conf.seq_type+'_learning.png')
    plt.savefig(file_out, dpi=400)
    plt.clf()
