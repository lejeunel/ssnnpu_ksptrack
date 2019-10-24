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

n_decs = 3
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

n_folds = 4
alpha = 0.3 #Transparency for std between curves
lw = 1
n_points = 2000

# Steps
f1_my_dict = dict()
f1_true_dict = dict()
# Make Inter-viewer frames
for key in rd.types:
    f1_my = []
    f1_true = []
    for fold in range(n_folds):
        path_ = os.path.join(rd.root_dir,
                            'learning_exps',
                            rd.learning_dirs_dict[key],
                            'scores_' + str(fold) + '.npz')
        print('Loading: ' + path_)

        scores = np.load(path_)

        f1_my.append(np.nanmax(scores['f1_my']))
        f1_true.append(np.nanmax(scores['f1_true']))
    f1_my_dict[key] = f1_my
    f1_true_dict[key] = f1_true
        # Make images/gts/gaze-point

df_my = pd.DataFrame.from_dict(f1_my_dict).T
df_my['mean'] = np.round_(df_my.mean(axis=1),decimals=n_decs)
df_true = pd.DataFrame.from_dict(f1_true_dict).T
df_true['mean'] = np.round_(df_true.mean(axis=1),decimals=n_decs)
df_all = pd.concat((df_my['mean'], df_true['mean']), axis=1)
df_all.columns = ['KSP', 'True']

print('Writing my scores to')
file_out_my = os.path.join(out_result_dir, 'learning_my.csv')
print(file_out_my)
df_my.to_csv(file_out_my)

print('Writing true scores to')
file_out_true = os.path.join(out_result_dir, 'learning_true.csv')
print(file_out_true)
df_true.to_csv(file_out_true)

print('Writing all scores to')
file_out_all = os.path.join(out_result_dir, 'learning_all.csv')
print(file_out_all)
df_all.loc[rd.types].to_csv(file_out_all)
