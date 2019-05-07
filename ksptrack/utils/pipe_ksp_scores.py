import os
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
from skimage.transform import resize
import glob
from labeling.exps import results_dirs as rd
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.cfgs import cfg

flat_dict_ksps = [[rd.res_dirs_dict_ksp[k],
                  rd.res_dirs_dict_ksp_overfeat[k],
                  rd.res_dirs_dict_ksp_rec[k]] for k in rd.types]
for i in range(3):
    flat_dict_ksps = [item for sublist in flat_dict_ksps for item in sublist]

for dir_ in flat_dict_ksps:

    print('Scoring:')
    print(dir_)
    path_ = os.path.join(rd.root_dir,
                            dir_)

    # Get config
    conf = cfg.load_and_convert(os.path.join(path_, 'cfg.yml'))

    conf.root_path = rd.root_dir
    conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
    l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
    gt = l_dataset.gt

    np_file = np.load(os.path.join(path_, 'metrics.npz'))
    pr_ksp = np_file['pr_ksp'][1]
    rc_ksp = np_file['rc_ksp'][1]
    f1_ksp = 2 * (pr_ksp * rc_ksp) / (pr_ksp + rc_ksp)

    pr_pm = np_file['pr_pm']
    rc_pm = np_file['rc_pm']

    f1_pm = np.nan_to_num(2 * (pr_pm * rc_pm) / (pr_pm + rc_pm))
    max_f1_pm = np.nanmax(f1_pm)
    max_pr_pm = pr_pm[0, np.argmax(f1_pm)]
    max_rc_pm = rc_pm[0, np.argmax(f1_pm)]
    file_out = os.path.join(path_, 'scores.csv')

    C = pd.Index(["F1", "PR", "RC"], name="columns")
    I = pd.Index(['KSP', 'KSP/PM'], name="Methods")
    data = np.asarray([f1_ksp, pr_ksp, rc_ksp,
                       max_f1_pm, max_pr_pm, max_rc_pm]).reshape(2, 3)
    df = pd.DataFrame(data=data, index=I, columns=C)
    print('Saving F1 score')
    df.to_csv(path_or_buf=file_out)
