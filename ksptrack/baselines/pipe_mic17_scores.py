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


for key in rd.types:
    for dir_ in rd.res_dirs_dict_mic17[key]:

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

        res_files = sorted(glob.glob(os.path.join(path_,'*.png')))
        preds = np.asarray([utls.imread(f) for f in res_files])
        preds = np.mean(preds, axis=3)
        preds = preds.transpose((1,2,0))

        pr, rc, _ = precision_recall_curve(gt.ravel(),
                                           preds.ravel())
        all_f1s = 2 * (pr * rc) / (pr + rc)
        max_f1 = np.nanmax(all_f1s)
        max_pr = pr[np.argmax(all_f1s)]
        max_rc = rc[np.argmax(all_f1s)]
        file_out = os.path.join(path_, 'scores.csv')

        C = pd.Index(["F1", "PR", "RC"], name="columns")
        I = pd.Index(['EEL'], name="Methods")
        data = np.asarray([max_f1, max_pr, max_rc]).reshape(1, 3)
        df = pd.DataFrame(data=data, index=I, columns=C)
        print('Saving F1 score')
        df.to_csv(path_or_buf=file_out)

        print('Saving (resized?) predictions')
        np.savez(os.path.join(path_, 'preds.npz'),
                 **{'preds': preds})

        #l_dataset.gt
