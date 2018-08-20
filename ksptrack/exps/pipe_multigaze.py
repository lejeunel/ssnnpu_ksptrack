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

# Steps
make_plots_metrics = False
make_tables = False
make_frames = True

# Run plots, metrics, etc..
if(make_plots_metrics):
    for key in rd.res_dirs_dict_ksp.keys():
        for dset in range(len(rd.res_dirs_dict_ksp[key])):
            confs = [rd.confs_dict_ksp[key][dset][g] for g in range(5)]

            logger = logging.getLogger('test_self_learning')

            if(rd.out_dirs_dict_ksp[key][dset] is not None):
                out_dir = os.path.join(rd.root_dir,
                                      'learning_exps',
                                      rd.out_dirs_dict_ksp[key][dset])
            else:
                now = datetime.datetime.now()
                dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
                out_dir = os.path.join(confs[0].dataOutRoot,
                                       'learning_exps',
                                      'learning_multigaze_' +
                                      confs[0].seq_type + '_' + dateTime)
                if(not os.path.exists(out_dir)):
                    os.mkdir(out_dir)
            utls.setup_logging(out_dir)
            pksp.main(out_dir, confs, logger=logger)
