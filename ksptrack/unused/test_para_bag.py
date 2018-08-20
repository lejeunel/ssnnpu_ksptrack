from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import cfg
import pandas as pd
import pickle as pk
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
import superPixels as spix
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import graphtracking as gtrack
import my_utils as utls
import dataset as ds
import selective_search as ss
import shutil as sh
import learning_dataset
import logging
from skimage import filters
import superpixelmanager as spm
import networkx as nx


def main(arg_cfg):
    data = dict()

    #Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['dataSetDir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)

    #Set logger

    conf.myGaze_fg = utls.readCsv(os.path.join(conf.dataInRoot,
                                                conf.dataSetDir,
                                                conf.gazeDir,
                                                conf.csvFileName_fg))


    if (conf.labelMatPath != ''):
        conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.dataSetDir, conf.frameDir,
                                    conf.labelMatPath)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.dataSetDir,
                                    conf.feats_files_dir)

    # ---------- Descriptors/superpixel costs
    my_dataset = ds.Dataset(conf)
    #my_dataset.flows_mat_to_np()

    my_dataset.calc_pm(utls.pandas_to_std_csv(                                         conf.myGaze_fg),
                       save=True,
                       marked_feats=None,
                       all_feats_df=my_dataset.get_sp_desc_from_file(),
                       in_type='csv_normalized',
                       max_n_feats=conf.max_n_feats,
                       mode='foreground',
                       feat_fields=['desc'],
                       n_jobs=8)

    fg_pm_para = my_dataset.fg_pm_df.copy()
    pm_para = my_dataset.get_pm_array(frames=[0, 1, 2])

    my_dataset.calc_pm(utls.pandas_to_std_csv(                                         conf.myGaze_fg),
                       save=True,
                       marked_feats=None,
                       all_feats_df=my_dataset.get_sp_desc_from_file(),
                       in_type='csv_normalized',
                       max_n_feats=conf.max_n_feats,
                       mode='foreground',
                       feat_fields=['desc'],
                       n_jobs=1)

    fg_pm_single = my_dataset.fg_pm_df.copy()
    pm_single = my_dataset.get_pm_array(frames=[0, 1, 2])
