import os
import yaml
from labeling.cfgs import cfg
import pandas as pd
import numpy as np
import labeling.graph_tracking as gtrack
from labeling.utils import my_utils as utls
from labeling.utils.data_manager import DataManager
import logging
import labeling.sp_manager as spm
from sklearn.metrics import (f1_score,
                             roc_curve)
from labeling.utils import plot_results_ksp_simple as pksp
import pickle
from labeling.exps import results_dirs as rd
from labeling.utils import learning_dataset
from labeling import tr
from labeling import tr_manager
import munch

#dict_ = pickle.load(open('g_for_dict', 'rb'))
#dict_['forward_tracklets']
#dict_['forward_sets']

#paths = utls.tracklet_set_to_sp_path(dict_['forward_tracklets'],
#                                     dict_['forward_sets'],
#                                     iter_=0)

root_dir = os.path.join('/home/laurent.lejeune/medical-labeling/',
                        'Dataset00/results')

#exp_dir = '2018-05-31_09-55-01_exp'
#exp_dir = '2018-05-31_09-55-40_exp'
exp_dir = '2018-05-31_09-56-40_exp'

conf = cfg.load_and_convert(os.path.join(root_dir, exp_dir, 'cfg.yml'))
pksp.main(conf)

#dataset = learning_dataset.LearningDataset(conf)
#labels = dataset.get_labels()

#im = [utls.imread(f) for f in conf.frameFileNames]
#res = np.load(os.path.join(dir_, 'results.npz'))
#list_paths_back = res['list_paths_back']
#list_paths_for = res['list_paths_for']
#seeds = utls.list_paths_to_seeds(list_paths_for,
#                                 list_paths_back)
#seeds += list(set([p.tolist() for p in list_paths_back[-1]]))
#seeds += list(set([p.tolist() for p in list_paths_for[-1]]))
