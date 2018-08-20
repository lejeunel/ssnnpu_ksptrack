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
import plot_results_ksp as pksp


dir_root = '/home/laurent.lejeune/medical-labeling'


dir_res_list = [
               #Tweez
               'Dataset00/results/2017-08-15_08-53-56_exp']

for i in range(len(dir_res_list)):
   dir_res = os.path.join(dir_root,dir_res_list[i])

   with open(os.path.join(dir_res, 'cfg.yml'), 'r') as outfile:
      conf = yaml.load(outfile)


   #list_ksp = np.load(os.path.join(conf.dataOutDir,'results.npz'))['list_ksp']
   #my_dataset = ds.Dataset(conf)
   #my_dataset.load_ss_from_file()
   #seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
   #f = 10
   #marked = seeds[seeds[:,0] == f, :][:,1]
   #ss.set_ratios(my_dataset.g_ss[f],marked)
   #conf.max_feats_ratio = 0.25

   pksp.main(conf)
