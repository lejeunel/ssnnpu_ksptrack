from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
from pygco import cut_from_graph
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
import networkx as nx


#dir_res = '/home/laurent.lejeune/medical-labeling/Dataset2/results/2017-07-21_11-39-17_exp'
dir_res = '/home/laurent.lejeune/medical-labeling/Dataset9/results/2017-08-01_19-24-33_exp'

with open(os.path.join(dir_res, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)


list_ksp = np.load(os.path.join(conf.dataOutDir,'results.npz'))['list_ksp']
#gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
my_dataset = ds.Dataset(conf)
#my_dataset.conf.ss_feature = ['desc']
#my_dataset.calc_sel_search(save=True)

my_dataset.load_ss_from_file()
my_dataset.load_labels_if_not_exist()
seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
#my_dataset.update_marked_sp(seeds,mode='foreground')



new_seeds = ss.thr_all_graphs(my_dataset.g_ss,seeds,0.5)


my_dataset.fg_marked = new_seeds
my_dataset.calc_pm(my_dataset.fg_marked,
                    save=False,
                    marked_feats=None,
                    all_feats_df=my_dataset.sp_desc_df,
                    in_type='not csv',
                    mode='foreground',
                bag_n_feats=conf.max_feats_ratio,
                    feat_fields=['desc'])
pm_scores_ss = my_dataset.get_pm_array(mode='foreground')

ksp_scores = utls.get_scores_from_sps(seeds,my_dataset.labels)
ksp_ss_scores = utls.get_scores_from_sps(new_seeds,my_dataset.labels)


l_dataset = learning_dataset.LearningDataset(conf,pos_thr=0.5)

f = 60
pm_thr = 0.8
gt = l_dataset.gt
cont_gt = segmentation.find_boundaries(gt[...,f],mode='thick')
idx_cont_gt = np.where(cont_gt)
im = utls.imread(conf.frameFileNames[f])
im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
im = gaze.drawGazePoint(conf.myGaze_fg,f,im,radius=7)
plt.subplot(231)
plt.imshow(ksp_scores[...,f])
plt.title('KSP')
plt.subplot(232)
plt.imshow(ksp_ss_scores[...,f])
plt.title('KSP+SS')
plt.subplot(233)
pm = plt.imshow(pm_scores_ss[...,f])
plt.title('KSP+SS PM')
plt.subplot(234)
plt.imshow(pm_scores_ss[...,f]>pm_thr)
plt.title('KSP+SS PM. thr = ' + str(pm_thr))
plt.subplot(235)
plt.imshow(im)
plt.title('image')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(pm, cax=cbar_ax)
plt.show()
