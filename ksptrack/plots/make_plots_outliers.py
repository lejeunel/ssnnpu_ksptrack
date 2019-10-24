from skimage.color import color_dict
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.exps import results_dirs as rd
from labeling.utils import learning_dataset as ld
from labeling.utils import my_utils as utls
from labeling.utils import plot_results_ksp_simple as pksp
from labeling.cfgs import cfg
from skimage import (color, segmentation, util,transform,io)
from skimage.color import color_dict

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

# Steps
file_out = os.path.join(out_result_dir, 'missing.png')

f1_list = []

seq = 'Dataset00'
missing_f1 = dict()
for c in rd.res_dirs_dict_ksp_miss[seq].keys():
    path_ = os.path.join(rd.root_dir,
                         seq,
                         'results',
                         rd.res_dirs_dict_ksp_miss['Dataset00'][c])
    conf = cfg.load_and_convert(os.path.join(path_, 'cfg.yml'))
    print('Loading: ' + path_)
    df_score = pd.read_csv(os.path.join(path_, 'scores.csv'))
    missing_f1[c] = df_score[(df_score['Methods'] == 'KSP')]['F1'][0]

neigh_5_f1 = dict()
dist = 5
for c in rd.res_dirs_dict_ksp_uni_neigh[seq][dist].keys():
    path_ = os.path.join(rd.root_dir,
                         seq,
                         'results',
                         rd.res_dirs_dict_ksp_uni_neigh[seq][dist][c])
    #conf = cfg.load_and_convert(os.path.join(path_, 'cfg.yml'))
    print('Loading: ' + path_)
    df_score = pd.read_csv(os.path.join(path_, 'scores.csv'))
    neigh_5_f1[c] = df_score[(df_score['Methods'] == 'KSP')]['F1'][0]

neigh_10_f1 = dict()
dist = 10
for c in rd.res_dirs_dict_ksp_uni_neigh[seq][dist].keys():
    path_ = os.path.join(rd.root_dir,
                         seq,
                         'results',
                         rd.res_dirs_dict_ksp_uni_neigh[seq][dist][c])
    #conf = cfg.load_and_convert(os.path.join(path_, 'cfg.yml'))
    print('Loading: ' + path_)
    df_score = pd.read_csv(os.path.join(path_, 'scores.csv'))
    neigh_10_f1[c] = df_score[(df_score['Methods'] == 'KSP')]['F1'][0]

all_bg_f1 = dict()
for c in rd.res_dirs_dict_ksp_uni_bg[seq].keys():
    path_ = os.path.join(rd.root_dir,
                         seq,
                         'results',
                         rd.res_dirs_dict_ksp_uni_bg[seq][c])
    #conf = cfg.load_and_convert(os.path.join(path_, 'cfg.yml'))
    print('Loading: ' + path_)
    df_score = pd.read_csv(os.path.join(path_, 'scores.csv'))
    all_bg_f1[c] = df_score[(df_score['Methods'] == 'KSP')]['F1'][0]

ratio = sorted(neigh_5_f1.keys())
f1s_all_bg = [all_bg_f1[r] for r in ratio]
f1s_neigh_10 = [neigh_10_f1[r] for r in ratio]
f1s_neigh_5 = [neigh_5_f1[r] for r in ratio]
f1s_missing = [missing_f1[r] for r in ratio]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.plot(ratio,
         f1s_neigh_5, 'o-',
         color=colors[0],
         label='5% distance of object')
plt.plot(ratio,
         f1s_neigh_10,
         'o-',
         color=colors[1],
         label='10% distance of object')
plt.plot(ratio,
         f1s_all_bg,
         'o-',
         color=colors[2],
         label='All background')
plt.plot(ratio,
         f1s_missing,
         'o-',
         color=colors[3],
         label='Missing locations')
plt.grid(True)
plt.ylabel('Tweezer')
plt.ylabel('F1')
plt.xlabel('Ratio [%]')
plt.legend(loc= 'lower left')
plt.savefig(os.path.join(out_result_dir, 'outliers.png'),
            dpi=100,bbox_inches='tight')
plt.clf()

#plot candidates
root_dir = os.path.join(rd.root_dir, 'Dataset00', 'gaze-measurements')
path_dist_5 = 'candidates1_neigh_ratio_50_dist_5.npz'
path_dist_10 = 'candidates1_neigh_ratio_50_dist_10.npz'
candidates_dist_5 = np.load(os.path.join(root_dir,
                                         path_dist_5))['candidates']
candidates_dist_10 = np.load(os.path.join(root_dir,
                                          path_dist_10))['candidates']

ims = [utls.imread(f) for f in conf.frameFileNames]

dataset = ld.LearningDataset(conf)
gt = dataset.gt

# Image
ind_f = 10

cont_gt = segmentation.find_boundaries(
    gt[..., ind_f], mode='thick')
im_orig = ims[ind_f]

#mask = (candidates_dist_10[..., 10] > 0) + gt[..., ind_f]
mask = (candidates_dist_10[..., 10] > 0)
im = color.label2rgb(mask,
                     im_orig,
                     alpha=.3,
                     bg_label=0,
                     colors=[color_dict['red']])
im = color.label2rgb(mask,
                     im_orig,
                     alpha=.3,
                     bg_label=0,
                     colors=[color_dict['red']])
#im[cont_gt, :] = (255, 0, 0)
io.imsave(os.path.join(out_result_dir, 'neigh_10.png'),
          (im*255).astype(np.uint8))

im_orig = ims[ind_f]

mask = (candidates_dist_5[..., 10] > 0)
#mask = (candidates_dist_5[..., 10] > 0) + gt[..., ind_f]
im = color.label2rgb(mask,
                     im_orig,
                     alpha=.3,
                     bg_label=0,
                     colors=[color_dict['red']])
#im[cont_gt, :] = (255, 0, 0)
io.imsave(os.path.join(out_result_dir, 'neigh_5.png'),
          (im*255).astype(np.uint8))
