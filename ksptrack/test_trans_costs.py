from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
from ksptrack.utils import learning_dataset
from ksptrack.utils import superpixel_utils as spix
from ksptrack import sp_manager as spm
from ksptrack.utils import csv_utils as csv
from ksptrack.utils.link_agent import LinkAgent
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_mask import LinkAgentMask
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.utils.lfda import myLFDA
from ksptrack.cfgs import params
import pandas as pd
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import ndimage
from skimage import (color, io, segmentation, draw)
import shutil as sh
import logging
from skimage import filters
import scipy as sp

p = params.get_params()

p.add('--in-path')

cfg = p.parse_args()
cfg.in_path = '/home/ubelix/lejeune/data/medical-labeling/Dataset50'
# mask_path = '/home/ubelix/lejeune/runs/unet_region/Dataset10_2019-08-06_16-10/Dataset12/entrance_masks/proba'
# mask_path = '/home/ubelix/lejeune/runs/unet_region/Dataset00_2019-08-06_13-48/Dataset00/entrance_masks/proba'
# mask_path = '/home/ubelix/lejeune/runs/unet_region/Dataset20_2019-08-06_11-46/Dataset20/entrance_masks/proba'

#Make frame file names
cfg.frameFileNames = utls.get_images(os.path.join(cfg.in_path, cfg.frame_dir))

locs2d = utls.readCsv(os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname))

cfg.precomp_desc_path = os.path.join(cfg.in_path, 'precomp_desc')

# ---------- Descriptors/superpixel costs
cfg.feats_mode = 'autoenc'
dm = DataManager(cfg.in_path, cfg.precomp_dir)
dm.calc_superpix(cfg.slic_compactness, cfg.slic_n_sp)

# sps_man = spm.SuperpixelManager(dm)

cfg.bag_t = 30
cfg.bag_jobs = 1

link_agent = LinkAgentRadius(
    csv_path=os.path.join(cfg.in_path, cfg.locs_dir, cfg.csv_fname),
    data_path=os.path.join(cfg.in_path),
    thr_entrance=cfg.thr_entrance,
    sigma=cfg.ml_sigma,
    # sigma=0.0005,
    entrance_radius=cfg.norm_neighbor_in)

dm.calc_pm(np.array(link_agent.get_all_entrance_sps(dm.sp_desc_df)),
           cfg.bag_n_feats,
           cfg.bag_t,
           cfg.bag_max_depth,
           cfg.bag_max_samples,
           cfg.bag_jobs)
labels = dm.labels
descs = dm.sp_desc_df

link_agent.update_trans_transform(dm.sp_desc_df,
                                  dm.fg_pm_df, [0.3, 0.7],
                                  1000,
                                  15,
                                  25,
                                  embedding_type='orthonormalized')

frame_in = 52
frame_out = 53
pm_scores_fg = dm.get_pm_array(mode='foreground')

trans_probas = np.zeros(labels.shape[:2])
trans_dists = np.zeros(labels.shape[:2])
i_in, j_in = link_agent.get_i_j(locs2d[locs2d['frame'] == frame_in])
label_in = labels[i_in, j_in, frame_in]
for l in np.unique(labels[..., frame_out]):
    proba = link_agent.get_proba(descs, frame_in, label_in, frame_out, l)
    dist = link_agent.get_distance(descs, frame_in, label_in, frame_out, l)
    trans_probas[labels[..., frame_out] == l] = proba
    trans_dists[labels[..., frame_out] == l] = dist

entrance_probas = np.zeros(labels.shape[:2])
label_in = labels[i_in, j_in, frame_in]
for l in np.unique(labels[..., frame_in]):
    if (link_agent.is_entrance(frame_in, l)):
        proba = link_agent.get_proba(descs, frame_in, label_in, frame_in, l)
        entrance_probas[labels[..., frame_in] == l] = proba

im1 = utls.imread(cfg.frameFileNames[frame_in])
label_cont = segmentation.find_boundaries(labels[..., frame_in], mode='thick')
aimed_cont = segmentation.find_boundaries(labels[..., frame_in] == label_in,
                                          mode='thick')

label_cont_im = np.zeros(im1.shape, dtype=np.uint8)
label_cont_i, label_cont_j = np.where(label_cont)
label_cont_im[label_cont_i, label_cont_j, :] = 255

io.imsave('conts.png', label_cont_im)

rr, cc = draw.circle_perimeter(i_in, j_in,
                               int(cfg.norm_neighbor_in * im1.shape[1]))

im1[rr, cc, 0] = 0
im1[rr, cc, 1] = 255
im1[rr, cc, 2] = 0

im1[aimed_cont, :] = (255, 0, 0)
im1 = csv.draw2DPoint(locs2d.to_numpy(), frame_in, im1, radius=7)

im2 = utls.imread(cfg.frameFileNames[frame_out])
label_cont = segmentation.find_boundaries(labels[..., frame_out], mode='thick')
im2[label_cont, :] = (255, 0, 0)

# label_out = labels[i_in, j_in, frame_out]
# hoof_inter = sps_man.graph[(frame_in, label_in)][(frame_out, label_out)]['forward']
# hoof_out = sps_man.graph[(frame_out, label_out)]['forward']

# cfg.pm_thr = 0.6
plt.subplot(331)
plt.imshow(im1)
plt.title('frame_1. ind: ' + str(frame_in))
plt.subplot(332)
plt.imshow(im2)
plt.title('frame_2. ind: ' + str(frame_out))
plt.subplot(333)
plt.imshow(pm_scores_fg[..., frame_out] > cfg.pm_thr)
plt.title('f2 pm > {}'.format(cfg.pm_thr))
plt.subplot(334)
plt.imshow(entrance_probas)
plt.title('entrance probas')
plt.subplot(335)
plt.imshow(trans_probas)
plt.title('trans probas')
plt.subplot(336)
plt.imshow(pm_scores_fg[..., frame_out])
plt.title('f2. pm')
plt.subplot(337)
plt.imshow(trans_dists)
plt.title('trans dists')
plt.show()
