import os
from labeling import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
import glob

extra_cfg = dict()

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_linking'] = False # Unused
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['calc_seen_feats'] = False # Unused
extra_cfg['calc_ss'] = False # Unused
extra_cfg['calc_desc_means'] = False # Unused
extra_cfg['n_iters_ksp'] = 10
extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm
extra_cfg['thresh_aux'] = []
extra_cfg['calc_sp_feats_unet_gaze_rec'] = False
extra_cfg['calc_oflow'] = True
extra_cfg['csvFileType'] = 'pandas'
extra_cfg['unet_interp_n_jobs'] = 4
extra_cfg['force_relabel'] = False

dataset = 'Dataset00'

root_dir = '/home/laurent.lejeune/medical-labeling/Dataset00/gaze-measurements'
locs_fnames = ['video1_neigh_ratio_5_dist_5.csv']

locs_fnames = sorted(locs_fnames)
n_chunks = 4

def chunkify(lst, n):

  return [lst[i::n] for i in range(n)]

chunks = chunkify(locs_fnames, n_chunks)

extra_cfg['ds_dir'] = dataset

ind = 0
#ind = 1
#ind = 2
#ind = 3

#for f in chunks[ind]:
for f in locs_fnames:
    extra_cfg['csvFileName_fg'] = f
    conf = iterative_ksp.main(extra_cfg)
