import os
import iterative_ksp
import numpy as np
import datetime
from cfgs import cfg_unet
import yaml
import numpy as np
import matplotlib.pyplot as plt

extra_cfg = dict()

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_oflow'] = True
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_linking'] = False
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['calc_seen_feats'] = True
extra_cfg['calc_ss'] = False
extra_cfg['calc_desc_means'] = False
extra_cfg['n_iters_ksp'] = 10
#extra_cfg['feat_extr_algorithm'] = 'unet'      # set unet as feature extractor algorithm
extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm
#extra_cfg['feats_graph'] = 'overfeat'      # set unet as feature extractor algorithm
extra_cfg['thresh_aux'] = []
extra_cfg['calc_sp_feats_unet_gaze_rec'] = False
extra_cfg['calc_sp_feats_unet_rec'] = False

extra_cfg['out_dir_prefix'] = 'exp'

all_datasets = ['Dataset00']

extra_cfg['root_path'] = '/mnt/labeling/medical-labeling/' # for docker image
extra_cfg['dataOutRoot'] = '/mnt/labeling/medical-labeling/' # for docker image

all_confs = []
conf_train = cfg_unet.Bunch(cfg_unet.cfg())

confs_fold = []

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['ds_dir'] = all_datasets[i]
    print("dset: " + all_datasets[i])
    for k in [3, 4, 5]:
    #for k in [1, 2, 3, 4, 5]:
        extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'
        conf = iterative_ksp.main(extra_cfg)
