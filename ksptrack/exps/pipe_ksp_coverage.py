import os
import iterative_ksp
import test_trans_costs
import numpy as np
import datetime
import cfg_unet
import yaml
import numpy as np
import matplotlib.pyplot as plt

extra_cfg = dict()

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_linking'] = False
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['calc_seen_feats'] = True
extra_cfg['calc_ss'] = False
extra_cfg['calc_desc_means'] = False
extra_cfg['n_iter_ksp'] = 10
extra_cfg['feat_extr_algorithm'] = 'unet_gaze'      # set unet as feature extractor algorithm
extra_cfg['thresh_aux'] = []
extra_cfg['calc_sp_feats_unet_gaze_rec'] = True

all_datasets = ['Dataset03']
root_dir = '/home/krakapwa/otlshare/medical-labeling'

#cov_suffix = ['40', '20']
cov_suffix = ['60']

all_confs = []
conf_train = cfg_unet.Bunch(cfg_unet.cfg())

confs_fold = []

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['dataSetDir'] = all_datasets[i]
    print("dset: " + all_datasets[i])
    for k in range(len(cov_suffix)):
    #for k in [5]:
    #for k in [3,4,5]:
        extra_cfg['csvFileName_fg'] = 'cov' + cov_suffix[k] + '.csv'
        conf = iterative_ksp.main(extra_cfg)
        #conf = test_trans_costs.main(extra_cfg)
