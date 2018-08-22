import os
from labeling import iterative_ksp
import numpy as np
import datetime
from labeling.cfgs import cfg_unet
import numpy as np
import matplotlib.pyplot as plt

extra_cfg = dict()

extra_cfg['calc_superpix'] = True  # Centroids and contours
#extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_sp_feats'] = True
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['n_iter_ksp'] = 10
extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm
#extra_cfg['feats_graph'] = 'unet'      # set unet as feature extractor algorithm
#extra_cfg['feats_graph'] = 'overfeat'      # set unet as feature extractor algorithm
extra_cfg['thresh_aux'] = 0.5
extra_cfg['calc_sp_feats_unet_gaze_rec'] = True
extra_cfg['calc_sp_feats_unet_rec'] = False
extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm

extra_cfg['fileOutPrefix'] = 'exp'

#all_datasets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
all_datasets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
#all_datasets = ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']
#root_dir = '/home/krakapwa/otlshare/medical-labeling'
extra_cfg['dataInRoot'] = '/home/krakapwa/Desktop/data'
extra_cfg['dataOutRoot'] = '/home/krakapwa/Desktop/data'

all_confs = []
conf_train = cfg_unet.Bunch(cfg_unet.cfg())

confs_fold = []

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['dataSetDir'] = all_datasets[i]
    print("dset: " + all_datasets[i])
    for k in [1]:
    #for k in ['_missing']:
    #for k in [1, 2, 3, 4, 5]:
        extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'
        conf = iterative_ksp.main(extra_cfg)
        #conf = test_trans_costs.main(extra_cfg)
