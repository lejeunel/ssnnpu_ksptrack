import os
import numpy as np
import datetime
from ksptrack.utils import write_frames_results as writef
from ksptrack import iterative_ksp
import numpy as np
import matplotlib.pyplot as plt

extra_cfg = dict()

extra_cfg['calc_superpix'] = True  # Centroids and contours
extra_cfg['calc_sp_feats'] = True
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['n_iters_ksp'] = 10
extra_cfg[
    'feats_graph'] = 'unet_gaze'  # set unet as feature extractor algorithm
extra_cfg['calc_sp_feats_unet_gaze_rec'] = True

extra_cfg['calc_sp_feats_unet_rec'] = False
extra_cfg[
    'feats_graph'] = 'unet_gaze'  # set unet as feature extractor algorithm

extra_cfg["cuda"] = True
extra_cfg['out_dir_prefix'] = 'exp'

extra_cfg['slic_n_sp'] = 1000
extra_cfg['use_hoof'] = True
extra_cfg['sp_trans_init_mode'] = 'radius'

all_datasets = [
    'Dataset35'
]

# all_datasets = [
#     'Dataset12', 'Dataset13'
# ]

# all_datasets = [
#     'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25'
# ]

# all_datasets = [
#     'Dataset32', 'Dataset33', 'Dataset34', 'Dataset35'
# ]

all_confs = []

confs_fold = []

# Run KSP on all seqs with first gaze-set and make prediction experiment
for i in range(len(all_datasets)):
    set_confs = []
    extra_cfg['ds_dir'] = all_datasets[i]
    print("dset: " + all_datasets[i])
    for k in [1]:
        #for k in ['_missing']:
        #for k in [1, 2, 3, 4, 5]:
        extra_cfg['csvFileName_fg'] = 'video' + str(k) + '.csv'
        conf, logger = iterative_ksp.main(extra_cfg)
        writef.main(conf, logger=logger)
        #conf = test_trans_costs.main(extra_cfg)
