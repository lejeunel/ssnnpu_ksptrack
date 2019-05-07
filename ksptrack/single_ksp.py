import os
from ksptrack import iterative_ksp
from ksptrack.cfgs import cfg_unet
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef
from os.path import join as pjoin

""" This is the main script for gazelabel.com
It requires no GPU (uses pre-trained VGG16)
It writes result frames in conf.dataOutDir
"""

# These parameters will override the ones from cfgs/cfg.py

extra_cfg = dict()

# This is where data will be stored

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['n_iters_ksp'] = 10 # sets max amount of iterations (merges)

extra_cfg['feats_graph'] = 'vgg16'

extra_cfg['thresh_aux'] = []
extra_cfg['calc_sp_feats_unet_gaze_rec'] = False
extra_cfg['calc_sp_feats_unet_rec'] = False
extra_cfg['calc_sp_feats_vgg16'] = False
extra_cfg['calc_oflow'] = False
extra_cfg['use_hoof'] = True

extra_cfg['out_dir_prefix'] = 'exp'
extra_cfg['ds_dir'] = 'Dataset00' #This is a test dataset

extra_cfg['root_path'] = '/home/laurent.lejeune/medical-labeling/'
extra_cfg['dataOutRoot'] = '/home/laurent.lejeune/medical-labeling/'
extra_cfg['entrance_mask_path'] = pjoin('/home/laurent.lejeune/medical-labeling/',
                                        'unet_region/runs/2019-04-10_11-38-11',
                                        extra_cfg['ds_dir'],
                                        'entrance_masks')

extra_cfg['sp_trans_init_mode'] = 'radius'
extra_cfg['frameDir'] = 'input-frames'
extra_cfg['resultDir'] = 'results'
extra_cfg['dataOutResultDir'] = ''
extra_cfg['locs_dir'] = 'gaze-measurements'
extra_cfg['truth_dir'] = 'ground_truth-frames'
extra_cfg['out_dir_prefix'] = 'exp'
extra_cfg['frame_prefix'] = 'frame_'
extra_cfg['frame_extension'] = '.png'
extra_cfg['frameDigits'] = 4 # input frames are of the form frame_xxxx.png
extra_cfg['csvFileName_fg'] = 'video1.csv'
extra_cfg['pca'] = True

# Run segmentation
conf, logger = iterative_ksp.main(extra_cfg)

# Write result frames
writef.main(conf, logger=logger)

