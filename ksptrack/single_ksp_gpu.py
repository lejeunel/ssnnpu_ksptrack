import os
from ksptrack import iterative_ksp
import numpy as np
import datetime
import numpy as np
import matplotlib.pyplot as plt
from ksptrack.utils import write_frames_results as writef

# These parameters will override the ones from cfgs/cfg.py

extra_cfg = dict()

# This is where data will be stored

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['n_iter_ksp'] = 10 # sets max amount of iterations (merges)

extra_cfg['feats_graph'] = 'unet_gaze'

extra_cfg['make_datetime_dir'] = True

extra_cfg['thresh_aux'] = []

# This is to force (re)computation of features

extra_cfg['calc_sp_feats_unet_gaze_rec'] = False
extra_cfg['calc_sp_feats_unet_rec'] = False
extra_cfg['calc_sp_feats_vgg16'] = False
extra_cfg['use_hoof'] = True

extra_cfg['fileOutPrefix'] = 'exp'

extra_cfg['dataSetDir'] = 'datasetTest' #This is a test dataset
extra_cfg['reqdsupervoxelsize'] = 20000

# extra_cfg['dataInRoot'] = '/home/laurent.lejeune/medical-labeling/'
# extra_cfg['dataOutRoot'] = '/home/laurent.lejeune/medical-labeling/'
extra_cfg['dataInRoot'] = '/home/krakapwa/Desktop/'
extra_cfg['dataOutRoot'] = '/home/krakapwa/Desktop/'
extra_cfg['frameDir'] = 'input-frames'
extra_cfg['resultDir'] = 'results'
extra_cfg['dataOutResultDir'] = ''
extra_cfg['gazeDir'] = 'gaze-measurements'
extra_cfg['gtFrameDir'] = 'ground_truth-frames'
extra_cfg['fileOutPrefix'] = 'exp'
extra_cfg['framePrefix'] = 'frame_'
extra_cfg['frameExtension'] = '.png'
extra_cfg['frameDigits'] = 4 # input frames are of the form frame_xxxx.png
extra_cfg['csvFileName_fg'] = '2dlocs.csv'
extra_cfg['pca'] = False

extra_cfg['tau_u'] = 0.7
extra_cfg['n_bins_hoof'] = 30

extra_cfg['sp_trans_init_mode'] = 'radius'

# Run segmentation
conf, logger = iterative_ksp.main(extra_cfg)

# Write result frames
writef.main(conf, logger=logger)
