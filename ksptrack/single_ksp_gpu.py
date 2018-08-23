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
#extra_cfg['dataOutDir'] = ...

extra_cfg['calc_superpix'] = True  # Centroids and contours
extra_cfg['calc_sp_feats'] = True
extra_cfg['calc_pm'] = True  # Calculate probability maps from marked SPs
extra_cfg['n_iter_ksp'] = 10 # sets max amount of iterations (merges)

extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm

extra_cfg['thresh_aux'] = []
extra_cfg['calc_sp_feats_unet_gaze_rec'] = True
extra_cfg['calc_sp_feats_unet_rec'] = False
extra_cfg['calc_sp_feats_vgg16'] = True
extra_cfg['calc_oflow'] = True

extra_cfg['fileOutPrefix'] = 'exp'

# This sets all paths (input data, output dir, ...)


extra_cfg['dataSetDir'] = 'Dataset04'
extra_cfg['csvFileName_fg'] = '2dlocs.csv'
extra_cfg['pca'] = False

# Run segmentation
conf, logger = iterative_ksp.main(extra_cfg)

writef.main(conf, logger=logger)

