import os
import yaml
from labeling.cfgs import cfg
import pandas as pd
import numpy as np
import labeling.graph_tracking as gtrack
import labeling.graph_tracking_cxx as gtrack_cxx
from labeling.utils import my_utils as utls
from labeling.utils.data_manager import DataManager
import labeling.sp_manager as spm
import logging
import pickle
import networkx as nx
import time
import inspect

extra_cfg = dict()

extra_cfg['calc_superpix'] = False  # Centroids and contours
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_linking'] = False
extra_cfg['calc_pm'] = False  # Calculate probability maps from marked SPs
extra_cfg['calc_seen_feats'] = True
extra_cfg['calc_ss'] = False
extra_cfg['calc_desc_means'] = False
extra_cfg['n_iter_ksp'] = 10
#extra_cfg['feat_extr_algorithm'] = 'unet'      # set unet as feature extractor algorithm
extra_cfg['feats_graph'] = 'unet_gaze'      # set unet as feature extractor algorithm
#extra_cfg['feats_graph'] = 'unet'      # set unet as feature extractor algorithm
#extra_cfg['feats_graph'] = 'overfeat'      # set unet as feature extractor algorithm
extra_cfg['thresh_aux'] = 0.5
extra_cfg['calc_sp_feats_unet_gaze_rec'] = False
extra_cfg['calc_sp_feats_unet_rec'] = False

extra_cfg['fileOutPrefix'] = 'exp'
extra_cfg['csvFileName_fg'] = 'video1.csv'
extra_cfg['dataSetDir'] = 'Dataset00'
extra_cfg['seq_type'] = cfg.datasetdir_to_type(extra_cfg['dataSetDir'])

extra_cfg['dataInRoot'] = '/home/krakapwa/Desktop/data/'
extra_cfg['dataOutRoot'] = '/home/krakapwa/Desktop/data/'

# Get path of this dir
path = os.path.split(inspect.stack()[0][1])[0]

cfg_dict = cfg.cfg()
cfg_dict.update(extra_cfg)
conf = cfg.dict_to_munch(cfg_dict)

conf.myGaze_fg = utls.readCsv(os.path.join(conf.dataInRoot,
                                           conf.dataSetDir,
                                           conf.gazeDir,
                                           conf.csvFileName_fg))

conf.precomp_desc_path = os.path.join(conf.dataOutRoot,
                                        conf.dataSetDir,
                                        conf.feats_files_dir)

my_dataset = DataManager(conf)

my_dataset.load_all_from_file()

sps_man_for = spm.SuperpixelManager(my_dataset,
                                    conf,
                                    direction='forward',
                                    with_flow=True)

sps_man_back = spm.SuperpixelManager(my_dataset,
                                    conf,
                                    direction='backward',
                                    with_flow=True)

sps_man_for.make_dicts()
sps_man_back.make_dicts()
sps_mans = {'forward': sps_man_for, 'backward': sps_man_back}
g_for_cxx = gtrack_cxx.GraphTracking(sps_mans,
                                     tol=0,
                                     mode='edge')

g_for_cxx.g.config(0,
                   1,
                   loglevel="info",
                   min_cost=True,
                   l_max=40)

g_for_cxx.make_trans_transform(my_dataset.sp_desc_df,
                               my_dataset.fg_pm_df,
                               conf.thresh_aux_fix,
                               conf.lfda_n_samps,
                               conf.lfda_dim,
                               conf.lfda_k,
                               pca=conf.pca)

g_for_cxx.makeFullGraph(
    my_dataset.get_sp_desc_from_file(),
    my_dataset.fg_pm_df,
    my_dataset.centroids_loc,
    utls.pandas_to_std_csv(my_dataset.conf.myGaze_fg),
    my_dataset.conf.normNeighbor_in,
    my_dataset.conf.thresh_aux,
    my_dataset.conf.tau_u,
    direction='forward',
    labels=my_dataset.labels)

utls.setup_logging('.')

g_for_cxx.run()
direction = 'forward'

g_for_cxx.merge_tracklets_temporally(my_dataset.centroids_loc,
                                    my_dataset.fg_pm_df,
                                    my_dataset.sp_desc_df,
                                    utls.pandas_to_std_csv(
                                        my_dataset.conf.myGaze_fg),
                                    my_dataset.conf.normNeighbor_in,
                                    my_dataset.conf.thresh_aux,
                                    my_dataset.get_labels(),
my_dataset.conf.tau_u)

g_for_cxx.run()
