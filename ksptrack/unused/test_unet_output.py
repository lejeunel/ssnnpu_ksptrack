from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import cfg
import pandas as pd
import pickle as pk
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
import superPixels as spix
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import graphtracking as gtrack
import my_utils as utls
import dataset as ds
import selective_search as ss
import shutil as sh
import learning_dataset
import logging
from skimage import filters
import superpixelmanager as spm
import networkx as nx


def main(arg_cfg):
    data = dict()

    #Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['ds_dir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)

    #Write config to result dir
    conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot,
                                         conf.ds_dir,
                                         conf.resultDir,
                                         conf.out_dir_prefix,
                                         conf.testing)

    #Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('ksp_'+conf.ds_dir)

    logger.info('---------------------------')
    logger.info('starting experiment on: ' + conf.ds_dir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('features type: ' + conf.feat_extr_algorithm)
    logger.info('features type for graph: ' + conf.feats_graph)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    #Make frame file names
    conf.frameFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.frameDir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    if(conf.csvFileType == 'pandas'):
        conf.myGaze_fg = pd.read_csv(os.path.join(conf.root_path,
                                                  conf.ds_dir,
                                                  conf.locs_dir,
                                                  conf.csvFileName_fg))
    else:
        conf.myGaze_fg = utls.readCsv(os.path.join(conf.root_path,
                                                   conf.ds_dir,
                                                   conf.locs_dir,
                                                   conf.csvFileName_fg))

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                    conf.feats_dir)

    # ---------- Descriptors/superpixel costs
    my_dataset = ds.Dataset(conf)

    dir_ = conf.unet_gaze_rec_path + '_' + 'g1'
    weights_save_path = os.path.join(conf.precomp_desc_path,
                                    dir_)
    out = my_dataset.get_unet_gaze_output(weights_save_path)

    data = {'out': out}
    np.savez(os.path.join(conf.precomp_desc_path,
                          'gaze_out_example.npz'),
             **data)

    return conf
