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


def main(arg_cfg):
    data = dict()

    #Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['dataSetDir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)

    #Write config to result dir
    conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.dataSetDir, conf.resultDir,
                                    conf.fileOutPrefix, conf.testing)

    #Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('feat_extr')


    logger.info('---------------------------')
    logger.info('starting feature extraction on: ' + conf.dataSetDir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('features type: ' + conf.feat_extr_algorithm)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    #Make frame file names
    gt_dir = os.path.join(conf.dataInRoot, conf.dataSetDir, conf.gtFrameDir)
    gtFileNames = utls.makeFrameFileNames(
        conf.framePrefix, conf.frameDigits, conf.gtFrameDir,
        conf.dataInRoot, conf.dataSetDir, conf.frameExtension)

    conf.frameFileNames = utls.makeFrameFileNames(
        conf.framePrefix, conf.frameDigits, conf.frameDir,
        conf.dataInRoot, conf.dataSetDir, conf.frameExtension)

    conf.myGaze_fg = utls.readCsv(os.path.join(conf.dataInRoot,conf.dataSetDir,conf.gazeDir,conf.csvFileName_fg))

    #conf.myGaze_bg = utls.readCsv(conf.csvName_bg)
    gt_positives = utls.getPositives(gtFileNames)

    if (conf.labelMatPath != ''):
        conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.dataSetDir, conf.frameDir,
                                    conf.labelMatPath)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.dataSetDir,
                                    conf.feats_files_dir)

    # ---------- Descriptors/superpixel costs
    my_dataset = ds.Dataset(conf)

    my_dataset.load_superpix_from_file()
    my_dataset.calc_sp_feats_unet_gaze_rec(save=True)

    with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=True)

    logger.info('Finished feature extraction: ' + conf.dataSetDir)

    return conf
