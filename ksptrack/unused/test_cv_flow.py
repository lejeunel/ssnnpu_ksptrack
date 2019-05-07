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
import scipy as sp
import cv2


def main(arg_cfg):
    data = dict()

    #Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['ds_dir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)

    #Write config to result dir
    conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir, conf.resultDir,
                                    conf.out_dir_prefix, conf.testing)

    #Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('iterative_ksp')


    logger.info('---------------------------')
    logger.info('starting experiment on: ' + conf.ds_dir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('features type: ' + conf.feat_extr_algorithm)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    #Make frame file names
    gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
    gtFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.truth_dir,
        conf.root_path, conf.ds_dir, conf.frame_extension)

    conf.frameFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.frameDir,
        conf.root_path, conf.ds_dir, conf.frame_extension)


    #conf.myGaze_fg = utls.readCsv(conf.csvName_fg)
    conf.myGaze_fg = utls.readCsv(os.path.join(conf.root_path,conf.ds_dir,conf.locs_dir,conf.csvFileName_fg))

    #conf.myGaze_bg = utls.readCsv(conf.csvName_bg)
    gt_positives = utls.getPositives(gtFileNames)

    if (conf.labelMatPath != ''):
        conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.ds_dir, conf.frameDir,
                                    conf.labelMatPath)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                    conf.feats_dir)

    # ---------- Descriptors/superpixel costs
    my_dataset = ds.Dataset(conf)
    #my_dataset.flows_mat_to_np()
    if(conf.calc_superpix): my_dataset.calc_superpix(save=True)

    my_dataset.load_superpix_from_file()
    #my_dataset.relabel(save=True,who=conf.relabel_who)

    from scipy.spatial.distance import mahalanobis
    from metric_learn import LFDA
    from sklearn.decomposition import PCA

    #Calculate covariance matrix
    descs = my_dataset.sp_desc_df
    labels = my_dataset.labels
    my_dataset.load_all_from_file()
    pm = my_dataset.fg_pm_df

    sps_man_for = spm.SuperpixelManager(my_dataset,
                                        conf,
                                        direction='forward',
                                        with_flow=True)
    sps_man_for.make_dicts()


    frame_1 = 14
    label_1 = 465
    label_1 = 295

    frame_2 = 15
    label_2 = 460


    im1 = utls.imread(conf.frameFileNames[frame_1])
    label_cont = segmentation.find_boundaries(labels[...,frame_1], mode='thick')
    aimed_cont = segmentation.find_boundaries(labels[...,frame_1] == label_1,
                                              mode = 'thick')
    im1[label_cont,:] = (255,255,255)
    im1[aimed_cont,:] = (255,0,0)

    im2 = utls.imread(conf.frameFileNames[frame_2])
    label_cont = segmentation.find_boundaries(labels[...,frame_1], mode='thick')
    im2[label_cont,:] = (255,255,255)

    import pdb; pdb.set_trace()
    flow = cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    plt.subplot(221)
    plt.imshow(im1)
    plt.title('frame_1. ind: ' + str(frame_1))
    plt.subplot(222)
    plt.imshow(im2)
    plt.title('frame_2. ind: ' + str(frame_2))
    plt.subplot(223)
    plt.imshow(labels[...,frame_1])
    plt.title('labels')
    plt.subplot(224)
    plt.imshow(dists)
    plt.title('dists')
    plt.show()

    return conf
