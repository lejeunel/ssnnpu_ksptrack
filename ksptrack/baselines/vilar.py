import os
import yaml
import cfg
import numpy as np
import graphtracking as gtrack
import my_utils as utls
import dataset_vilar as ds
import learning_dataset as learning_ds
import selective_search as ss
import logging
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
from multiprocessing import Pool
from functools import partial
from skimage.util import pad
from skimage import (color, io, segmentation, morphology)
from skimage.filters.rank import median
from skimage.transform import rescale
import datetime
from joblib import Parallel, delayed
import sys
import matplotlib.pyplot as plt
import progressbar
import plot_results_vilar as pvilar
import pandas as pd
import pickle as pk


def df_to_mat(X):

    out = X.as_matrix()[:,3]
    return arr_reduce(out)

def arr_reduce(x):
    """
    x is an array of shape (m) of arrays of shape (n)
    returns y of shape (m,n)
    """
    y = np.empty((x.shape[0],x[0].shape[0]))
    for i in range(x.shape[0]):
        #print(i)
        y[i,:] = x[i]

    return y


def make_training_patches_and_fit(arg_cfg, out_dir=None):
    data = dict()

    # Update config
    cfg_dict = cfg.cfg()
    arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['ds_dir'])
    cfg_dict.update(arg_cfg)
    conf = cfg.Bunch(cfg_dict)
    conf.out_dir_prefix = 'exp_vilar'

    # Write config to result dir
    if(out_dir is None):
        if('dataOutDir' not in arg_cfg):
            conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir,
                                                conf.resultDir, conf.out_dir_prefix,
                                                conf.testing)
    else:
        conf.dataOutDir = out_dir


    conf.myGaze_fg = utls.readCsv(
        os.path.join(conf.root_path, conf.ds_dir, conf.locs_dir,
                     conf.csvFileName_fg))

    # Set logger
    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('vilar')

    logger.info('---------------------------')
    logger.info('Extracting training a patches and fit on: ' + conf.ds_dir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('n_frames : ' + str(conf.vilar_n_frames))
    logger.info('(gamma,C) = (' + str(conf.gamma) + ', ' + str(conf.C) + ')')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')

    # Make frame file names
    gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
    gtFileNames = utls.makeFrameFileNames(conf.frame_prefix, conf.frameDigits,
                                          conf.truth_dir, conf.root_path,
                                          conf.ds_dir, conf.frame_extension)

    conf.frameFileNames = utls.makeFrameFileNames(
        conf.frame_prefix, conf.frameDigits, conf.frameDir, conf.root_path,
        conf.ds_dir, conf.frame_extension)

    # conf.myGaze_fg = utls.readCsv(conf.csvName_fg)
    conf.myGaze_fg = utls.readCsv(
        os.path.join(conf.root_path, conf.ds_dir, conf.locs_dir,
                     conf.csvFileName_fg))

    # conf.myGaze_bg = utls.readCsv(conf.csvName_bg)
    gt_positives = utls.getPositives(gtFileNames)

    conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                          conf.feats_dir)

    my_dataset = ds.DatasetVilar(conf)

    with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=True)

    # Extract seen and unseen patches (to fit SVC)
    if(os.path.exists(os.path.join(conf.dataOutDir,'vilar','vilar_seen_patches_df.p')) &
       os.path.exists(os.path.join(conf.dataOutDir,'vilar','vilar_unseen_patches_df.p'))):
        logger.info('seen and unseen patches already computed. Skipping.')
    else:
        my_dataset.calc_training_patches(save=True)



    if(not os.path.exists(os.path.join(conf.dataOutDir,'clf.p'))):

        my_dataset.load_patches()

        X_pos = df_to_mat(my_dataset.seen_patches_df)
        X_neg = df_to_mat(my_dataset.unseen_patches_df)
        X_train = np.concatenate((X_pos, X_neg))

        y_pos = np.ones(df_to_mat(my_dataset.seen_patches_df).shape[0])
        y_neg = np.zeros(df_to_mat(my_dataset.unseen_patches_df).shape[0])
        y_train = np.concatenate((y_pos, y_neg))

        clf = SVC(gamma=conf.gamma,
                C = conf.C,
                class_weight='balanced',
                verbose=True)
        logger.info('fitting')
        clf.fit(X_train, y_train)
        with open(os.path.join(conf.dataOutDir,'clf.p'), 'wb') as f:
            pk.dump(clf, f)
        logger.info('Saved classifier.')
    else:
        with open(os.path.join(conf.dataOutDir,'clf.p'), 'rb') as f:
            clf = pk.load(f)

    return conf, clf

def predict(conf, clf, out_dir=None, n_jobs=1):
    """
    Parallel extraction and prediction of patches on every pixel

    :imgs: list of images
    :returns predictions
    """

    # Set logger
    if(out_dir is not None):
        conf.dataOutDir = out_dir

    save_path = os.path.join(conf.dataOutDir,
                            'vilar')

    utls.setup_logging(conf.dataOutDir)

    logger = logging.getLogger('vilar (predict)')

    logger.info('---------------------------')
    logger.info('Predicting on: ' + conf.ds_dir)
    logger.info('type of sequence: ' + conf.seq_type)
    logger.info('gaze filename: ' + conf.csvFileName_fg)
    logger.info('Result dir:')
    logger.info(conf.dataOutDir)
    logger.info('---------------------------')
    n_frames = conf.vilar_n_frames
    if(n_frames == -1):
        n_frames = len(conf.frameFileNames)

    batch_size = n_frames
    batch_idx = np.array_split(np.arange(0, n_frames), n_frames/n_jobs)


    selem = morphology.square(5)
    ims = []

    for i in range(n_frames):

        img = utls.imread(conf.frameFileNames[i])
        img = (color.rgb2gray(img)*255).astype(np.uint8)
        #img = (color.rgb2gray(img)*255).astype(np.uint8)
        #img = utls.imread(self.conf.frameFileNames[i])
        img = median(img, selem=selem)  # Uses square of size 3
        ims.append(img)    # make feat segment dictionary list

    save_path = os.path.join(conf.dataOutDir,
                            'vilar')

    if(not os.path.exists(save_path)):
        os.mkdir(save_path)

    for b in range(len(batch_idx)):
        stack = list()
        cnt = 0
        for b_ in range(batch_idx[b].shape[0]):
            i = batch_idx[b][b_]
            stack.append(dict())
            stack[cnt]['clf'] = clf
            stack[cnt]['im'] = ims[i]
            stack[cnt]['shape'] = ims[i].shape
            stack[cnt]['f_num'] = i
            stack[cnt]['ps'] = conf.patch_size
            stack[cnt]['scale_factor'] = conf.scale_factor
            stack[cnt]['save_path_patch'] = os.path.join(save_path,
                                                'all_patches_im_'+str(i)+'.npz')
            stack[cnt]['save_path_pred'] = os.path.join(save_path,
                                                'pred_im_'+str(i)+'.npz')
            cnt += 1

        with Pool(processes=n_jobs) as pool:
            pool.map(predict_job, stack)

    return conf

def predict_job(dionysos):

    im = dionysos['im']
    ps = dionysos['ps']
    clf = dionysos['clf']
    shape = dionysos['shape']
    scale_factor = dionysos['scale_factor']
    f_num = dionysos['f_num']
    save_path_patch = dionysos['save_path_patch']
    save_path_pred = dionysos['save_path_pred']

    if(os.path.exists(save_path_patch)):
        print(save_path_patch + ' exists. Delete to recompute')
        sys.stdout.flush()
        patches = np.load(os.path.join(save_path_patch))
    else:


        print('started extraction on image ' + str(f_num))
        sys.stdout.flush()

        data = dict()

        im_padded = pad(im, ((ps, ps),), mode='symmetric')

        idx_i, idx_j = np.meshgrid(np.arange(0,im.shape[0]),
                                np.arange(0,im.shape[1]),
                                indexing='ij')
        idx_i = idx_i.ravel()
        idx_j = idx_j.ravel()

        data['frame'] = f_num
        patches = []

        for k in range(idx_i.shape[0]):
            patch = im_padded[int(idx_i[k] + ps/2):int(idx_i[k] + 3*ps/2),
                                int(idx_j[k] + ps/2):int(idx_j[k] + 3*ps/2)]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            if(patch_std == 0): patch_std = 1
            patch = (patch - patch_mean) / patch_std
            patch = rescale(
                patch,
                scale=scale_factor,
                order=1,
                mode='reflect',
                preserve_range=True)
            patches.append(patch.ravel())

        data['patches'] = np.asarray(patches)
        data['idx_i'] = idx_i
        data['idx_j'] = idx_j

        np.savez_compressed(save_path_patch,**data)
        print('finished extraction on image ' + str(f_num))
        sys.stdout.flush()

    if(not os.path.exists(save_path_pred)):
        print('starting prediction on image ' + str(f_num))
        sys.stdout.flush()

        X = np.load(save_path_patch)['patches']
        pred = clf.predict(X)
        pred = pred.reshape(shape[0:2])

        data = dict()
        data['pred'] = pred
        print('saving to: ' + save_path_pred)
        np.savez(save_path_pred, **data)
        sys.stdout.flush()

        print('np.sum(pred): ' + str(np.sum(pred)))
        sys.stdout.flush()

    else:
        print(save_path_pred + ' exists. Delete to recompute')
        sys.stdout.flush()

    return True

def calc_score(conf):

    dataset = ds.DatasetVilar(conf)
    dir_ = os.path.join(conf.dataOutDir,'vilar')
    preds = np.asarray([dataset.get_pred_frame(f,dir_) for f in range(len(conf.frameFileNames))]).transpose(1,2,0)
    gts = dataset.gt
    y_true = utls.make_y_array_true(gts, dataset.get_labels())
    y_pred = utls.make_y_array_true(preds, dataset.get_labels())
    f1_sp = f1_score(y_true[:, 2], y_pred[:, 2])
    f1_px = f1_score(gts.ravel(),preds.ravel())
    file_out = os.path.join(conf.dataOutDir,'score.csv')

    C = pd.Index(['F1_sp', 'F1_px'], name="columns")
    data = np.asarray([f1_sp, f1_px]).reshape(1,2)
    df = pd.DataFrame(data=data, columns=C)
    df.to_csv(path_or_buf=file_out)
