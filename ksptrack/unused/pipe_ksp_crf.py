import os
import iterative_ksp
import test_trans_costs
import numpy as np
import datetime
import cfg_unet
import yaml
import numpy as np
import matplotlib.pyplot as plt
import results_dirs as rd
import glob
import pydensecrf.densecrf as dcrf
import my_utils as utls
import progressbar
import pandas as pd
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
from skimage import (color, segmentation, util,transform,io)
import learning_dataset
import dataset as ds
import scipy.ndimage as spnd
import gazeCsv as gaze
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

idx_score_append = 3

def make_gaze_prior(shape, center, sigma):

    x, y = np.meshgrid(np.linspace(0,1,shape[1]),
                       np.linspace(0,1,shape[0]))
    x = x - center[0]
    y = y - center[1]
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )

    return g

def make_full_feats(labels, desc_df, f, feat_dim=512):
    shape = labels[..., f].shape
    out_feats = np.zeros((shape[0], shape[1], feat_dim))
    for l in np.unique(labels[..., f]):
        mask = labels[..., f] == l
        idx = np.where(mask)
        out_feats[idx[0], idx[1], :] = desc_df.loc[(desc_df['frame'] == f) & (desc_df['sp_label'] == l)]['desc'].as_matrix()[0]

    return out_feats

def do_3d_crf(conf, crf_shape):
    metric_path = os.path.join(conf.dataOutDir,
                                'metrics.npz')

    pm = np.load(metric_path)['pm_ksp']
    orig_shape = pm[...,0].shape
    pm = [transform.resize(pm[...,f], crf_shape) for f in range(pm.shape[-1])]
    pm = np.asarray(pm).transpose(1,2,0)
    shape = pm.shape
    pm = np.stack((pm, 1-pm), axis=0)
    ksp = np.load(metric_path)['ksp_scores']
    ims_rgb = [utls.imread(f) for f in conf.frameFileNames]
    ims_gray = [(color.rgb2gray(i)*255).astype(np.uint8) for i in ims_rgb]

    ims_rgb_resized = [transform.resize(ims_rgb[f], crf_shape) for f in range(pm.shape[-1])]
    ims_rgb_resized = np.asarray(ims_rgb_resized).transpose(1,2,3,0)
    ims_rgb_resized = np.ascontiguousarray(ims_rgb_resized)

    ims_gray_resized = [(transform.resize(ims_gray[f], crf_shape)*255).astype(np.uint8) for f in range(pm.shape[-1])]
    ims_gray_resized = np.asarray(ims_gray_resized).transpose(1,2,0)
    ims_gray_resized = np.ascontiguousarray(ims_gray_resized)

    nlabels = 2
    map_list = list()
    l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
    dataset = ds.Dataset(conf)
    dataset.load_labels_if_not_exist()
    d = dcrf.DenseCRF(np.prod(shape), nlabels)
    U_fg = -np.log(pm[0, ...].ravel()).reshape((1,-1))
    U_bg = -np.log(pm[1, ...].ravel()).reshape((1,-1))
    U = np.concatenate((U_fg,U_bg),axis=0).astype('float32')
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(1.0, 1.0, 1.0), shape=shape)
    d.addPairwiseEnergy(feats,
                        compat=3,
                        kernel=dcrf.FULL_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80),
                                      schan=(13, 13),
                                      img=ims_gray_resized,
                                      chdim=-1)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    map_ = 1 - np.argmax(Q, axis=0)
    map_.shape = crf_shape

    f = 50
    plt.subplot(221)
    plt.imshow(ims_rgb[f])
    plt.subplot(222)
    plt.imshow(ksp[...,f])
    plt.title('ksp')
    plt.subplot(223)
    plt.imshow(map_[..., f])
    plt.title('crf')
    plt.subplot(224)
    plt.imshow(pm[0,..., f])
    plt.title('pm')
    plt.show()


    crf_f1 = f1_score(l_dataset.gt.ravel(), map_.ravel())
    df_score = pd.read_csv(os.path.join(conf.dataOutDir,
                                        'scores.csv'))
    df_score.loc[df_score['Methods'] == 'CRF', 'F1'] = crf_f1
    df_score.to_csv(path_or_buf=os.path.join(conf.dataOutDir,
                                                'scores.csv'))
    data = {'map': map_}
    np.savez(os.path.join(conf.dataOutDir, 'crf.npz'), **data)

def do_crf(conf):
    metric_path = os.path.join(conf.dataOutDir,
                                'metrics.npz')

    pm = np.load(metric_path)['pm_ksp']
    ksp = np.load(metric_path)['ksp_scores']
    shape = pm.shape[0:2]
    nlabels = 2
    map_list = list()
    l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
    dataset = ds.Dataset(conf)
    dataset.load_labels_if_not_exist()
    labels = dataset.labels
    #gamma = 0.5
    gamma = 0.
    sigma = 0.2
    with progressbar.ProgressBar(maxval=len(conf.frameFileNames)) as bar:
        for f in range(pm.shape[-1]):
        #for f in [50]:
            bar.update(f)
            d = dcrf.DenseCRF2D(shape[1], shape[0], nlabels)
            im = utls.imread(conf.frameFileNames[f])
            im = np.ascontiguousarray(im)
            gaze_center = conf.myGaze_fg[conf.myGaze_fg[:,0]==f, 3:5].ravel()
            gaze_prior = make_gaze_prior(pm[..., f].shape,
                                         gaze_center,
                                         sigma)
            pm_gaze = (1-gamma)*(ksp[...,f] + pm[..., f]) + gamma*gaze_prior
            pm_gaze = np.clip(pm_gaze, a_min=0, a_max=1)
            #pm_gaze = ksp[..., f]
            U_fg = -np.log(pm_gaze.ravel()).reshape((1,-1))
            U_bg = -np.log(1-pm_gaze.ravel()).reshape((1,-1))
            U = np.concatenate((U_fg,U_bg),axis=0).astype('float32')
            d.setUnaryEnergy(U)
            #sp_desc_df = dataset.get_sp_desc_from_file()
            #all_feats = make_full_feats(labels, sp_desc_df, f).transpose(2,0,1).astype('float32')
            #all_feats = np.ascontiguousarray(all_feats)
            #d.addPairwiseEnergy(all_feats, compat=3)

            # This adds the color-independent term, features are the locations only.
            #d.addPairwiseGaussian(sxy=(3, 3),
            #                    compat=3,
            #                    kernel=dcrf.DIAG_KERNEL,
            #                    normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
            # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
            d.addPairwiseBilateral(sxy=(80, 80),
                                srgb=(13, 13, 13),
                                rgbim=im,
                                compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q = d.inference(5)
            map_ = 1-np.argmax(Q, axis=0).reshape((shape[0],shape[1]))
            #plt.subplot(231)
            #plt.imshow(im)
            #plt.subplot(232)
            #plt.imshow(ksp[...,f])
            #plt.title('ksp')
            #plt.subplot(233)
            #plt.imshow(map_)
            #plt.title('crf')
            #plt.subplot(234)
            #plt.imshow(pm[..., f])
            #plt.title('pm')
            #plt.subplot(235)
            #plt.imshow(gaze_prior)
            #plt.title('gaze_prior')
            #plt.subplot(236)
            #plt.imshow(pm_gaze)
            #plt.title('pm*gaze_prior')
            #plt.show()
            map_list.append(map_)

    maps = np.asarray(map_list).transpose(1,2,0)
    crf_f1 = f1_score(l_dataset.gt.ravel(), maps.ravel())
    df_score = pd.read_csv(os.path.join(conf.dataOutDir,
                                        'scores.csv'))
    df_score.loc[df_score['Methods'] == 'CRF', 'F1'] = crf_f1
    df_score.to_csv(path_or_buf=os.path.join(conf.dataOutDir,
                                                'scores.csv'))
    data = {'map': maps}
    np.savez(os.path.join(conf.dataOutDir, 'crf.npz'), **data)


type_ = [rd.types[2]]
crf_shape = (500,500)

# Run CRF on all seqs with all gaze-set
for t in type_:
    for d in rd.confs_dict_ksp[t].keys():
        for g in rd.confs_dict_ksp[t][d].keys():
            conf = rd.confs_dict_ksp[t][d][g]
            print("dset: " + conf.dataOutDir)
            #do_3d_crf(conf, crf_shape)
            do_crf(conf)
