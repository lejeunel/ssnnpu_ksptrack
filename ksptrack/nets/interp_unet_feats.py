import pandas as pd
import h5py
import numpy as np
import graph_tool as gt
from multiprocessing import Pool
from functools import partial
import my_utils as utls
import yaml, progressbar
import dataset as my_dataset
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import pandas as pd
import os

def upscale_feat(xx, yy, feats, labels,im_idx):
    """
    Uses interpolation to upscale the extracted features to image resolution.
    Should be called in a multiprocessing manner.

    :param xx:        Corresponding xx-vector for interpolation
    :param yy:        Corresponding yy-vector for interpolation
    :param imFeat:    Feature vector
    :param segm:      Superpixel segments
    :param feat_d:    Dimension of feature
    :param idx_range: Corresponding index range
    :return:          None
    """


    unique_sps = np.unique(labels)
    nbr_sp = unique_sps.shape[0]

    x_new = np.arange(labels.shape[1])
    y_new = np.arange(labels.shape[0])

    feat_d = feats.shape[-1]

    # save all interpolated features in that array
    # having shape: (n filters, height, width)
    feat_interp = np.zeros((feat_d, labels.shape[0], labels.shape[1]))

    # iterate over feature dimensions
    for feat_idx in range(feat_d):
        # kind = ‘linear’ ‘cubic’ ‘quintic’
        f = interp2d(xx, yy, feats[..., feat_idx], kind='cubic')
        feat_interp[feat_idx, ...] = f(x_new, y_new)

    out = []
    # iterate over superpixels
    for sp_idx in unique_sps:
        coord_y, coord_x = np.where(labels == sp_idx)
        this_feat = np.mean(feat_interp[:, coord_y, coord_x], axis=1)
        out.append((im_idx,sp_idx,this_feat))


    return out

dir_root = '/home/laurent.lejeune/medical-labeling/'
ds_dir = ['Dataset12']

feat_path = 'precomp_descriptors/unet/feat.h5'


for i in range(len(ds_dir)):

    print('Interpolating Unet features on:')
    print(ds_dir[i])
    labels = np.load(os.path.join(dir_root,                                      ds_dir[i],'input-frames',                                         'sp_labels.npz'))['sp_labels']
    frameFileNames = utls.makeFrameFileNames(
    'frame_', 4, 'input-frames',
        dir_root, ds_dir[i], 'png')

    im = utls.imread(frameFileNames[0])
    img_height = im.shape[0]
    img_width = im.shape[1]

    # initialize hdf5 files
    hdInFile = h5py.File(os.path.join(dir_root,ds_dir[i],feat_path), 'r')
    feats_arr = hdInFile['raw_feat'][...]
    hdInFile.close()

    modelDepth = 4

    # get feat properties
    feat_h = feats_arr.shape[1]
    feat_w = feats_arr.shape[2]
    feat_d = feats_arr.shape[3]

    patDist = (modelDepth-1)**2 -0.5

    patDist_y = patDist
    patDist_x = patDist
    y_s = patDist_y
    y_e = img_height - 1 - patDist_y
    x_s = patDist_x
    x_e = img_width - 1 - patDist_x

    # get the vectors
    yy = np.linspace(y_s, y_e, feat_h)
    xx = np.linspace(x_s, x_e, feat_w)

    x_vec = []
    #frames = np.arange(0,feats_arr.shape[0])
    frames = np.arange(0,feats_arr.shape[0])

    with progressbar.ProgressBar(maxval=frames.shape[0]) as bar:
        for j in range(frames.shape[0]):
            bar.update(j)
            x_vec += upscale_feat(xx,yy,feats_arr[frames[j],...],labels[...,j],j)

    out = pd.DataFrame(
        x_vec, columns=["frame", "sp_label", "desc"])
    out.sort_values(['frame','sp_label'],inplace=True)
