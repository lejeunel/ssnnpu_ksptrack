from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation, util,transform,io)
from skimage.util import montage
import os
import datetime
import yaml
from labeling.utils import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd
from PIL import Image, ImageFont, ImageDraw


"""
Makes plots self
"""

def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

file_out = os.path.join(rd.root_dir, 'plots_results')

n_sets_per_type = 1

dfs = []
# Self-learning

ims = []
gts = []

for key in rd.types: # Types
    dset = np.asarray(rd.best_dict_ksp[key][0:n_sets_per_type])[0][0]
    frame = rd.all_frames_dict[key][dset][0]
    conf = rd.confs_dict_ksp[key][dset][0]

    # Load config

    dataset = learning_dataset.LearningDataset(conf)
    gt = dataset.gt

    file_ksp = os.path.join(rd.root_dir,
                            rd.res_dirs_dict_ksp[key][dset][0],
                            'metrics.npz')

    print('Loading: ' + file_ksp)
    npzfile = np.load(file_ksp)


    # Image
    im = utls.imread(conf.frameFileNames[frame])
    ims.append(im)

    gts.append(np.tile(gt[...,frame][...,np.newaxis],(1,1,3)))


widths = [ims[i].shape[1] for i in range(len(ims))]
heights = [ims[i].shape[0] for i in range(len(ims))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [np.ceil((((ims[i].shape[0]-min_height)/2,(ims[i].shape[0]-min_height)/2), ((ims[i].shape[1]-min_width)/2,(ims[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_gt = [np.ceil((((gts[i].shape[0]-min_height)/2,(gts[i].shape[0]-min_height)/2), ((gts[i].shape[1]-min_width)/2,(gts[i].shape[1]-min_width)/2),(0,0))) for i in range(len(gts))]

ims_crop = [util.crop(ims[i], to_crop_im[i]) for i in range(len(ims))]
gts_crop = [util.crop(gts[i], to_crop_gt[i]) for i in range(len(ims))]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
gts_res = [(transform.resize(gts_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]

pw = 10
ims_pad = np.concatenate([util.pad(ims_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))], axis=1)
gts_pad = np.concatenate([util.pad(gts_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))], axis=1)

all_ = np.concatenate([ims_pad, gts_pad], axis=0)

import pdb; pdb.set_trace()
file_out_im = os.path.join(file_out,'intro.png')
print('Saving to: ' + file_out_im)
io.imsave(file_out_im,all_)
