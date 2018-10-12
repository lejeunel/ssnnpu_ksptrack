import glob
from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation, util,transform,io)
import os
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ksptrack.utils import my_utils as utls
from ksptrack.utils import learning_dataset
from ksptrack.exps import results_dirs as rd
from PIL import Image, ImageFont, ImageDraw


def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

save_dir_root = os.path.join(rd.root_dir, 'plots_results', 'frames_labels2018')

n_sets_per_type = 1

dfs = []
# Self-learning

ims = []
gts = []

for key in rd.types: # Types
    dset = np.asarray(rd.best_dict_ksp[key][0:n_sets_per_type])[0][0]
    conf = rd.confs_dict_ksp[key][dset][0]
    save_dir = os.path.join(save_dir_root, key)
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    dataset = learning_dataset.LearningDataset(conf)
    gts_fname = sorted(glob.glob(os.path.join(conf.dataInRoot,
                                              conf.dataSetDir,
                                              conf.gtFrameDir,
                                              '*.png')))
    gts = [utls.imread(f) for f in gts_fname]
    imgs = [utls.imread(f) for f in conf.frameFileNames]

    for i, (im, gt) in enumerate(zip(imgs, gts)):

        both_ = np.concatenate((im, gt), axis=1)
        fname = os.path.join(save_dir, 'im_{}.png'.format(i))
        io.imsave(fname, both_)
