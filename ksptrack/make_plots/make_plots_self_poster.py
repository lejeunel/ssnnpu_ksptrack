from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation, util,transform,io)
from skimage.util import montage
import os
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import learning_dataset
from ksptrack.utils import my_utils as utls
from ksptrack.exps import results_dirs as rd
from PIL import Image, ImageFont, ImageDraw
import glob

"""
Makes plots self
"""

cmap = plt.get_cmap('viridis')

def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

file_out = os.path.join(rd.root_dir, 'plots_results')
placehold = utls.imread(os.path.join(file_out, 'placeholder.png'))

n_sets_per_type = 1

dfs = []
# Self-learning

ims = []
ksp = []
pm = []

#for key in rd.res_dirs_dict_ksp.keys(): # Types
for key in rd.types: # Types
    ims.append([])
    ksp.append([])

    dsets_to_plot = np.asarray(rd.best_dict_ksp[key][0:n_sets_per_type])
    for dset, gset in zip(dsets_to_plot[:,0], dsets_to_plot[:,1]):

        confs = [rd.confs_dict_ksp[key][dset][g] for g in range(5)]

        # Load config

        dataset = learning_dataset.LearningDataset(confs[0])
        gt = dataset.gt

        f = rd.self_frames_dict[key][dset]

        # Image
        cont_gt = segmentation.find_boundaries(
            gt[..., f], mode='thick')
        idx_cont_gt = np.where(cont_gt)
        im = utls.imread(confs[0].frameFileNames[f])
        im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)

        locs2d = utls.readCsv(os.path.join(confs[0].root_path,
                                           confs[0].ds_dir,
                                           confs[0].locs_dir,
                                           confs[0].csvFileName_fg))
        im =  csv.draw2DPoint(locs2d,
                              f,
                              im,
                              radius=7)
        ims[-1].append(im)

        file_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][dset][gset],
                                'metrics.npz')

        file_pm = sorted(glob.glob(os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][dset][gset],
                                'pm_scores*.npz')))[-1]

        print('Loading (KSP): ' + file_ksp)
        npzfile = np.load(file_ksp)
        ksp_ = gray2rgb(npzfile['ksp_scores'][..., f])
        ksp[-1].append(ksp_)

        print('Loading (PM): ' + file_pm)
        npzfile = np.load(file_pm)

        pm_ = (cmap(npzfile['pm'])[...,0:3]*255).astype(np.uint8)
        pm[-1].append(pm_)


ims = [ims[i][j] for i in range(len(ims)) for j in range(len(ims[i]))]
ksp = [ksp[i][j] for i in range(len(ksp)) for j in range(len(ksp[i]))]

widths = [ims[i].shape[1] for i in range(len(ims))]
heights = [ims[i].shape[0] for i in range(len(ims))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [np.ceil((((ims[i].shape[0]-min_height)/2,(ims[i].shape[0]-min_height)/2), ((ims[i].shape[1]-min_width)/2,(ims[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_ksp = [np.ceil((((ksp[i].shape[0]-min_height)/2,(ksp[i].shape[0]-min_height)/2), ((ksp[i].shape[1]-min_width)/2,(ksp[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]

ims_crop = [util.crop(ims[i], to_crop_im[i]) for i in range(len(ims))]
ksp_crop = [util.crop(ksp[i], to_crop_ksp[i]) for i in range(len(ims))]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
ksp_res = [(transform.resize(ksp_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]

pw = 10
ims_pad = [util.pad(ims_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
ksp_pad = [util.pad(ksp_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]

all_ = np.concatenate([np.concatenate([ims_pad[i], ksp_pad[i]], axis=0) for i in range(len(ims_pad))], axis=1)

#Write methods on image
header_height = 150

im_path = os.path.join(file_out,'all_self_poster.png')
print('Saving to: ' + im_path)
io.imsave(im_path, all_)
