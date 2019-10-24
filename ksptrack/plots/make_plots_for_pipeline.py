from sklearn.metrics import (precision_recall_curve)
from skimage import (color, segmentation, util, transform, io)
from skimage.util import montage
import os
from ksptrack.utils import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ksptrack.utils import learning_dataset
from ksptrack.utils import csv_utils as csv
from ksptrack.exps import results_dirs as rd
from PIL import Image, ImageFont, ImageDraw
"""
Makes plots self
"""


def gray2rgb(im):
    return (color.gray2rgb(im) * 255).astype(np.uint8)


file_out = os.path.join(rd.root_dir, 'plots_results', 'for_pipeline')

n_sets_per_type = 1

dfs = []
# Self-learning

ims = []
ims_gaze = []
ims_gaze_sp = []
ksp = []

types = ['Brain']

#for key in rd.res_dirs_dict_ksp.keys(): # Types
for key in types:  # Types
    dset = np.asarray(rd.best_dict_ksp[key][0:n_sets_per_type])[0][0]
    frames_idx = rd.all_frames_dict[key][dset]
    conf = rd.confs_dict_ksp[key][dset][0]

    # Load config

    dataset = learning_dataset.LearningDataset(conf)
    dataset.load_labels_contours_if_not_exist()
    gt = dataset.gt
    sp_conts = dataset.labelContourMask

    file_ksp = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp[key][dset][0],
                            'metrics.npz')

    print('Loading: ' + file_ksp)
    npzfile = np.load(file_ksp)

    for f in frames_idx:

        # Image
        im = utls.imread(conf.frameFileNames[f])
        ims.append(im)
        locs2d = utls.readCsv(
            os.path.join(conf.root_path,
                         conf.ds_dir,
                         conf.locs_dir,
                         conf.csvFileName_fg))
        im = csv.draw2DPoint(locs2d, f, im, radius=14)
        ims_gaze.append(np.copy(im))

        mi, mj = np.where(sp_conts[..., f])
        im[mi, mj, :] = (255, 255, 255)

        ksp_ = gray2rgb(npzfile['ksp_scores'][..., f])
        ksp.append(ksp_)

widths = [ims[i].shape[1] for i in range(len(ims))]
heights = [ims[i].shape[0] for i in range(len(ims))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [
    np.ceil((((ims[i].shape[0] - min_height) / 2,
              (ims[i].shape[0] - min_height) / 2),
             ((ims[i].shape[1] - min_width) / 2,
              (ims[i].shape[1] - min_width) / 2), (0, 0)))
    for i in range(len(ims))
]
to_crop_ksp = [
    np.ceil((((ksp[i].shape[0] - min_height) / 2,
              (ksp[i].shape[0] - min_height) / 2),
             ((ksp[i].shape[1] - min_width) / 2,
              (ksp[i].shape[1] - min_width) / 2), (0, 0)))
    for i in range(len(ims))
]

ims_gaze_crop = [
    util.crop(ims_gaze[i], to_crop_im[i]) for i in range(len(ims))
]
ims_crop = [util.crop(ims[i], to_crop_im[i]) for i in range(len(ims))]
ksp_crop = [util.crop(ksp[i], to_crop_ksp[i]) for i in range(len(ims))]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size) * 255).astype(np.uint8)
           for i in range(len(ims))]
ims_gaze_res = [(transform.resize(ims_gaze_crop[i], f_size) * 255).astype(
    np.uint8) for i in range(len(ims))]
ksp_res = [(transform.resize(ksp_crop[i], f_size) * 255).astype(np.uint8)
           for i in range(len(ims))]

pw = 10
ims_gaze_pad = [
    util.pad(
        ims_gaze_res[i], ((pw, pw), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=255) for i in range(len(ims))
]
ims_pad = [
    util.pad(
        ims_res[i], ((pw, pw), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=255) for i in range(len(ims))
]
ksp_pad = [
    util.pad(
        ksp_res[i], ((pw, pw), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=255) for i in range(len(ims))
]

if (not os.path.exists(file_out)):
    os.makedirs(file_out)

for i in range(len(ims_pad)):
    io.imsave(
        os.path.join(file_out, 'im_gaze_{0}.png'.format(i)), ims_gaze_pad[i])
    io.imsave(os.path.join(file_out, 'im_{0}.png'.format(i)), ims_pad[i])
    io.imsave(os.path.join(file_out, 'ksp_{0}.png'.format(i)), ksp_pad[i])
