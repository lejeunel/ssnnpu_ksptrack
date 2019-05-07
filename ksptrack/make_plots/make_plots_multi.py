from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
import my_utils as utls
import plot_results_ksp_simple_multigaze as pksp
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import learning_dataset as ld
from skimage import (color, segmentation, util, transform, io)
import gazeCsv as gaze
import results_dirs as rd
import shutil as sh

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

# Steps

# Make Inter-viewer table / Get frames
file_out = os.path.join(out_result_dir, 'multigaze.npz')
if (not os.path.exists(file_out)):
    ims = []
    ksp_means = []
    #for key in rd.out_dirs_dict_ksp.keys():
    for key in rd.types:
        f1 = []
        for dset in range(len(rd.out_dirs_dict_ksp[key])):
            print('Loading: ' + str(rd.out_dirs_dict_ksp[key][dset]))
            path_ = os.path.join(rd.root_dir, 'learning_exps',
                                 rd.out_dirs_dict_ksp[key][dset])

            # Make images/gts/gaze-point
            ims.append([])
            ksp_means.append([])
            l_dataset = ld.LearningDataset(
                rd.confs_dict_ksp[key][dset][0], pos_thr=0.5)
            confs = rd.confs_dict_ksp[key][dset]
            gt = l_dataset.gt
            ksp_mean_all = np.load(os.path.join(
                path_, 'dataset.npz'))['mean_ksp_scores']
            for f in rd.all_frames_dict[key][dset]:
                cont_gt = segmentation.find_boundaries(
                    gt[..., f], mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(confs[0].frameFileNames[f])
                im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
                for key_conf in confs.keys():
                    im = gaze.drawGazePoint(
                        confs[key_conf].myGaze_fg, f, im, radius=7)

                ims[-1].append(im)
                ksp_means[-1].append(ksp_mean_all[..., f])

    data = {'ims': ims, 'ksp_means': ksp_means}
    np.savez(file_out, **data)
else:
    print('Loading: ' + file_out)
    npzfile = np.load(file_out)
    ims = npzfile['ims']
    ksp_means = npzfile['ksp_means']

cmap = plt.get_cmap('viridis')
ims_flat = [ims[i][j] for i in range(len(ims)) for j in range(dsets_per_type)]
ksp_flat = [(cmap(ksp_means[i][j])[..., 0:3] * 255).astype(np.uint8)
            for i in range(len(ksp_means)) for j in range(dsets_per_type)]

widths = [ims_flat[i].shape[1] for i in range(len(ims_flat))]
heights = [ims_flat[i].shape[0] for i in range(len(ims_flat))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [
    np.ceil((((ims_flat[i].shape[0] - min_height) / 2,
              (ims_flat[i].shape[0] - min_height) / 2),
             ((ims_flat[i].shape[1] - min_width) / 2,
              (ims_flat[i].shape[1] - min_width) / 2), (0, 0)))
    for i in range(len(ims_flat))
]
to_crop_ksp = [
    np.ceil((((ksp_flat[i].shape[0] - min_height) / 2,
              (ksp_flat[i].shape[0] - min_height) / 2),
             ((ksp_flat[i].shape[1] - min_width) / 2,
              (ksp_flat[i].shape[1] - min_width) / 2), (0, 0)))
    for i in range(len(ksp_flat))
]

ims_crop = [
    util.crop(ims_flat[i], to_crop_im[i]) for i in range(len(ims_flat))
]
ksp_crop = [
    util.crop(ksp_flat[i], to_crop_ksp[i]) for i in range(len(ksp_flat))
]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size) * 255).astype(np.uint8)
           for i in range(len(ims_crop))]
ksp_res = [(transform.resize(ksp_crop[i], f_size) * 255).astype(np.uint8)
           for i in range(len(ksp_crop))]

pw = 10
ims_pad = [
    util.pad(
        ims_res[i], ((pw, pw), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=255) for i in range(len(ims_res))
]
ksp_pad = [
    util.pad(
        ksp_res[i], ((pw, pw), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=255) for i in range(len(ims_res))
]

all_ = np.concatenate([
    np.concatenate([ims_pad[i], ksp_pad[i]], axis=1)
    for i in range(len(ims_pad))
],
                      axis=0)
all_ = np.split(all_, 4, axis=0)
all_ = np.concatenate(all_, axis=1)

im_path = os.path.join(out_result_dir, 'all_multi.png')
print('Saving to: ' + im_path)
io.imsave(im_path, all_)
