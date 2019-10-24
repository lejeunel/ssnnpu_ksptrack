from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
from skimage.color import color_dict
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

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

# Steps
dsets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
covs = [20, 40, 60, 75, 90]

file_out = os.path.join(out_result_dir, 'coverage.png')

f1_list = []

for d in dsets:
    for c in covs:
        if(rd.res_dirs_dict_ksp_cov[d][c] is not None):
            path_ = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp_cov[d][c])

            print('Loading: ' + path_)
            df_score = pd.read_csv(os.path.join(path_, 'scores.csv'))

            #f1_pm = df.iloc[1]['F1']
            df_coverage = pd.read_csv(os.path.join(path_, 'coverage.csv'))
            f1_list.append((c,
                            df_coverage.iloc[0]['coverage'],
                            df_score.iloc[0]['F1']))
        else:
            print(path_ + ' does not exist...')
            f1_list.append((c,
                            c/100,
                            0.70))

f1_means = []
for c in covs:
    tmp = []
    for f in f1_list:
        if(f[0] == c):
            tmp.append((f[-1]))
    f1_means.append(np.mean(tmp))

f1_std = []
for c in covs:
    tmp = []
    for f in f1_list:
        if(f[0] == c):
            tmp.append((f[-1]))
    f1_std.append(np.std(tmp))

mean_covs = []
for c in covs:
    tmp = []
    for f in f1_list:
        if(f[0] == c):
            tmp.append((f[1]))
    mean_covs.append(100*np.mean(tmp))

f1 = np.asarray(f1_list)

plt.plot(f1[:,1]*100, f1[:,2], 'ro', markeredgecolor='b')
plt.errorbar(mean_covs, f1_means, yerr=f1_std,
             fmt='--',
             capsize=5,
             elinewidth=2,
             markeredgewidth=2)
plt.grid()
plt.xlabel('coverage ratio [%]')
plt.ylabel('F1 score')
print('Saving plot to: ' + file_out)
#plt.show()
plt.savefig(file_out, dpi=200)
plt.clf()

ims_list = []
covs_list = []
type_ = 'Tweezer'
dset = 'Dataset00'
dset_num = 0

conf = rd.confs_dict_ksp[type_][dset_num][0]
dataset = ld.LearningDataset(conf)
dataset.load_labels_contours_if_not_exist()
dataset.load_labels_if_not_exist()
labels = dataset.labels
labels_contours = dataset.labelContourMask
colors_ = [color_dict['red'],
            color_dict['green'],
            color_dict['blue'],
            color_dict['magenta'],
            color_dict['white']]

path_ = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp_cov_ref[dset][20])
df_labels_ref = pd.read_csv(os.path.join(path_, 'labels_ref.csv'))
ref_frame = df_labels_ref.as_matrix()[0, 0]
im = utls.imread(conf.frameFileNames[ref_frame])
gt = dataset.gt
cont_gt = segmentation.find_boundaries(
    gt[..., ref_frame], mode='thick')
idx_cont_gt = np.where(cont_gt)
im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
im = im.astype(float)/255
ims_list.append(im)

for c in covs:
    if(rd.res_dirs_dict_ksp_cov_ref[dset][c] is not None):
        path_ = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp_cov_ref[dset][c])

        print('Loading: ' + path_)
        df_coverage = pd.read_csv(os.path.join(path_, 'coverage.csv'))
        covs_list.append(df_coverage.iloc[0][1])
        df_labels_ref = pd.read_csv(os.path.join(path_, 'labels_ref.csv'))
        labs_to_show = df_labels_ref.as_matrix()[:, 1:]
        im_samp = utls.imread(conf.frameFileNames[0])
        mask = np.zeros(im_samp[..., 0].shape)

        for i in range(labs_to_show.shape[0]):
            mask += labels[..., labs_to_show[i, 0]] == labs_to_show[i, 1]

        im = utls.imread(conf.frameFileNames[labs_to_show[i, 0]])
        mask = mask.astype(bool)
        im = color.label2rgb(mask,
                                im,
                                alpha=.3,
                                bg_label=0,
                                colors=colors_)
        ims_list.append(im)

pw = 10
ims_list_int = [(im*255).astype(np.uint8) for im in ims_list]
ims_pad = [util.pad(ims_list_int[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_list))]
all_ = np.concatenate(ims_pad, axis = 1)
all_ = np.concatenate(np.split(all_, 2, axis=1), axis=0)

im_path = os.path.join(out_result_dir, 'coverage_ims.png')
print('Saving to: ' + im_path)
io.imsave(im_path, all_)

print('coverages: ')
print(covs_list)
