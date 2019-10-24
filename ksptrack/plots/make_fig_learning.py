from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
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
from PIL import Image, ImageFont, ImageDraw

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

# Steps

# Make Inter-viewer frames
ims = []
preds_my = []
preds_true = []
for key in rd.types:
    ims.append([])
    preds_my.append([])
    preds_true.append([])
    path_ = os.path.join(rd.root_dir,
                        'learning_exps',
                        rd.learning_dirs_dict[key])
    print('Loading: ' + path_)
    dset = np.load(os.path.join(path_, 'datasets.npz'))['datasets']
    folds = rd.best_folds_learning[key][0:2]
    for fold in folds:
        conf = dset[fold].conf
        f = rd.self_frames_dict[key][fold]

        cont_gt = segmentation.find_boundaries(
            dset[fold].gt[..., f], mode='thick')
        idx_cont_gt = np.where(cont_gt)
        im = utls.imread(conf.frameFileNames[f])
        im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
        ims[-1].append(im)
        im_shape = im.shape
        path_my = os.path.join(path_,
                               'my',
                               'fold_'+str(fold),
                               'pred_res',
                               'img',
                               'frame'+str(f)+'.png')
        path_true = os.path.join(path_,
                                 'true',
                                 'fold_'+str(fold),
                                 'pred_res',
                                 'img',
                                 'frame'+str(f)+'.png')
        pred_my = utls.imread(os.path.join(path_my))
        pred_my = pred_my/np.max(pred_my)
        pred_my = transform.resize(pred_my, im_shape)

        pred_true = utls.imread(os.path.join(path_true))
        pred_true = pred_true/np.max(pred_true)
        pred_true = transform.resize(pred_true, im_shape)

        preds_my[-1].append(pred_my)
        preds_true[-1].append(pred_true)

ims_flat = [ims[i][j] for i in range(len(ims)) for j in range(len(folds))]
preds_my_flat = [preds_my[i][j] for i in range(len(ims)) for j in range(len(folds))]
preds_true_flat = [preds_true[i][j] for i in range(len(ims)) for j in range(len(folds))]

widths = [ims_flat[i].shape[1] for i in range(len(ims_flat))]
heights = [ims_flat[i].shape[0] for i in range(len(ims_flat))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [np.ceil((((ims_flat[i].shape[0]-min_height)/2,(ims_flat[i].shape[0]-min_height)/2), ((ims_flat[i].shape[1]-min_width)/2,(ims_flat[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims_flat))]

ims_crop = [util.crop(ims_flat[i], to_crop_im[i]) for i in range(len(ims_flat))]
preds_my_crop = [util.crop(preds_my_flat[i], to_crop_im[i]) for i in range(len(ims_flat))]
preds_true_crop = [util.crop(preds_true_flat[i], to_crop_im[i]) for i in range(len(ims_flat))]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims_crop))]
preds_my_res = [(transform.resize(preds_my_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims_crop))]
preds_true_res = [(transform.resize(preds_true_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims_crop))]

pw = 10
ims_pad = [util.pad(ims_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_res))]
preds_my_pad = [util.pad(preds_my_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_res))]
preds_true_pad = [util.pad(preds_true_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_res))]

all_ = np.concatenate([np.concatenate([ims_pad[i], preds_my_pad[i], preds_true_pad[i]], axis=1) for i in range(len(ims_pad))], axis=0)
all_ = np.split(all_, 2, axis=0)
all_ = np.concatenate(all_,axis=1)

#Write methods on image
header_height = 150
header = np.ones((header_height,all_.shape[1],3)).astype(np.uint8)*255
all_with_header = np.concatenate((header, all_),axis=0)

im_path = os.path.join(out_result_dir, 'all_learning.png')
print('Saving to: ' + im_path)
io.imsave(im_path, all_with_header)

img = Image.open(im_path)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("micross.ttf", 70)
texts = ['Original', 'KSPTrack', 'Manual','Original', 'KSPTrack', 'Manual']
# Get sizes of text
textsizes = [draw.textsize(texts[i],font=font) for i in range(len(texts))]
#Calculate placements
stride = all_.shape[1]//len(texts)//2
centers_cols = np.linspace(stride, all_.shape[1]-stride,len(texts))
left_coords = [int(centers_cols[i]-textsizes[i][0]//2) for i in range(len(texts))]
top_coord = header_height//3

file_out = os.path.join(out_result_dir,'all_learning_header.png')

for i in range(len(texts)):
    draw.text((left_coords[i], top_coord), texts[i],(0,0,0),font=font)

print('Saving to: ' + file_out)
img.save(file_out)
