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
#import dataset as ds
#import learning_dataset
from labeling.exps import results_dirs as rd
from PIL import Image, ImageFont, ImageDraw
import glob

"""
Makes plots self
"""

def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

file_out = os.path.join(rd.root_dir, 'plots_results')
placehold = utls.imread(os.path.join(file_out, 'placeholder.png'))

n_sets_per_type = 2

dfs = []
# Self-learning

ims = []
ksp = []
vilar = []
g2s = []
mic17 = []
wtp = []

#for key in rd.res_dirs_dict_ksp.keys(): # Types
for key in rd.types: # Types
    ims.append([])
    ksp.append([])
    vilar.append([])
    g2s.append([])
    mic17.append([])
    wtp.append([])

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
        locs2d = csv.readCsv(os.path.join(confs[0].dataInRoot,
                                          confs[0].dataSetDir,
                                          confs[0].gazeDir,
                                          confs[0].csvFileName_fg))
        im =  csv.draw2DPoint(locs2d,
                                f,
                                im,
                                radius=7)
        ims[-1].append(im)

        file_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][dset][gset],
                                'results.npz')

        print('Loading (KSP): ' + file_ksp)
        npzfile = np.load(file_ksp)
        ksp_ = gray2rgb(npzfile['ksp_scores'][..., f])
        ksp[-1].append(ksp_)

        # vilar
        file_vilar = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_vilar[key][dset],
                                'vilar',
                                'pred_im_' + str(f) + '.npz')

        print('Loading (Vilar): ' + file_vilar)
        npzfile = np.load(file_vilar)
        vilar_ = gray2rgb(npzfile['pred'])
        vilar[-1].append(vilar_)

        # g2s
        file_g2s = os.path.join(rd.confs_dict_ksp[key][dset][gset].precomp_desc_path,
                                'g2s_rws.npz')

        if(os.path.exists(file_g2s)):
            print('Loading (g2s): ' + file_g2s)
            npzfile = np.load(file_g2s)
            g2s_ = gray2rgb(npzfile['rws'][f, ...]-1)
            g2s[-1].append(g2s_)
        else:
            print(file_g2s + ' does not exist')
            g2s[-1].append(placehold)

        # mic17
        file_mic17 = os.path.join(rd.root_dir,
                                    rd.res_dirs_dict_mic17[key][dset])

        if(os.path.exists(file_mic17)):
            print('Loading (mic17): ' + file_mic17)
            mic17_files = sorted(glob.glob(os.path.join(file_mic17, '*.png')))
            mic17_images = np.asarray([utls.imread(f) for f in mic17_files])
            mic17[-1].append(mic17_images[f])
        else:
            print(file_mic17 + ' does not exist')
            mic17[-1].append(placehold)

        # wtp
        file_wtp = os.path.join(rd.root_dir,
                                    rd.res_dirs_dict_wtp[key][dset],
                                    'preds.npz')

        if(os.path.exists(file_wtp)):
            print('Loading (WTP): ' + file_wtp)
            npzfile = np.load(file_wtp)
            # Get threshold of max F1

            #df = pd.read_csv(os.path.join(rd.root_dir,
            #                              rd.res_dirs_dict_wtp[key][dset],
            #                              'scores.csv'))
            #thr_ind = df.loc[0][-1]
            preds = npzfile['preds']
            #thr = sorted(np.unique(preds))[int(thr_ind)]

            #wtp_ = gray2rgb(preds[...,f] > thr)
            wtp_ = gray2rgb(preds[...,f])
            wtp[-1].append(wtp_)
        else:
            print(file_wtp + ' does not exist')
            wtp[-1].append(placehold)

ims = [ims[i][j] for i in range(len(ims)) for j in range(len(ims[i]))]
ksp = [ksp[i][j] for i in range(len(ksp)) for j in range(len(ksp[i]))]
mic17 = [mic17[i][j] for i in range(len(mic17)) for j in range(len(mic17[i]))]
g2s = [g2s[i][j] for i in range(len(g2s)) for j in range(len(g2s[i]))]
vilar = [vilar[i][j] for i in range(len(vilar)) for j in range(len(vilar[i]))]
wtp = [wtp[i][j] for i in range(len(wtp)) for j in range(len(wtp[i]))]

widths = [ims[i].shape[1] for i in range(len(ims))]
heights = [ims[i].shape[0] for i in range(len(ims))]
min_width = np.min(widths)
min_height = np.min(heights)

to_crop_im = [np.ceil((((ims[i].shape[0]-min_height)/2,(ims[i].shape[0]-min_height)/2), ((ims[i].shape[1]-min_width)/2,(ims[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_ksp = [np.ceil((((ksp[i].shape[0]-min_height)/2,(ksp[i].shape[0]-min_height)/2), ((ksp[i].shape[1]-min_width)/2,(ksp[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_vilar = [np.ceil((((vilar[i].shape[0]-min_height)/2,(vilar[i].shape[0]-min_height)/2), ((vilar[i].shape[1]-min_width)/2,(vilar[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_g2s = [np.ceil((((g2s[i].shape[0]-min_height)/2,(g2s[i].shape[0]-min_height)/2), ((g2s[i].shape[1]-min_width)/2,(g2s[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_mic17 = [np.ceil((((mic17[i].shape[0]-min_height)/2,(mic17[i].shape[0]-min_height)/2), ((mic17[i].shape[1]-min_width)/2,(mic17[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]
to_crop_wtp = [np.ceil((((wtp[i].shape[0]-min_height)/2,(wtp[i].shape[0]-min_height)/2), ((wtp[i].shape[1]-min_width)/2,(wtp[i].shape[1]-min_width)/2),(0,0))) for i in range(len(ims))]

ims_crop = [util.crop(ims[i], to_crop_im[i]) for i in range(len(ims))]
ksp_crop = [util.crop(ksp[i], to_crop_ksp[i]) for i in range(len(ims))]
mic17_crop = [util.crop(mic17[i], to_crop_mic17[i]) for i in range(len(ims))]
vilar_crop = [util.crop(vilar[i], to_crop_vilar[i]) for i in range(len(ims))]
g2s_crop = [util.crop(g2s[i], to_crop_g2s[i]) for i in range(len(ims))]
wtp_crop = [util.crop(wtp[i], to_crop_wtp[i]) for i in range(len(ims))]

f_size = ims_crop[0].shape

ims_res = [(transform.resize(ims_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
ksp_res = [(transform.resize(ksp_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
mic17_res = [(transform.resize(mic17_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
vilar_res = [(transform.resize(vilar_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
g2s_res = [(transform.resize(g2s_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]
wtp_res = [(transform.resize(wtp_crop[i], f_size)*255).astype(np.uint8) for i in range(len(ims))]

pw = 10
ims_pad = [util.pad(ims_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
ksp_pad = [util.pad(ksp_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
mic17_pad = [util.pad(mic17_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
vilar_pad = [util.pad(vilar_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
g2s_pad = [util.pad(g2s_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]
wtp_pad = [util.pad(wtp_res[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims))]

all_ = np.concatenate([np.concatenate([ims_pad[i], ksp_pad[i], mic17_pad[i], vilar_pad[i], g2s_pad[i], wtp_pad[i]], axis=1) for i in range(len(ims_pad))], axis=0)

#Write methods on image
header_height = 150
header = np.ones((header_height,all_.shape[1],3)).astype(np.uint8)*255
all_with_header = np.concatenate((header, all_),axis=0)

im_path = os.path.join(file_out, 'all_self.png')
print('Saving to: ' + im_path)
io.imsave(im_path, all_with_header)

img = Image.open(im_path)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("micross.ttf", 70)
texts = ['Original', 'KSPTrack', 'EEL', 'P-SVM', 'Gaze2Segment', 'DL-prior']
# Get sizes of text
textsizes = [draw.textsize(texts[i],font=font) for i in range(len(texts))]
#Calculate placements
stride = all_.shape[1]//len(texts)//2
centers_cols = np.linspace(stride, all_.shape[1]-stride,len(texts))
left_coords = [int(centers_cols[i]-textsizes[i][0]//2) for i in range(len(texts))]
top_coord = header_height//3

for i in range(len(texts)):
    draw.text((left_coords[i], top_coord), texts[i],(0,0,0),font=font)
img.save(os.path.join(file_out,'all_self_header.png'))
