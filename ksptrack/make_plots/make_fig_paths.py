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
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.exps import results_dirs as rd
from labeling.utils import csv_utils as csv
from labeling.cfgs import cfg
from PIL import Image, ImageFont, ImageDraw
import glob
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage.measurements import center_of_mass
import copy

"""
Makes plots self
"""

def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

file_out = os.path.join(rd.root_dir, 'plots_results')

dir_ = 'Dataset30/results/2018-06-01_14-22-56_for_paths'

conf = cfg.load_and_convert(os.path.join(rd.root_dir, dir_, 'cfg.yml'))

res = np.load(os.path.join(rd.root_dir, dir_, 'results.npz'))
list_paths_back = res['list_paths_back']
list_paths_forw = res['list_paths_for']

color_sps = (0, 0, 255)

ims = []
ksp = []

# Load config
dataset = learning_dataset.LearningDataset(conf)
gt = dataset.gt

# Image
ims = [utls.imread(f) for f in conf.frameFileNames]
cont_gts = [segmentation.find_boundaries(gt[..., i])
            for i in range(gt.shape[-1])]
idx_cont_gts = [np.where(g) for g in cont_gts]
path_csv = os.path.join(conf.dataInRoot,
                        conf.dataSetDir,
                        conf.gazeDir,
                        conf.csvFileName_fg)

labels = dataset.get_labels()
dataset.load_labels_contours_if_not_exist()
label_conts = dataset.labelContourMask
locs = utls.readCsv(path_csv)

for i in range(len(ims)):
        ims[i][idx_cont_gts[i][0], idx_cont_gts[i][1], :] = (255, 0, 0)
        #ims[i] = csv.draw2DPoint(locs,
        #                        i,
        #                        ims[i],
        #                        radius=7)

font = ImageFont.truetype('micross.ttf', 20)

ims_for = copy.deepcopy(ims)

# Forward paths
for p, p_ind in zip(list_paths_forw[-1], range(len(list_paths_forw[-1]))):
    for s in p:
        s_ = s[0]
        mask = labels[..., s_[0]] == s_[1]
        cont_mask = segmentation.find_boundaries(mask)
        idx_mask_cont = np.where(cont_mask)
        idx_mask = np.where(mask)

        ims_for[s_[0]][idx_mask[0], idx_mask[1], : ] = (255, 255, 255)
        ims_for[s_[0]][idx_mask_cont[0], idx_mask_cont[1], : ] = color_sps
        im_pil = Image.fromarray(ims_for[s_[0]])
        draw = ImageDraw.Draw(im_pil)
        coord_center = center_of_mass(mask)
        textsize = draw.textsize(str(p_ind), font=font)
        left = int(coord_center[1] - textsize[0]//2)
        top = int(coord_center[0] - textsize[1]//2)
        #top = coord_center[0]
        draw.text((left, top),
                  str(p_ind),
                  color_sps,
                  font=font)
        ims_for[s_[0]] = np.array(im_pil)

ims_back = copy.deepcopy(ims)

# Forward paths
for p, p_ind in zip(list_paths_back[-1], range(len(list_paths_back[-1]))):
    for s in p:
        s_ = s[0]
        mask = labels[..., s_[0]] == s_[1]
        cont_mask = segmentation.find_boundaries(mask)
        idx_mask_cont = np.where(cont_mask)
        idx_mask = np.where(mask)

        ims_back[s_[0]][idx_mask[0], idx_mask[1], : ] = (255, 255, 255)
        ims_back[s_[0]][idx_mask_cont[0], idx_mask_cont[1], : ] = color_sps
        im_pil = Image.fromarray(ims_back[s_[0]])
        draw = ImageDraw.Draw(im_pil)
        coord_center = center_of_mass(mask)
        textsize = draw.textsize(str(p_ind), font=font)
        left = int(coord_center[1] - textsize[0]//2)
        top = int(coord_center[0] - textsize[1]//2)
        #top = coord_center[0]
        draw.text((left, top),
                  str(p_ind),
                  color_sps,
                  font=font)
        ims_back[s_[0]] = np.array(im_pil)

f_ind = np.arange(0, len(conf.frameFileNames), 15)

upsamp_ratio = 2

ims_back_upsamp = [transform.rescale(ims_back[i], upsamp_ratio)
                   for i in range(len(ims_back)) if i in f_ind]
ims_for_upsamp = [transform.rescale(ims_for[i], upsamp_ratio)
                  for i in range(len(ims_back)) if i in f_ind]

region = [(upsamp_ratio*100, upsamp_ratio*350),
          (upsamp_ratio*300, upsamp_ratio*550), ]
shape = ims_back_upsamp[0].shape
crop = ((region[0][0], shape[0]-region[0][1]),
        (region[1][0], shape[1]-region[1][1]),(0, 0))
ims_back_less = [(util.crop(ims_back_upsamp[i], crop)*255).astype(np.uint8)
                 for i in range(len(ims_back_upsamp))]
ims_for_less = [(util.crop(ims_for_upsamp[i], crop)*255).astype(np.uint8)
                for i in range(len(ims_back_upsamp))]

pw = 10
ims_back_pad = [util.pad(ims_back_less[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_back_less))]
ims_for_pad = [util.pad(ims_for_less[i], ((pw,pw),(pw,pw),(0,0)), mode='constant', constant_values=255) for i in range(len(ims_for_less))]

ims_back_concat = np.concatenate(ims_back_pad, axis=1)
ims_for_concat = np.concatenate(ims_for_pad, axis=1)

ims_all = np.concatenate((ims_back_concat, ims_for_concat), axis=0)

header_height = 150
header = np.ones((header_height, ims_all.shape[1],3)).astype(np.uint8)*255

time_labels = [str(f+1) + '/' + str(len(conf.frameFileNames)) for f in f_ind]
textsizes = [draw.textsize(time_labels[i],font=font)
             for i in range(len(time_labels))]

stride = ims_all.shape[1]//len(time_labels)//2
centers_cols = np.linspace(stride, ims_all.shape[1]-stride,len(time_labels))
left_coords = [int(centers_cols[i]-textsizes[i][0]//2)
               for i in range(len(time_labels))]
top_coord = header_height//3

im_header = Image.fromarray(header)
draw = ImageDraw.Draw(im_header)
font = ImageFont.truetype("micross.ttf", 70)

for i in range(len(time_labels)):
    draw.text((left_coords[i], top_coord), time_labels[i],(0,0,0),font=font)

header = np.array(im_header)

texts = ['Forward', 'Backward']

header_height = 200
header_direc = np.ones((header_height,ims_all.shape[0],3)).astype(np.uint8)*255
header_direc = Image.fromarray(header_direc, 'RGB')

draw = ImageDraw.Draw(header_direc)
font = ImageFont.truetype("micross.ttf", 70)

# Get sizes of text
textsizes = [draw.textsize(texts[i],font=font) for i in range(len(texts))]

#Calculate placements
stride = ims_all.shape[0]//len(texts)//2
centers_cols = np.linspace(stride, ims_all.shape[0]-stride,len(texts))
left_coords = [int(centers_cols[i]-textsizes[i][0]//2) for i in range(len(texts))]
top_coord = header_height//3

for i in range(len(texts)):
    draw.text((left_coords[i], top_coord), texts[i],(0,0,0),font=font)

header_direc = np.array(header_direc)
hackshit = np.ones((header_direc.shape[0],
                    header.shape[0],
                    3)).astype(np.uint8)*255
header_direc = np.concatenate((header_direc, hackshit),axis=1)
header_direc = np.rot90(header_direc)

all_top_header = np.concatenate((header, ims_all), axis=0)
all_left_header = np.concatenate((header_direc, all_top_header), axis=1)

save_out = os.path.join(file_out,'brain_paths.png')
print('Saving image to {}'.format(save_out))
io.imsave(save_out, all_left_header.astype(np.uint8))
plt.imshow(all_left_header);plt.show()
