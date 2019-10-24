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
from labeling.cfgs import cfg
from PIL import Image, ImageFont, ImageDraw
import glob
import progressbar

"""
Makes plots self
"""

def gray2rgb(im):
    return (color.gray2rgb(im)*255).astype(np.uint8)

file_out = os.path.join(rd.root_dir,
                        'plots_results',
                        'videos')
if(not os.path.exists(file_out)):
    print('{} doesnt exist. Creating.'.format(file_out))
    os.mkdir(file_out)

cmap = plt.get_cmap('viridis')

for key in ['Slitlamp']:
#for key in rd.types: # Types

    # Get first gaze-set of every dataset
    for i in [0]:
    #for i in range(4):
        #My model
        dir_ksp = os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][i][0])

        conf = cfg.load_and_convert(os.path.join(dir_ksp,
                                                 'cfg.yml'))


        print('Type: {}. Dset: {}'.format(key, conf.ds_dir))


        # Load config
        dataset = learning_dataset.LearningDataset(conf)
        gt = dataset.gt

        exp_dir = '{}_{}'.format(key, i+1)
        dir_out = os.path.join(file_out, exp_dir)

        if(not os.path.exists(dir_out)):
            print('{} doesnt exist. Creating.'.format(dir_out))
            os.mkdir(dir_out)

        file_ksp = os.path.join(dir_ksp,
                                'results.npz')

        ksp = np.load(file_ksp)['ksp_scores_mat']
        ksp = [cmap((ksp[..., i]*255).astype(np.uint8))[..., 0:3]
               for i in range(ksp.shape[-1])]

        with progressbar.ProgressBar(maxval=len(conf.frameFileNames)) as bar:
            for f, i in zip(conf.frameFileNames, range(len(conf.frameFileNames))):
                bar.update(i)

                # Image
                cont_gt = segmentation.find_boundaries(
                    gt[..., i], mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(f)
                im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
                locs2d = csv.readCsv(os.path.join(conf.root_path,
                                                conf.ds_dir,
                                                conf.locs_dir,
                                                conf.csvFileName_fg))
                im =  csv.draw2DPoint(locs2d,
                                        i,
                                        im,
                                        radius=7)
                im = im/255

                im = np.concatenate((im, ksp[i].copy()), axis=1)
                im = (im*255).astype(np.uint8)

                io.imsave(os.path.join(dir_out, 'frame_{:04d}.png'.format(i+1)),
                          im)
