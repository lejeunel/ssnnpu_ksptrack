import scipy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import superPixels as spix
import pickle as pk
import my_utils as utls
import progressbar
import gazeCsv as gaze
from skimage import (color, io, segmentation)
from sklearn import (mixture, metrics, preprocessing, decomposition)
from scipy import (ndimage,io)
import glob, itertools
import bagging as bag
import selective_search
import features
import shutil as sh

def relabel_desc_df(desc_df):
    #Relabel: labels, centroids_loc, seen_feats, sp_entr
    print('Relabeling desc_df')

    with progressbar.ProgressBar(maxval=np.unique(desc_df['frame']).shape[0]) as bar:
        for f in np.unique(desc_df['frame']):
        #for f in range(desc_df['frame'].shape[0]):
            bar.update(f)
            this_labels = desc_df[desc_df['frame'] == f]['sp_label']
            this_unique_labels = np.unique(this_labels).ravel()
            sorted_labels = np.arange(0,this_unique_labels.shape[0])
            for i in range(this_unique_labels.shape[0]):
                desc_df.loc[(desc_df['frame'] == f) & (desc_df['sp_label'] == this_unique_labels[i]),'sp_label'] = sorted_labels[i]

    return desc_df

def relabel(labels):
    #Relabel: labels, centroids_loc, seen_feats, sp_entr
    print('Relabeling labels')

    with progressbar.ProgressBar(maxval=labels.shape[2]) as bar:
        for i in range(labels.shape[2]):
            bar.update(i)
            this_labels = labels[...,i]
            sorted_labels = np.asarray(sorted(np.unique(this_labels).ravel()))
            if(np.any((sorted_labels[1:] - sorted_labels[0:-1])>1)):
                has_changed = True
                map_dict = {sorted_labels[i]:i for i in range(sorted_labels.shape[0])}
                this_labels = utls.relabel(this_labels,map_dict)
                labels[...,i] = this_labels

    return labels


#datasets = ['Dataset1','Dataset2','Dataset3','Dataset4','Dataset9','Dataset11','Dataset12','Dataset13','Dataset14','Dataset15','Dataset16','Dataset17','Dataset18','Dataset20']
#datasets = ['Dataset1','Dataset2','Dataset3','Dataset4','Dataset9','Dataset11','Dataset12','Dataset13','Dataset14','Dataset15','Dataset16','Dataset17','Dataset18']
#datasets = ['Dataset15']
datasets = ['Dataset13']

dir_root = '/home/laurent.lejeune/medical-labeling'
dir_frames = 'input-frames'

for i in range(len(datasets)):

    old_labels = io.loadmat(os.path.join(dir_root,datasets[i],dir_frames,'sp_labels.mat'))['sp_labels']

    new_labels = relabel(old_labels)

    data = dict()
    data['sp_labels'] = new_labels
    mat_file_out = os.path.join(dir_root,datasets[i],dir_frames,'sp_labels.npz')
    np.savez(mat_file_out,**data)


    sp_desc_df =  pd.read_pickle(
        os.path.join(dir_root,datasets[i],'precomp_descriptors', 'sp_desc_df.p'))

    ##Move old
    print('backing up old descriptor df')
    sh.copyfile(os.path.join(dir_root,datasets[i],'precomp_descriptors', 'sp_desc_df.p'),
              os.path.join(dir_root,datasets[i],'precomp_descriptors', 'sp_desc_df_old.p'))

    sp_desc_df = relabel_desc_df(sp_desc_df)
    #sp_desc_df.drop('hoof_b',axis=1,inplace=True)
    #sp_desc_df.drop('hoof_f',axis=1,inplace=True)

    sp_desc_df.to_pickle(os.path.join(dir_root,datasets[i],'precomp_descriptors', 'sp_desc_df.p'))

    #Relabel
