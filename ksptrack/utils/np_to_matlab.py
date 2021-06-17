import os

import numpy as np
import pandas as pd
import progressbar
from scipy import io
from skimage import io

import my_utils as utls


def labels_to_ml(labels, path, fname='sp_labels_ml.mat'):
    #Relabel: labels, centroids_loc, seen_feats, sp_entr
    print('Relabeling labels')
    f_out = os.path.join(path,fname)

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

    print('Saving to: ' + f_out)
    mdict = {'labels': labels}
    io.savemat(f_out, mdict)

def sp_desc_to_ml(sp_desc_df,path,fname='sp_desc.mat'):
    #Relabel: labels, centroids_loc, seen_feats, sp_entr
    f_out = os.path.join(path,fname)
    sp_desc_mat = sp_desc_df.as_matrix()

    print('Saving to: ' + f_out)
    io.savemat(f_out,{'sp_desc': sp_desc_mat})

datasets = ['Dataset00','Dataset01','Dataset02','Dataset03',
            'Dataset10','Dataset11','Dataset12','Dataset13',
            'Dataset20','Dataset21','Dataset22','Dataset23',
            'Dataset30','Dataset31','Dataset22','Dataset33']

dir_root = '/home/laurent.lejeune/medical-labeling'
dir_frames = 'input-frames'
sp_desc_fnames = ['sp_desc_ung_g1.p',
                  'sp_desc_ung_g2.p',
                  'sp_desc_ung_g3.p',
                  'sp_desc_ung_g4.p',
                  'sp_desc_ung_g5.p']

for i in range(len(datasets)):
    labels = np.load(os.path.join(dir_root,
                                  datasets[i],
                                  dir_frames,
                                  'sp_labels.npz'))['sp_labels']
    path_out = os.path.join(dir_root,datasets[i],'EE')
    if(not os.path.exists(path_out)):
        os.mkdir(path_out)

    labels_to_ml(labels,path_out)

    for sp_desc_fname in sp_desc_fnames:
        sp_desc_df = pd.read_pickle(os.path.join(dir_root,
                                                datasets[i],
                                                'precomp_descriptors',
                                                sp_desc_fname))
        sp_desc_mat_fname = os.path.splitext(sp_desc_fname)[0] + '.mat'
        sp_desc_to_ml(sp_desc_df,
                      path_out,
                      fname=sp_desc_mat_fname)
