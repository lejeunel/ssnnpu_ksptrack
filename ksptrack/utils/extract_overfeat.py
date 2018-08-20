import my_utils as utls
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import scipy.io
from skimage.transform import resize
from skimage.morphology import binary_dilation
from skimage.morphology import square
from skimage.io import imsave
import os
from nets.overfeat.my_overfeat import OverfeatFeatureExtractor
import string
import random
import os, string, subprocess, sys, os.path
#from sklearn_theano.feature_extraction import OverfeatTransformer


def run_on_dset(dataset_dir):

    # read image
    labelMatPath = 'EE/sp_labels_ml.mat'
    data_root_dir = '/home/laurent.lejeune/medical-labeling/'
    frame_extension = '.png'
    frame_dir = 'input-frames'
    frame_prefix = 'frame_'
    frame_digits = 4

    wide = True
    patch_size = 231

    if(wide):
        feat_out_dir = 'EE/overfeat_wide'
    else:
        feat_out_dir = 'EE/overfeat'

    labels = scipy.io.loadmat(os.path.join(data_root_dir,dataset_dir,labelMatPath))['labels']

    filenames = utls.makeFrameFileNames(
        frame_prefix, frame_digits, frame_dir,
        data_root_dir, dataset_dir, frame_extension)

    # initialize overfeat. Note that this takes time, so do it only once if possible
    overfeat_dir = '/home/laurent.lejeune/Documents/OverFeat'
    overfeat_extr = OverfeatFeatureExtractor(os.path.join(overfeat_dir,'data','default'),
                                             os.path.join(overfeat_dir,
                                                          'bin',
                                                          'linux_64',
                                                          'cuda',
                                                          'overfeatcmd_cuda'))

    out_path = os.path.join(data_root_dir,dataset_dir,feat_out_dir)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    selem = square(2)
    for i in range(len(filenames)):
        patches = []
        this_filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        outfile = os.path.join(data_root_dir,dataset_dir,feat_out_dir,this_filename)
        if(not os.path.isfile(outfile + '.npz')):
            print('frame {}/{}'.format(i+1, len(filenames)))
            features = []
            image = utls.imread(filenames[i])
            if(image.shape[2]>3): image = image[:,:,0:3]
            for j in np.unique(labels[:,:,i]):
                if(wide):
                    this_mask = labels[:,:,i] == j
                    this_mask = binary_dilation(this_mask,selem)
                    this_mask_idx = np.where(this_mask)
                    this_mask_labels = np.unique(labels[this_mask_idx[0],this_mask_idx[1],i])
                    this_mask = np.in1d(labels[:,:,i],this_mask_labels).reshape(image.shape[0],image.shape[1])
                else:
                    this_mask = labels[:,:,i] == j
                i_mask, j_mask = np.where(this_mask)
                w = max(j_mask) - min(j_mask)
                h = max(i_mask) - min(i_mask)
                if(w < h):
                    cols_to_add = h-w+1
                    idx_i = np.arange(min(i_mask), max(i_mask) + 1).astype(int)
                    idx_j = np.arange(min(j_mask) - np.floor(cols_to_add/2), max(j_mask) + np.ceil(cols_to_add/2)).astype(int)
                elif(w > h):
                    rows_to_add = w-h+1
                    idx_i = np.arange(min(i_mask)-np.floor(rows_to_add/2), max(i_mask) + np.ceil(rows_to_add/2)).astype(int)
                    idx_j = np.arange(min(j_mask), max(j_mask) + 1).astype(int)
                else:
                    idx_i = np.arange(min(i_mask), max(i_mask) + 1)
                    idx_j = np.arange(min(j_mask), max(j_mask) + 1)
                patches.append(resize(image.take(idx_i,mode='wrap',axis=0).take(idx_j,mode='wrap',axis=1),
                                     (patch_size,patch_size)).astype(np.float32))


            X = np.asarray(patches)
            features = overfeat_extr.get_feats_overfeat(X, np.unique(labels[:,:,i]), '/tmp')
            data = dict()
            data['features'] = features

            np.savez_compressed(outfile+'.npz',**data)
            scipy.io.savemat(outfile+'.mat',data)
        else:
            print("File: " + outfile + " exists. Delete to recompute...")

#all_datasets = ['Dataset00', 'Dataset01', 'Dataset02', 'Dataset03']
all_datasets = ['Dataset23']
#all_datasets = ['Dataset10', 'Dataset11', 'Dataset12', 'Dataset13']
#all_datasets = ['Dataset20', 'Dataset21', 'Dataset22', 'Dataset23']
#all_datasets = ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']

for ds in all_datasets:
    print("-------------------")
    print("Running on: " + ds)
    print("-------------------")
    run_on_dset(ds)
