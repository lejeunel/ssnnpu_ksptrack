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
import numpy as np

all_datasets = ['Dataset23']
#                'Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']
#all_datasets = ['Dataset30', 'Dataset31', 'Dataset32', 'Dataset33']

test = []

for ds in all_datasets:

    print(ds)
    test.append([])
    frame_extension = '.png'
    frame_dir = 'input-frames'
    frame_prefix = 'frame_'
    frame_digits = 4
    data_root_dir = '/home/laurent.lejeune/medical-labeling/'
    dataset_dir = ds
    filenames = utls.makeFrameFileNames(
        frame_prefix, frame_digits, frame_dir,
        data_root_dir, dataset_dir, frame_extension)
    filenames = [os.path.splitext(os.path.split(f)[-1])[0] for f in filenames]

    path = os.path.join(data_root_dir, dataset_dir, 'EE')

    labels = scipy.io.loadmat(os.path.join(path,'sp_labels_ml.mat'))['labels']

    diff_ = 0
    for f in range(len(filenames)):

        print('frame {}/{}'.format(f+1,len(filenames)))
        #print('----')
        try:
            feats = scipy.io.loadmat(os.path.join(path,'overfeat_wide', filenames[f] + '.mat'))['features']
            #print('num. unique labels: ' + str(np.unique(labels[...,f]).shape[0]))
            #print('num. rows feats: ' + str(feats.shape[0]))
            #print('max. label labels: ' + str(np.unique(labels[...,f])[-1]))
            #print('max. label feats: ' + str(feats[-1, 0]))
            diff_labels = np.unique(labels[...,f]) - feats[:,0]
            diff_ += np.sum(diff_labels)
            test[-1].append(diff_)
        except ValueError:
            print(os.path.join(path,'overfeat_wide', filenames[f] + '.mat'))
            print('!!! FAILED')

#ds = 'Dataset01'
#
#filenames = utls.makeFrameFileNames(
#    frame_prefix, frame_digits, frame_dir,
#    data_root_dir, dataset_dir, frame_extension)
#filenames = [os.path.splitext(os.path.split(f)[-1])[0] for f in filenames]
#path = os.path.join(data_root_dir, dataset_dir, 'EE')
#labels = scipy.io.loadmat(os.path.join(path,'sp_labels_ml.mat'))['labels']
#
#for f in range(100, len(filenames)):
#
#    print('frame {}/{}'.format(f+1,len(filenames)))
#    feats = np.load(os.path.join(path,'overfeat_wide', filenames[f] + '.npz'))['features']
#    scipy.io.savemat(os.path.join(path,'overfeat_wide', filenames[f] + '.mat'),{'features': feats})
