#!/usr/bin/python3
import sys
import numpy as np
import os
import gazeCsv as gaze
import pickle
import matplotlib.pyplot as plt
import graphmatching as gm
import superPixels as spix
import flowNet as fn
from sklearn import preprocessing
import scipy.io
import skimage.segmentation
import networkx as nx
import graphksp as gksp
from skimage import io
from skimage import (color,feature)
import my_imread as my
import datetime
from sklearn import mixture
import matplotlib.colors

def comp_bernoulli(p,min_val):
    if(p > 1-min_val):
        p = 1-min_val
    if(p<0):
        p = min_val

    res = np.log(p/(1-p))
    return res

def bhatta_coeff ( hist1,  hist2):
    # calculate mean of hist1
    h1_ = np.mean(hist1);

    # calculate mean of hist2
    h2_ = np.mean(hist2);

    hist_shape = hist1.shape[0]
    # calculate score
    score = 0;
    for i in range(hist_shape):
        score += np.sqrt( hist1[i] * hist2[i] );
    score = np.sqrt( 1 - ( 1 / np.sqrt(h1_*h2_*hist_shape*hist_shape) ) * score );
    return score;

desc_file = '/home/laurent.lejeune/server/medical-labeling/data/Dataset2/precomp_descriptors/-5387396212591884047'
print('loading data (labels,descriptors,...)')
npzfile = np.load(desc_file+'.npz',fix_imports=True,encoding='bytes')
labels = npzfile['labels']
labelContourMask = npzfile['labelContourMask']
avgDesc = pickle.load( open( desc_file + '_desc.pkl', "rb" ) )
centroidsLoc = pickle.load( open( desc_file + '_descloc.pkl', "rb" ) )

results_file ='/home/laurent.lejeune/server/medical-labeling/data/Dataset2/results/2016-11-16_15-28-08_exp/'
npzfile = np.load(results_file+'results.npz',fix_imports=True,encoding='bytes')
frameFileNames = npzfile['frameFileNames']
myGaze = npzfile['myGaze']
gaze_radius = 30

seenDesc = list()
seenVals = np.array([]).reshape(0,3)
desc_list = list()
img = my.imread(frameFileNames[0])
desc_file_path = '/home/laurent.lejeune/server/medical-labeling/data/Dataset2/daisy_desc/'

n_bins = 10
chans = 3
seen_hist = np.array([]).reshape(0,chans*n_bins)
for i in range(len(frameFileNames)):
    print('Frame ' + str(i))
    img = my.imread(frameFileNames[i])
    #img = color.rgb2hsv(img)

    _,this_vals = spix.getMeanColorInCircle(img,(myGaze[i,3],myGaze[i,4]),gaze_radius,normCst=1)
    this_hist = np.asarray([np.histogram(this_vals[:,i],bins=n_bins,density=True)[0] for i in range(chans)])
    this_hist = this_hist.ravel().reshape(1,-1)

    #this_vals = np.append(this_vals,desc_center,axis=1)
    seen_hist = np.append(seen_hist,this_hist,axis=0)

print('Done')

#for i in range(len(frameFileNames)):
prob_app = list()
for i in range(2):
    this_img = my.imread(frameFileNames[i])
    #this_img = color.rgb2hsv(this_img)
    prob_app.append(np.zeros((this_img.shape[0],this_img.shape[1])))
    for j in np.unique(labels[:,:,i]):
        this_mask = labels[:,:,i] == j
        idx_mask = np.where(this_mask)
        x = this_img[idx_mask[0],idx_mask[1],:]
        import pdb; pdb.set_trace()
        this_hist = np.asarray([np.histogram(x[:,i],bins=n_bins,density=True)[0] for i in range(chans)])
        this_hist = this_hist.ravel().reshape(1,-1)
        max_coeff = 0
        for k in range(seen_hist.shape[0]):
            this_coeff = bhatta_coeff( this_hist.ravel(),  seen_hist[k,:].ravel())
            if (this_coeff > max_coeff ): max_coeff = this_coeff
        the_prob = max_coeff
        print(max_coeff)
        prob_app[i] += this_mask*comp_bernoulli(the_prob,10e-6)

plt.imshow(prob_app[0]); plt.colorbar();plt.title('Bhattacharyya');plt.show()
