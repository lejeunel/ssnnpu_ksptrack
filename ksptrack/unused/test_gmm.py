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
from sklearn import (preprocessing,mixture)
import scipy.io
import skimage.segmentation
import networkx as nx
import graphksp as gksp
from skimage import io
from skimage import (color,feature)
import my_imread as my
import datetime
import matplotlib.colors

def comp_bernoulli(p,min_val):
    if(p > 1-min_val):
        p = 1-min_val
    if(p<0):
        p = min_val

    res = np.log(p/(1-p))
    return res

def img_crop(im, w, h):
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    is_2d = len(im.shape) < 3
    half_border_width = int((imgwidth-w)/2)
    half_border_height = int((imgheight-h)/2)
    if is_2d:
        im_out = im[half_border_height-1:imgheight-half_border_height-1,
                    half_border_width-1:imgwidth-half_border_width-1]
    else:
        im_out = im[(half_border_height-1):(imgheight-half_border_height-1),
                    (half_border_width-1):(imgwidth-half_border_width-1),:]
    return im_out

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
gaze_radius = 10

seenDesc = list()
seenVals = np.array([]).reshape(0,3)
#seenVals = np.array([]).reshape(0,30)

img = my.imread(frameFileNames[0])
M = img.shape[0]
N = img.shape[1]
radius = 2
step = 1
P = np.ceil((M - radius*2) / step)
Q = np.ceil((N - radius*2) / step)
desc_list = list()

print('Getting DAISY descriptors')
for i in range(len(frameFileNames)):
    print('Frame ' + str(i))
    #img = color.rgb2hsv(my.imread(frameFileNames[i]))
    img = my.imread(frameFileNames[i])
    img_orig_shape = img.shape
    #img = np.pad(img,((radius, radius), (radius, radius)), 'constant')
    #desc = feature.daisy(color.rgb2gray(img),step=step,radius=radius)

    circle_idx,_ = spix.getCircleIdx((myGaze[i,3],myGaze[i,4]),img_orig_shape,10)
    this_vals = img[circle_idx[0],circle_idx[1],:]
    #this_hist = np.asarray([np.histogram(this_vals[:,i],range=(0,255),density=True)[0] for i in range(img.shape[2])]).ravel()
    #seenVals = np.append(seenVals,this_hist.reshape(1,-1),axis=0)
    #seenVals = np.append(seenVals,this_vals,axis=0)

    seenVals = np.append(seenVals,np.mean(this_vals,axis=0).reshape(1,-1),axis=0)

print('Done')

#data = dict()
#data['desc_list'] = desc_list
#data['seenVals'] = seenVals
#np.savez('daisy_desc.npz',**data)

n_components=4
print('fit a Gaussian Mixture Model with ' + str(n_components) + ' components')

clf = mixture.GaussianMixture(n_components=n_components, covariance_type='tied')
#x_train = seenVals[:,0:2]
x_train = seenVals
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
clf.fit(x_train_scaled)

bound = 10e-6
max_prob = 1-10e-6
prob_app = list()
#for i in range(len(frameFileNames)):
for i in range(2):
    print(i)
    #this_im = color.rgb2hsv(my.imread(frameFileNames[i]))
    this_im = my.imread(frameFileNames[i])
    #this_im = color.rgb2gray(this_im)
    prob_app.append(np.zeros((this_im.shape[0],this_im.shape[1])))
    for j in np.unique(labels[:,:,i]):
        this_mask = labels[:,:,i] == j
        idx_mask = np.where(this_mask)
        this_vals = np.mean(img[idx_mask[0],idx_mask[1],:],axis=0)
        #x = np.asarray([np.histogram(this_vals[:,i],range=(0,255),density=True)[0] for i in range(img.shape[2])]).ravel().reshape(1,-1)
        x = this_vals.reshape(1,-1)
        x = scaler.transform(x)
        this_score = clf.score(x)
        #the_prob = np.exp(this_score)
        the_prob = this_score
        #if the_prob == np.inf: the_prob = max_prob
        prob_app[i] += this_mask*this_score
        #prob_app[i] += this_mask*comp_bernoulli(the_prob,10e-11)

plt.imshow(-prob_app[i]); plt.colorbar();plt.show()
