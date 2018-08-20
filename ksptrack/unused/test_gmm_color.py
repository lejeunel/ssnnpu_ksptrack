#!/usr/bin/python3
import cv2
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
from sklearn import (mixture,decomposition,preprocessing,cluster)
import matplotlib.colors
from scipy import (ndimage,cluster)
import tb as tb

def comp_bernoulli(p,min_val):

    if(p > 1-min_val):
        p = 1-min_val
    if(p<0):
        p = min_val

    res = np.log(p/(1-p))
    return res

def comp_bernoulli_mat(p,min_val):

    above = np.where(p>1-min_val)
    below = np.where(p<=0)
    p[above[0],above[1]] = 1-min_val
    p[below[0],below[1]] = min_val

    res = np.log(p/(1-p))
    return res

def get_sift_densely(img):

    step = 1
    kpDense = [cv2.KeyPoint(x, y, step) for y in range(0, img.shape[0], step)  for x in range(0, img.shape[1], step)]
    kps,des = sift.compute((color.rgb2gray(img)*255).astype(np.uint8),kpDense)

    return des.reshape(img.shape[0],img.shape[1],-1)

def extract_normalize_sift(des,kps_idx):

    des = des[kps_idx[0],kps_idx[1],:]
    norms = np.linalg.norm(des,axis=1)
    non_zero_norms = np.nonzero(norms)
    des[non_zero_norms,:] = des[non_zero_norms,:]/norms[non_zero_norms].reshape(-1,1)
    aboves = np.where(des>=0.2)
    des[aboves[0],aboves[1]] = 0.2
    norms = np.linalg.norm(des,axis=1)
    non_zero_norms = np.nonzero(norms)
    des[non_zero_norms,:] = des[non_zero_norms,:]/norms[non_zero_norms].reshape(-1,1)

    return des

#desc_file = '/home/laurent.lejeune/server/medical-labeling/data/Dataset2/precomp_descriptors/-5387396212591884047'
desc_file = '/home/laurent.lejeune/server/medical-labeling/data/Dataset9/precomp_descriptors/074c8d04e0400f6cf8be925dce10ac70'
print('loading data (labels,descriptors,...)')
npzfile = np.load(desc_file+'.npz',fix_imports=True,encoding='bytes')
labels = npzfile['labels']
labelContourMask = npzfile['labelContourMask']
centroidsLoc = pickle.load( open( desc_file + '_descloc.pkl', "rb" ) )

#results_file ='/home/laurent.lejeune/server/medical-labeling/data/Dataset2/results/2016-11-16_15-28-08_exp/'
results_file ='/home/laurent.lejeune/server/medical-labeling/data/Dataset9/results/2016-11-30_13-51-42_exp/'
#results_file ='/home/laurent.lejeune/server/medical-labeling/data/Dataset9/results/2016-11-30_11-50-39_exp/'
npzfile = np.load(results_file+'results.npz',fix_imports=True,encoding='bytes')
frameFileNames = npzfile['frameFileNames']
myGaze = npzfile['myGaze']
gaze_radius = 5
gmm_components = 5

sift = cv2.xfeatures2d.SIFT_create(sigma=1)
dense_sift = list()
print("Getting seen region colors and SIFT descriptors")
seen_sift = np.array([]).reshape(0,128)
seen_rgb = np.array([]).reshape(0,3)
for i in range(len(frameFileNames)):
    print(str(i+1) + '/' + str(len(frameFileNames)))
    img_rgb = my.imread(frameFileNames[i])
    if(img_rgb.shape[2] > 3): img_rgb = img_rgb[:,:,0:3]
    img = img_rgb
    #img = color.rgb2hsv(img_rgb)
    this_gaze = gaze.gazeCoord2Pixel(myGaze[i,3],myGaze[i,4],img.shape[1],img.shape[0])
    _,this_mask = spix.getValuesInCircle(img,(this_gaze[1],this_gaze[0]),gaze_radius)
    this_colors = img[np.where(this_mask)[0],np.where(this_mask)[1]]
    idx_to_keep = np.asarray([i for i in range(this_colors.shape[0]) if(np.all(this_colors[i,:] != 0))])
    print(idx_to_keep.shape)

    idx_mask = (np.where(this_mask)[0][idx_to_keep],np.where(this_mask)[1][idx_to_keep])
    this_mask = np.zeros((this_mask.shape))
    this_mask[idx_mask[0],idx_mask[1]] = 1
    dense_sift.append(get_sift_densely(img_rgb))
    this_sift = extract_normalize_sift(dense_sift[-1],idx_mask)
    _,this_vals = spix.getMeanColorInCircle(img,(myGaze[i,3],myGaze[i,4]),gaze_radius,normCst=1)
    seen_sift = np.append(seen_sift,this_sift,axis=0)
    seen_rgb = np.append(seen_rgb,this_vals[idx_to_keep,:],axis=0)

dense_sift = np.asarray(dense_sift)
dense_sift_lin = dense_sift.reshape(-1,128)

n_components_pca = 50
n_samples_pca = 900
print("PCA on seen regions")
pca = decomposition.PCA(n_components_pca)
sift_for_bow_subs = dense_sift_lin[np.random.randint(0,dense_sift_lin.shape[0],n_samples_pca),:]
pca.fit(sift_for_bow_subs)
sift_for_bow_subs = pca.transform(sift_for_bow_subs)

print('Generating SIFT codebook on ' + str(sift_for_bow_subs.shape[0]) + ' samples  with ' + str(n_components_pca) + ' clusters')
codebook, distortion = cluster.vq.kmeans(sift_for_bow_subs, n_components_pca,thresh=1)

print('Generating SIFT codes on ' + str(dense_sift.shape[0]) + ' images')
dense_sift_bow = list()
for j in range(dense_sift.shape[0]):
    this_im_sift = dense_sift[j].reshape(-1,128)
    this_im_sift = pca.transform(this_im_sift)
    this_code_sift, dist = cluster.vq.vq(this_im_sift, codebook)
    this_code_sift = this_code_sift.reshape(img.shape[0],img.shape[1])
    dense_sift_bow.append(this_code_sift)

print('Generating bag-of-words on ' + str(dense_sift.shape[0]) + ' images')
sp_bags = list()
for i in range(len(dense_sift_bow)):
    this_im = my.imread(frameFileNames[i])[:,:,0:3]
    sp_bags.append([])
    for j in np.unique(labels[:,:,i]):
        this_mask = labels[:,:,i] == j
        this_mask_idx = np.where(this_mask)
        this_hist = np.histogram(dense_sift_bow[i][this_mask_idx[0],this_mask_idx[1]],bins=n_components_pca,density=True)[0].reshape(1,-1)
        this_mean_color = np.mean(this_im[this_mask_idx[0],this_mask_idx[1],:],axis=0).reshape(1,-1)
        sp_bags[-1].append((j,np.concatenate((this_hist,this_mean_color),axis=1)))

print('Generating bag-of-words on ' + str(dense_sift.shape[0]) + ' seen regions')

print("Clustering seen region with GMM")
clf = mixture.GaussianMixture(n_components=gmm_components, covariance_type='tied')
#x_train = seenVals[:,0:2]
clf.fit(seenVals)

print("Assigning costs to superpixels")

sp_prob_arr = np.zeros((len(frameFileNames),labels.shape[0],labels.shape[1]))
sp_costs = list()
for i in range(len(frameFileNames)):
    print(str(i+1) + '/' + str(len(frameFileNames)))
    this_im_rgb = my.imread(frameFileNames[i])
    if(this_im_rgb.shape[2] > 3): this_im_rgb = this_im_rgb[:,:,0:3]
    #this_im = color.rgb2hsv(this_im_rgb)
    this_im = this_im_rgb
    sp_costs.append([])
    for j in np.unique(labels[:,:,i]):
        this_mask = labels[:,:,i] == j
        idx_mask = np.where(this_mask)
        this_sift = extract_normalize_sift(dense_sift[i],idx_mask)
        this_vals = this_im[idx_mask[0],idx_mask[1],:]
        x = np.concatenate((this_vals,this_sift),axis=1)
        x = pca.transform(x)
        the_score = clf.score(x)
        the_prob = np.exp(the_score)
        sp_prob_arr[i,idx_mask[0],idx_mask[1]] = the_prob

#plt.imshow(sp_costs_arr[10,:,:]); plt.show()

#for i in range(len(sp_costs_arr)):
#    prob_plot = sp_costs_arr[i,:,:]
#    prob_plot -= np.min(prob_plot)
#    prob_plot /= np.max(prob_plot)
#    plt.subplot(121)
#    this_im = my.imread(frameFileNames[i])
#    plt.imshow(this_im)
#    plt.subplot(122)
#    plt.imshow(comp_bernoulli_mat(prob_plot,10e-6))
#    plt.colorbar();
#    #plt.show()
#    plt.savefig('brats_' + str(i) + '.png')
#    plt.clf()

idx = 25
prob_plot = sp_prob_arr[idx,:,:]
prob_plot -= np.min(prob_plot)
prob_plot /= np.max(prob_plot)
plt.subplot(121)
this_im = my.imread(frameFileNames[idx])
plt.imshow(this_im)
plt.subplot(122)
plt.imshow(-comp_bernoulli_mat(prob_plot,10e-12))
#plt.imshow(prob_plot)
plt.colorbar();
plt.show()
#plt.savefig('brats_' + str(i) + '.png')
#plt.clf()
