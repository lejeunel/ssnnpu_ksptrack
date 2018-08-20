#!/usr/bin/python3
import glob
from gco_python.pygco import cut_from_graph
import warnings
import itertools
import pickle
import progressbar
import pandas as pd
import cPickle as pk
import sys
import numpy as np
import os
import gazeCsv as gaze
import matplotlib.pyplot as plt
import tb as tb
import datetime
import superPixels as spix
import flowNet as fn
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage.transform import rescale
import networkx as nx
import graphksp as gksp
from skimage import (color,io,segmentation)
from sklearn import (mixture,metrics,preprocessing,decomposition)
import my_imread as my
import datetime
import yaml
import hashlib
import json

def my_gc(scores,pm,frameFileNames,frame_idx,gamma,lambda_,bg_prob=0.5,fg_prob=0.5):

    small = np.finfo(np.float).eps

    gc_maps = np.zeros(scores.shape)
    gc_scores = []

    for j in frame_idx:
      min_p = 0.01
      n = 1000

      this_pom = pm[j,:,:].copy()

      this_fg = fg_prob*np.ones(this_pom.shape)
      this_fg[scores[j,:,:] > 0] = 1

      this_bg = bg_prob*np.ones(this_pom.shape)
      this_bg[this_pom == 0] = 1

      this_pm_fg = np.clip(this_pom,a_max=1.0,a_min=min_p)

      this_fg_costs = -np.log(this_fg+small)/-np.log(small)-gamma*np.log(this_pm_fg+small)/-np.log(small)
      this_bg_costs = -np.log(this_bg+small)/-np.log(small)-gamma*np.log(1-this_pm_fg+small)/-np.log(small)

      #Digitize
      bin_min = np.min([this_fg_costs,this_bg_costs])
      bin_max = np.max([this_fg_costs,this_bg_costs])
      bins = np.linspace(bin_min,bin_max,n)


      this_fg_costs = np.digitize(this_fg_costs,bins)
      this_bg_costs = np.digitize(this_bg_costs,bins)

      unaries = ((np.dstack([this_fg_costs, this_bg_costs]).copy("C"))).astype(np.int32)

      # use the gerneral graph algorithm
      # first, we construct the grid graph
      inds = np.arange(scores[j,:,:].size).reshape(scores[j,:,:].shape)
      horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
      vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

      orig_image = color.rgb2gray(my.imread(frameFileNames[j]))
      dx_im = orig_image[:,0:-1] - orig_image[:,1::]
      dy_im = orig_image[0:-1,:] - orig_image[1::,:]
      mean_squared = np.mean(np.hstack([(dx_im**2).ravel(),(dy_im**2).ravel()]))
      beta = 1./(2*mean_squared)
      horz_weights = np.exp(-dx_im**2*beta)
      vert_weights = np.exp(-dy_im**2*beta)

      #Digitize
      bin_min = np.min([np.min(horz_weights),np.min(vert_weights)])
      bin_max = np.max([np.max(horz_weights),np.max(vert_weights)])
      bins = np.linspace(bin_min,bin_max,n)
      horz_weights = np.digitize(horz_weights,bins)
      vert_weights = np.digitize(vert_weights,bins)

      #pairwise = -100* np.eye(2, dtype=np.int32)
      pairwise = 1-np.eye(2, dtype=np.int32)

      weights = np.vstack([lambda_*horz_weights.reshape(-1,1), lambda_*vert_weights.reshape(-1,1)])
      edges_idx = np.vstack([horz, vert]).astype(np.int32)
      edges = np.hstack([edges_idx, weights]).astype(np.int32)

      this_gc = np.logical_not(cut_from_graph(edges, unaries.reshape(-1, 2), pairwise).reshape(scores[j,:,:].shape))
      gc_maps[j,:,:] = this_gc
    #conf_mat = metrics.confusion_matrix(gt.ravel(),gc_maps.ravel())
    #precision = float(conf_mat[1,1])/float(conf_mat[1,1] + conf_mat[0,1])
    #recall = float(conf_mat[1,1])/float(conf_mat[1,1] + conf_mat[1,0])
    #f1 = float(2*conf_mat[1,1])/float(2*conf_mat[1,1] + conf_mat[0,1] + conf_mat[1,0])
    #gc_scores.append((lambda_,gamma,f1,precision,recall))
    #print(gc_scores[-1])
    return gc_maps

#Load back results
res_dir = ['/home/laurent.lejeune/server/medical-labeling/data/Dataset2/results/2017-02-06_11-59-34_exp',
           '/home/laurent.lejeune/server/medical-labeling/data/Dataset11/results/2017-02-12_14-06-56_exp/',
           '/home/laurent.lejeune/server/medical-labeling/data/Dataset12/results/2017-02-13_14-05-15_exp/',
           '/home/laurent.lejeune/server/medical-labeling/data/Dataset9/results/2017-02-11_17-33-52_exp/']

#Extract segmentations with optimal parameters
thr_pm = [0.09,0.8,0.93,0.73]
gamma = [0.084,0.8,0.1,0.4]
lambda_ = [5,50,20,5]
bg_prob = [0.4,0.5,0.5,0.5]
gc_maps = []
pm_maps = []
frames = []
frames_idx = [np.array([44]),
              np.array([44]),
              np.array([25]),
              np.array([20]),
]
for i in range(len(res_dir)):

  print('processing: ' + res_dir[i])
  npz_file = np.load(os.path.join(res_dir[i],'results_gc.npz'))
  frameFileNames = np.load(os.path.join(res_dir[i],'results.npz'))['frameFileNames']
  label_contours = np.load(os.path.join(res_dir[i],'results.npz'))['labelContourMask']
  myGaze = np.load(os.path.join(res_dir[i],'results.npz'))['myGaze']
  scores = npz_file['scores']
  #this_pm_maps = npz_file['pom_mat'] > thr_pm[i]
  this_pm_maps = npz_file['pom_mat']

  #Extract ground-truth files
  frameInd = np.arange(0,len(frameFileNames))
  gtFileNames = [frameFileNames[j].replace('input-frames','ground_truth-frames') for j in range(len(frameFileNames))]
  gt = np.zeros((len(frameFileNames),scores.shape[1],scores.shape[2]))
  for k in range(len(gtFileNames)):
      this_gt = my.imread(gtFileNames[k])
      gt[k,:,:] = (this_gt[:,:,0] > 0)

  if(label_contours.shape[0] != len(frameFileNames)):
    label_contours = label_contours.transpose((2,0,1))

  if(this_pm_maps.shape[0] != len(frameFileNames)):
    this_pm_maps = this_pm_maps.transpose((2,0,1))

  this_gc_maps = my_gc(scores,this_pm_maps,frameFileNames,frames_idx[i],gamma[i],lambda_[i],bg_prob[i])
  gc_maps.append(this_gc_maps[frames_idx[i][0],:,:])
  pm_maps.append(this_pm_maps[frames_idx[i][0]] > thr_pm[i])

  this_seq_frames = []
  for j in range(len(frames_idx[i])):
    im = my.imread(frameFileNames[frames_idx[i][j]])
    if(im.shape[2] > 3): im = im[:,:,0:3]
    cont_gt = segmentation.find_boundaries(gt[frames_idx[i][j],:,:],mode='thick')
    idx_cont_gt = np.where(cont_gt)
    idx_cont_sp= np.where(label_contours[frames_idx[i][j],:,:])
    im[idx_cont_sp[0],idx_cont_sp[1],:] = (255,255,255)
    im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
    im = gaze.drawGazePoint(myGaze,frames_idx[i][j],im,radius=7)
    this_seq_frames.append(im)
  frames.append(this_seq_frames)

from matplotlib import gridspec
nrows = 3
fig = plt.figure(figsize=(8,9))
fig, axes = plt.subplots(ncols=4, nrows = 3, sharey=True)
w_max = np.max(np.asarray([frames[i][0].shape[1] for i in range(len(frames))]))
#ratios = np.asarray([float(frames[i][0].shape[0])/h_max for i in range(len(frames))])
ratios = np.asarray([float(w_max)/float(frames[i][0].shape[1]) for i in range(len(frames))])
#gs = gridspec.GridSpec(nrows, len(frames), width_ratios=ratios)
#Adjust heights

for f in range(axes.shape[1]):
  im = rescale(frames[f][0],ratios[f])
  pm = rescale(pm_maps[f],ratios[f])
  gc = rescale(gc_maps[f],ratios[f])
  print(im.shape,pm.shape,gc.shape)
  axes[0,f].imshow(im)
  axes[0,f].axis('off')
  axes[1,f].imshow(pm,cmap=plt.get_cmap('gray'));
  axes[1,f].axis('off')
  axes[2,f].imshow(gc,cmap=plt.get_cmap('gray'));
  axes[2,f].axis('off')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0.01)
#fig.tight_layout()
fig.show()
#fig.savefig(os.path.join('/home/laurent.lejeune/Desktop','all.eps'), dpi=800, bbox_inches='tight')
fig.savefig(os.path.join('/home/laurent.lejeune/Desktop','all.eps'), dpi=200)

#Extract all frames for video
gc_maps = []
pm_maps = []
frames = []
#for i in range(len(res_dir)):
for i in [1,2,3]:

  print('processing: ' + res_dir[i])
  npz_file = np.load(os.path.join(res_dir[i],'results_gc.npz'))
  frameFileNames = np.load(os.path.join(res_dir[i],'results.npz'))['frameFileNames']
  label_contours = np.load(os.path.join(res_dir[i],'results.npz'))['labelContourMask']
  myGaze = np.load(os.path.join(res_dir[i],'results.npz'))['myGaze']
  scores = npz_file['scores']
  #this_pm_maps = npz_file['pom_mat'] > thr_pm[i]
  this_pm_maps = npz_file['pom_mat']

  #Extract ground-truth files
  frameInd = np.arange(0,len(frameFileNames))
  gtFileNames = [frameFileNames[j].replace('input-frames','ground_truth-frames') for j in range(len(frameFileNames))]
  gt = np.zeros((len(frameFileNames),scores.shape[1],scores.shape[2]))
  for k in range(len(gtFileNames)):
      this_gt = my.imread(gtFileNames[k])
      gt[k,:,:] = (this_gt[:,:,0] > 0)

  if(label_contours.shape[0] != len(frameFileNames)):
    label_contours = label_contours.transpose((2,0,1))

  if(this_pm_maps.shape[0] != len(frameFileNames)):
    this_pm_maps = this_pm_maps.transpose((2,0,1))

  this_gc_maps = my_gc(scores,this_pm_maps,frameFileNames,np.arange(0,len(frameFileNames)),gamma[i],lambda_[i],bg_prob[i])

  this_seq_frames = []
  for j in range(len(frameFileNames)):
    fig, axes = plt.subplots(1, 2,figsize=(5.5,2.2))
    im = my.imread(frameFileNames[j])
    if(im.shape[2] > 3): im = im[:,:,0:3]
    cont_gt = segmentation.find_boundaries(gt[j,:,:],mode='thick')
    idx_cont_gt = np.where(cont_gt)
    im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
    im = gaze.drawGazePoint(myGaze,j,im,radius=7)
    axes[0].imshow(im);
    axes[0].axis('off')
    axes[1].imshow(this_gc_maps[j,:,:],cmap=plt.get_cmap('gray'))
    axes[1].axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0)
    #fig.show()
    fig.savefig(os.path.join(res_dir[i],'final_'+os.path.splitext(os.path.split(frameFileNames[j])[1])[0]+'.png'), dpi=200)
    fig.clf()


#Plot cochlea comic book
res_dir = '/home/laurent.lejeune/server/medical-labeling/data/Dataset11/results/2017-02-12_14-06-56_exp/'
pom_mat = np.load(os.path.join(res_dir,'results_gc.npz'))['pom_mat']
scores = np.load(os.path.join(res_dir,'results_gc.npz'))['scores']
label_contours = np.load(os.path.join(res_dir,'results.npz'))['labelContourMask']
myGaze = np.load(os.path.join(res_dir,'results.npz'))['myGaze']
frameFileNames = np.load(os.path.join(res_dir,'results.npz'))['frameFileNames']
gtFileNames = [frameFileNames[i].replace("input-frames","ground_truth-frames") for i in range(len(frameFileNames))]
gt = np.asarray([(my.imread(gtFileNames[i])>0)[:,:,0] for i in range(len(gtFileNames))])
frames_idx = np.arange(0,len(frameFileNames))
gamma = 0.8
lambda_ = 50
gc_maps = my_gc(scores,pom_mat,frameFileNames,frames_idx,gamma,lambda_)

to_plot = np.array([1,20,44,67,88])
nrows = 3
fig, axes = plt.subplots(nrows, to_plot.size,figsize=(5.3,2.4))

for f in range(axes.shape[1]):
  im = my.imread(frameFileNames[to_plot[f]])
  if(im.shape[2] > 3): im = im[:,:,0:3]
  cont_gt = segmentation.find_boundaries(gt[to_plot[f],:,:],mode='thick')
  idx_cont_gt = np.where(cont_gt)
  idx_cont_sp= np.where(label_contours[to_plot[f],:,:])
  im[idx_cont_sp[0],idx_cont_sp[1],:] = (255,255,255)
  im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
  im = gaze.drawGazePoint(myGaze,to_plot[f],im,radius=7)

  this_gc_map = gc_maps[to_plot[f],:,:]
  axes[0,f].imshow(im);
  axes[0,f].axis('off')
  axes[1,f].imshow(pom_mat[to_plot[f],:,:],cmap=plt.get_cmap('viridis'));
  axes[1,f].axis('off')
  axes[2,f].imshow(this_gc_map,cmap=plt.get_cmap('gray'));
  axes[2,f].axis('off')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
#fig.show()
fig.savefig(os.path.join('/home/laurent.lejeune/Desktop','cochlea.eps'), dpi=200)
