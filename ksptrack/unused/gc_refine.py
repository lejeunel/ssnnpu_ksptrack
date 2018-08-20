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
import networkx as nx
import graphksp as gksp
from skimage import (color,io,segmentation)
from sklearn import (mixture,metrics,preprocessing,decomposition)
import my_imread as my
import datetime
import yaml
import hashlib
import json
from imdescrip.extractor import extract_smp
from imdescrip.descriptors.ScSPM import ScSPM

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def cfg():

    #Paths, dirs, names ...
    dataInRoot = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/'
    dataOutRoot = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/'
    dataOutResultDir = ''
    resultDir = 'results'
    gazeDir = 'gaze-measurements'
    gtFrameDir = 'ground_truth-frames'
    dataSetDir = 'Dataset12'
    fileOutPrefix = 'exp'
    framePrefix = 'frame_'
    frameExtension = '.png'
    frameDigits = 4
    frameDir = 'input-frames'
    csvFileName = 'video1.csv'
    csvName = os.path.join(dataInRoot,dataSetDir,gazeDir, csvFileName)

    comment = 'Comment of experiment'

    # Sequences---------------
    #Cochlea
    #seqStart = 1
    #seqEnd = 96

    #Tweezer
    #seqStart = 140
    #seqEnd = 260

    #tumor
    #seqStart = 67
    #seqEnd = 126

    #slit-lamp
    seqStart = 1
    seqEnd = 194
    #-------------------------

    #Descriptor method (sift/hsift/bow)
    feat_descriptorMethod = 'colors'
    #feat_descriptorMethod = 'overfeat'

    #Descriptors/codebooks ready-to-load.
    feats_files_dir = 'precomp_descriptors'
    feats_compute = True #This value should not be changed here.

    #Segmentation SEEDS
    num_superpixels = 200
    prior = 2
    num_levels = 4
    num_histogram_bins = 5
    num_iterations = 4

    #Segmentation from TPS
    #labelMatPath = 'TSP_flows/sp_labels_tsp.mat'
    labelMatPath = 'TSP_flows'
    #labelMatPath = ''

    #SIFT keypoints step size
    step_size = 1

    #Histogram descriptor bins
    nbins = 8

    #BOW
    dictionarySize = 15

    #flow parameters
    sig_a = 0.5
    sig_r_in = 0.4
    sig_r_trans = 0.3

    #sig_t = 10
    #sig_t = winWidth/2
    gaze_radius = 5
    #gaze_radius = 70
    #max_paths = 50
    max_paths = 40

    #Graph parameters
    normNeighbor = 0.2 #Cut edges outside neighborhood
    normNeighbor_in = 0.5 #Cut edges outside neighborhood

    testing = False #Won't save results if True

    return locals()

def get_scores_rates(sets,winInd,frameInd,gt_dir,frameFileNames,Kmax=None,mode='all'):

    #Extract ground-truth files
    gt = np.zeros((len(frameFileNames),labels.shape[0],labels.shape[1]))
    for i in range(len(frameFileNames)):
        base,fname = os.path.split(frameFileNames[i])
        this_gt = my.imread(os.path.join(gt_dir,fname))
        gt[i,:,:] = (this_gt[:,:,0] > 0)

    #scores = np.zeros((len(frameFileNames),labels.shape[0],labels.shape[1]))
    scores = fn.score_path_sets(sets,labels,'s','t',set_idx = winInd,frame_idx = frameInd,Kmax=Kmax, mode=mode)

    return scores

def makeFrameFileNames(framePrefix,seqStart,seqEnd,frameDigits,frameDir,dataInRoot,dataSetDir,frameExtension):

    frameFileNames = []
    path = dataInRoot+dataSetDir+'/'+frameDir+'/'
    idx = np.arange(seqStart,seqEnd+1)
    frameFileNames = []
    formatStr =  path + framePrefix + '%0' + str(frameDigits) + 'd'

    for i in range(idx.shape[0]):
        frameFileNames.append(str(formatStr%idx[i])+frameExtension)

    return frameFileNames

def readCsv(csvName,seqStart,seqEnd):
    return np.loadtxt(open(csvName,"rb"),delimiter=";",skiprows=5)[seqStart:seqEnd+1,:]

def getDataOutDir(dataOutRoot,dataSetDir,resultDir,fileOutPrefix,testing):

    now = datetime.datetime.now()
    dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")

    dataOutDir = os.path.join(dataOutRoot,dataSetDir,resultDir)
    dataOutResultDir = os.path.join(dataOutDir,dateTime+'_' + fileOutPrefix)

    print(dataOutResultDir)
    if (not os.path.exists(dataOutResultDir)) and (not testing):
        os.makedirs(dataOutResultDir)

    return dataOutResultDir


data= dict()

cfg_dict = cfg()
c = Bunch(cfg_dict)

#Write config to result dir
dataOutDir =  getDataOutDir(c.dataOutRoot,c.dataSetDir,c.resultDir,c.fileOutPrefix,c.testing)

#Make frame file names from seqStart and seqEnd
gtFileNames = makeFrameFileNames(c.framePrefix,c.seqStart,c.seqEnd,c.frameDigits,c.gtFrameDir,c.dataInRoot,c.dataSetDir,c.frameExtension)

frameFileNames = makeFrameFileNames(c.framePrefix,c.seqStart,c.seqEnd,c.frameDigits,c.frameDir,c.dataInRoot,c.dataSetDir,c.frameExtension)

myGaze = readCsv(c.csvName,c.seqStart-1,c.seqEnd)
gt_positives = tb.getPositives(gtFileNames)

precomp_desc_path = os.path.join(c.dataInRoot,c.dataSetDir,'precomp_descriptors')
labelMatPath = os.path.join(c.dataInRoot,c.dataSetDir,c.frameDir,'TSP_flows')

print('loading data (labels,descriptors,...)')
centroids_loc = pd.read_pickle(os.path.join(precomp_desc_path,'centroids_loc_df.p'))
labels = scipy.io.loadmat(os.path.join(labelMatPath,'sp_labels_tsp.mat'))['sp_labels']
npzfile = np.load(os.path.join(precomp_desc_path,'sp_labels_tsp_contours.npz'),fix_imports=True,encoding='bytes')
labelContourMask = npzfile['labelContourMask']

print("Loading seen descriptors")
seen_feats_df = pd.read_pickle(os.path.join(precomp_desc_path,'seen_feats_df.p')).T

print("Loading superpixels descriptors")
sp_desc_df = pd.read_pickle(os.path.join(precomp_desc_path,'sp_desc_df.p'))

print("Loading link data")
sp_link_df = pd.read_pickle(os.path.join(precomp_desc_path,'sp_link_df.p'))

print("Loading seen-to-sp histogram intersections")
sp_entr_df = pd.read_pickle(os.path.join(precomp_desc_path,'sp_entr_df.p'))

print("Loading POM")
sp_pom_df = pd.read_pickle(os.path.join(precomp_desc_path,'sp_pom_df.p'))

print("Loading POM matrix")
pom_mat = np.load(os.path.join(precomp_desc_path,'pom_mat.npz'))['pom_mat']

#Test plots
frameInd = np.arange(0,len(frameFileNames))
gtFileNames = makeFrameFileNames(c.framePrefix,c.seqStart,c.seqEnd,c.frameDigits,c.gtFrameDir,c.dataInRoot,c.dataSetDir,c.frameExtension)
gt_dir = os.path.join(c.dataInRoot,c.dataSetDir,c.gtFrameDir)
#Extract ground-truth files
gt = np.zeros((len(frameFileNames),labels.shape[0],labels.shape[1]))
for i in range(len(frameFileNames)):
    base,fname = os.path.split(frameFileNames[i])
    this_gt = my.imread(os.path.join(gt_dir,fname))
    gt[i,:,:] = (this_gt[:,:,0] > 0)

gaze_points = np.delete(myGaze,(0,1,2,5),axis=1)

# Loading back
#dir_out = '/home/laurent.lejeune/server/medical-labeling/data/Dataset11/results/2017-02-12_14-06-56_exp'
#dir_out = '/home/laurent.lejeune/server/medical-labeling/data/Dataset2/results/2017-02-06_11-59-34_exp'
#dir_out = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/Dataset9/results/2017-02-11_17-33-52_exp'
dir_out = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/Dataset12/results/2017-02-13_14-05-15_exp'
npz_file = np.load(os.path.join(dir_out,'results.npz'))

labels = npz_file['labels']
labelContourMask = npz_file['labelContourMask']
sets = npz_file['sets']
frameFileNames = npz_file['frameFileNames']
#pom_mat = npz_file['pom_mat']
gt_dir = os.path.split(frameFileNames[0])[0].replace("input-frames","ground_truth-frames")

scores = get_scores_rates(sets,np.array([0]),frameInd,gt_dir,frameFileNames,Kmax=370)

#gamma = np.linspace(1,1.5,10)
gamma = np.array([0.1])

lambda_ = 20

tpr_gc = []
fpr_gc = []

small = np.finfo(np.float).eps

if(pom_mat.shape[0] != len(frameFileNames)):
  pom_mat = pom_mat.transpose((2,0,1))
norm_scores = scores.astype(float)/np.max(scores)
heat_maps = np.zeros(scores.shape)
gc_maps = np.zeros(scores.shape)
gc_scores = []
#with progressbar.ProgressBar(maxval=gamma.shape[0]) as bar:
for i in range(gamma.shape[0]):
    #for j in range(len(frameFileNames)):
    for j in np.asarray([25]):
      print("(j,tau) = (" + str(j+1) + "/" + str(len(frameFileNames)) + ", " + str(gamma[i]) + ")")
      min_p = 0.01
      n = 1000

      this_pom = pom_mat[j,:,:].copy()

      this_fg = 0.5*np.ones(this_pom.shape)
      this_fg[scores[j,:,:] > 0] = 1

      this_bg = 0.5*np.ones(this_pom.shape)
      this_bg[this_pom == 0] = 1

      this_pm_fg = np.clip(this_pom,a_max=1.0,a_min=min_p)

      #this_fg_costs = -gamma[i]*np.log(this_fg+small)/-np.log(small)-np.log(this_pm_fg+small)/-np.log(small)
      #this_bg_costs = -gamma[i]*np.log(this_bg+small)/-np.log(small)-np.log(1-this_pm_fg+small)/-np.log(small)
      this_fg_costs = -np.log(this_fg+small)/-np.log(small)-gamma[i]*np.log(this_pm_fg+small)/-np.log(small)
      this_bg_costs = -np.log(this_bg+small)/-np.log(small)-gamma[i]*np.log(1-this_pm_fg+small)/-np.log(small)

      #Digitize
      bin_min = np.min([this_fg_costs,this_bg_costs])
      bin_max = np.max([this_fg_costs,this_bg_costs])
      bins = np.linspace(bin_min,bin_max,n)


      this_fg_costs = np.digitize(this_fg_costs,bins)
      this_bg_costs = np.digitize(this_bg_costs,bins)
      #this_fg_costs = (n*this_fg_costs).astype(np.int32)
      #this_bg_costs = (n*this_bg_costs).astype(np.int32)

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
    #gc_scores.append((lambda_,gamma[i],f1,precision,recall))
    #print(gc_scores[-1])

ind = 25
plt.subplot(221); plt.imshow(color.label2rgb(gt[ind,:,:],my.imread(frameFileNames[ind])));
plt.subplot(222); plt.imshow(gc_maps[ind,:,:]); plt.title('heat map')
plt.subplot(223); plt.imshow(scores[ind,:,:]);plt.title('score map')
plt.subplot(224); plt.imshow(pom_mat[ind,:,:]);plt.title('POM')
plt.show()

print('Getting F1-scores on probability map')
min_p = 0.1
pm_scores = []
thr_pm = np.unique(pom_mat)
thr_pm = thr_pm[(thr_pm > min_p)]
thr_pm_ = thr_pm[np.linspace(0,thr_pm.shape[0]-1,40).astype(int)]
thr_pm_ = np.array([0.93])
with progressbar.ProgressBar(maxval=thr_pm_.shape[0]) as bar:
  for i in range(thr_pm_.shape[0]):
    bar.update(i)
    seg_pm = pom_mat >= thr_pm_[i]
    conf_mat = metrics.confusion_matrix(gt.ravel(),seg_pm.ravel())
    precision = float(conf_mat[1,1])/float(conf_mat[1,1] + conf_mat[0,1])
    recall = float(conf_mat[1,1])/float(conf_mat[1,1] + conf_mat[1,0])
    f1 = float(2*conf_mat[1,1])/float(2*conf_mat[1,1] + conf_mat[0,1] + conf_mat[1,0])
    pm_scores.append((thr_pm_[i],f1,precision,recall))
    print(pm_scores[-1])

#Saving data
fileOut = os.path.join(dir_out,'results_gc.npz')
data = dict()
data['scores'] = scores
data['pom_mat']= pom_mat
print("Saving stuff to: ", fileOut)
np.savez(fileOut, **data)
print("done")

#Load back results
res_dir = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/Dataset11/results/2017-02-17_09-32-49_exp/'

npz_res = np.load(os.path.join(res_dir,'results.npz'))
labelContourMask = npz_res['labelContourMask']
labels = npz_res['labels']
frameFileNames = npz_res['frameFileNames']
myGaze = npz_res['myGaze']

npz_res = np.load(os.path.join(res_dir,'results_gc.npz'))

print("AUC_pom= " + str(npz_res['auc_pom']))
print("AUC_gc= " + str(npz_res['auc_gc']))

heat_maps = npz_res['heat_maps']
pom_mat = npz_res['pom_mat']

tpr_gc_arr = np.asarray(sorted(tpr_tmp))
fpr_gc_arr = np.asarray(sorted(fpr_tmp))
tpr_ksp_arr = np.asarray(sorted(tpr_ksp))[0:-2]
fpr_ksp_arr = np.asarray(sorted(fpr_ksp))[0:-2]

#Plot ROC curves
auc_gc = metrics.auc(fpr_gc_arr,tpr_gc_arr,reorder=True)
plt.plot(fpr_gc_arr,tpr_gc_arr,'ro-',label='GC')
plt.plot(fpr_pom,tpr_pom,'b',label='POM')
plt.title('AUC:' + str(np.around(auc_gc,decimals=4)))
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.savefig(os.path.join(dir_out,'ROC.pdf'))
plt.show()
plt.clf()

frames = np.arange(0,len(frameFileNames))
with progressbar.ProgressBar(maxval=frames.shape[0]) as bar:
  for f in frames:
    bar.update(f)
    im = my.imread(frameFileNames[f])
    if(im.shape[2] > 3): im = im[:,:,0:3]
    cont_gt = segmentation.find_boundaries(gt[f,:,:],mode='thick')
    idx_cont_gt = np.where(cont_gt)
    idx_cont_sp= np.where(labelContourMask[f,:,:])
    im[idx_cont_sp[0],idx_cont_sp[1],:] = (255,255,255)
    im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
    im = gaze.drawGazePoint(myGaze,f,im,radius=7)
    plt.subplot(131)
    plt.imshow(im);plt.axis('off')
    plt.subplot(132)
    plt.imshow(pom_mat[f,:,:],cmap=plt.get_cmap('viridis'));plt.axis('off')
    plt.subplot(133)
    plt.imshow(gc_maps[f,:,:],cmap=plt.get_cmap('viridis'));plt.axis('off')
    plt.subplots_adjust(wspace=.05)
    plt.savefig(os.path.join(dir_out,'res_'+os.path.split(frameFileNames[f])[1]), dpi=300, bbox_inches='tight')

#Cochlea
to_plot = np.array([1,20,44,67,90])

#Tweezer
to_plot = np.array([1,20,44,67,90])

#Brain
to_plot = np.array([1,10,20,30])

#Slit-lamp
to_plot = np.array([1,20,40,60,80])

#to_plot = np.array([1,44,67])
nrows = 3
fig, axes = plt.subplots(nrows, to_plot.size,figsize=(2.8,1))

for f in range(axes.shape[1]):
  im = my.imread(frameFileNames[to_plot[f]])
  if(im.shape[2] > 3): im = im[:,:,0:3]
  cont_gt = segmentation.find_boundaries(gt[to_plot[f],:,:],mode='thick')
  idx_cont_gt = np.where(cont_gt)
  idx_cont_sp= np.where(labelContourMask[to_plot[f],:,:])
  im[idx_cont_sp[0],idx_cont_sp[1],:] = (255,255,255)
  im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
  im = gaze.drawGazePoint(myGaze,to_plot[f],im,radius=7)

  this_heat_map = heat_maps[to_plot[f],:,:]
  this_heat_map -= np.min(this_heat_map)
  this_heat_map /= np.max(this_heat_map)
  axes[0,f].imshow(im);
  axes[0,f].axis('off')
  axes[1,f].imshow(pom_mat[to_plot[f],:,:],cmap=plt.get_cmap('viridis'));
  axes[1,f].axis('off')
  axes[2,f].imshow(this_heat_map,cmap=plt.get_cmap('viridis'));
  axes[2,f].axis('off')

fig.subplots_adjust(wspace=0.01,hspace=0.01,top=1,bottom=0)
fig.savefig(os.path.join(dataOutDir,'all.eps'), dpi=800, bbox_inches='tight')

#Plot column
to_plot = 44
im = my.imread(frameFileNames[to_plot])
if(im.shape[2] > 3): im = im[:,:,0:3]
cont_gt = segmentation.find_boundaries(gt[to_plot,:,:],mode='thick')
idx_cont_gt = np.where(cont_gt)
idx_cont_sp= np.where(labelContourMask[:,:,to_plot])
im[idx_cont_sp[0],idx_cont_sp[1],:] = (255,255,255)
im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
im = gaze.drawGazePoint(myGaze,to_plot,im,radius=7)
this_heat_map = gc_maps[to_plot,:,:]
this_heat_map -= np.min(this_heat_map)
this_heat_map /= np.max(this_heat_map)
plt.subplot(311)
plt.imshow(im);
plt.axis('off')
plt.subplot(312)
plt.imshow(pom_mat[to_plot,:,:],cmap=plt.get_cmap('viridis'));
plt.axis('off')
plt.subplot(313)
plt.imshow(this_heat_map,cmap=plt.get_cmap('viridis'));
plt.axis('off')
plt.subplots_adjust(wspace=0.01,hspace=0.01,top=1,bottom=0)
#plt.show()
plt.savefig(os.path.join(dataOutDir,'all.eps'), dpi=800, bbox_inches='tight')
