#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import warnings
import numpy
import skimage.io
import features
import color_space
import selective_search
from scipy import (ndimage,io)
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import my_utils as utls


dataInRoot = '/home/laurent.lejeune/medical-labeling/'
dataSetDir = 'Dataset2'
frameDir = 'input-frames'
im_name = 'frame_0140.png'
out_dir = 'results'

image = os.path.join(dataInRoot,dataSetDir,frameDir,im_name)

color = 'rgb'
feature = ['texture','fill','color']
output = 'result'
k = 50
alpha = 1.0
import pdb; pdb.set_trace()
img = skimage.io.imread(image)

label = io.loadmat(os.path.join(dataInRoot,                                      dataSetDir,                                          frameDir,'sp_labels.mat'))['sp_labels'][...,0]


if len(img.shape) == 2:
    img = skimage.color.gray2rgb(img)

print('k:', k)
print('color:', color)
print('feature:', ' '.join(feature))


start_t = time.time()
mask = features.SimilarityMask('size' in feature, 'color' in feature, 'texture' in feature, 'fill' in feature)
#R: stores region label and its parent (empty if initial).
# record merged region (larger region should come first)
R, F, g = selective_search.hierarchical_segmentation(img, mask,F0=label)
end_t = time.time()
print('Built hierarchy in ' + str(end_t - start_t) + ' secs')

S = [399,313]

import my_utils as utls
out = selective_search.get_merge_candidates(g,S,0.6)
print('input candidates: ' + str(S))
print('output set: ' + str(out))


#if(os.path.exists(out_dir)):
#    print('Deleting content of dir: ' + out_dir)
#    fileList = os.listdir(out_dir)
#    for fileName in fileList:
#        #os.remove(out_dir+"/"+fileName)
#        os.remove(os.path.join(out_dir,fileName))
#
#if(not os.path.exists(out_dir)):
#    print('output directory does not exist... creating')
#    os.mkdir(out_dir)
#
#print('Saving images to dir: ' + str(out_dir))
#start_t = time.time()
#colors = generate_color_table(R)
#for depth, label in enumerate(F):
#    result = colors[label]
#    result = (result * alpha + img * (1. - alpha)).astype(numpy.uint8)
#    fn = "%s_%04d.png" % (os.path.splitext(im_name)[0], depth)
#    fn = os.path.join(out_dir,fn)
#    skimage.io.imsave(fn, result)
#    print('.', end="")
#    sys.stdout.flush()
#end_t = time.time()
#print()
#print('Saved images in ' + str(end_t - start_t) + ' secs')
