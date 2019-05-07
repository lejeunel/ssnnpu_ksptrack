import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import my_utils as utls
from sklearn.metrics import f1_score,confusion_matrix,auc
from sklearn.metrics import confusion_matrix
import progressbar
import gazeCsv as gaze
import selective_search
from scipy import (ndimage,io)
import features
import skimage.io
import networkx as nx

dir_in_root = '/home/laurent.lejeune/medical-labeling'

#dir_result = 'Dataset12/results/2017-06-14_14-47-27_exp/'
dir_results = 'Dataset2/results/2017-06-14_22-26-11_exp'

iter_ = 0

npz_file = np.load(os.path.join(dir_in_root,dir_results,'pm_scores_iter_' + str(iter_) + '.npz'))
pm_scores = npz_file['pm_scores_fg']
ksp_scores = npz_file['ksp_scores_mat']

npz_file = np.load(os.path.join(dir_in_root,dir_results,'results.npz'))
list_ksp = npz_file['list_ksp']

with open(os.path.join(dir_in_root,dir_results, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)

g_ss = []
F = []

color = 'rgb'
feature = ['texture','fill','color']

label = io.loadmat(os.path.join(conf.root_path,                                      conf.ds_dir,                                           conf.frameDir,'sp_labels.mat'))['sp_labels']
mask = features.SimilarityMask('size' in feature, 'color' in feature, 'texture' in feature, 'fill' in feature)

frames = np.array([100])
#Make graphs of selective search for every frame
#for i in range(len(conf.frameFileNames)):
for i in range(frames.shape[0]):
    this_label = label[...,frames[i]]

    img = skimage.io.imread(conf.frameFileNames[frames[i]])
    _, f, g = selective_search.hierarchical_segmentation(img, mask,F0=this_label,return_stacks=True)
    g_ss.append(g)
    F.append(f)

thr = 0.5

all_sps = np.asarray(utls.get_node_list_tracklets(list_ksp[iter_]))
merged_scores = []
#Merge
print('Extracting marked SPs and merging')
idx = np.where(all_sps[:,0] == frames[0] )
sps = all_sps[idx,1].ravel().tolist()
merged, unmerged = selective_search.get_merge_candidates(g_ss[0],sps,thr)
stacks = nx.get_node_attributes(g_ss[0],'stack')

tmp_score = np.zeros(F[0][0].shape).astype(int)
color = 1
for j in range(len(merged)):
    tmp_score += color*((F[i][stacks[merged[j][0]]] == merged[j][0]).astype(int))
    color+=1

for j in range(len(unmerged)):
    tmp_score += color*((F[i][stacks[unmerged[j][0]]] == unmerged[j][0]).astype(int))
    color+=1
merged_scores.append(tmp_score)

pix_orig = np.sum(ksp_scores[frames[0],...].ravel()>0)
pix_merged = np.sum(merged_scores[0].ravel() > 0)
gain_pix = (pix_merged-pix_orig)/pix_orig
print('Gain in pixels: ' + str(gain_pix))

plt.subplot(121)
plt.imshow(ksp_scores[frames[0],...]);
plt.title('KSP')
plt.subplot(122)
plt.imshow(merged_scores[0]);
plt.title('Merged')
plt.suptitle('Gain in pixels: ' + str(np.round(100*gain_pix)) + '%')
plt.show()
