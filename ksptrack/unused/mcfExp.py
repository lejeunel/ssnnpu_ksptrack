from sklearn.metrics import f1_score
import glob
from pygco import cut_from_graph
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import cfg
import pandas as pd
import pickle as pk
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
import superPixels as spix
import flowNet as fn
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import networkx as nx
import graphksp as gksp
import my_utils as utls
import calc_features

data = dict()

cfg_dict = cfg.cfg()
conf = cfg.Bunch(cfg_dict)

#Write config to result dir
conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir, conf.resultDir,
                                  conf.out_dir_prefix, conf.testing)

#Make frame file names from seqStart and seqEnd
gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
gtFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.seqStart, conf.seqEnd, conf.frameDigits, conf.truth_dir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

conf.frameFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.seqStart, conf.seqEnd, conf.frameDigits, conf.frameDir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

conf.myGaze_fg = utls.readCsv(conf.csvName_fg, conf.seqStart, conf.seqEnd+1)
conf.myGaze_bg = utls.readCsv(conf.csvName_bg, conf.seqStart, conf.seqEnd+1)
gt_positives = utls.getPositives(gtFileNames)

if (conf.labelMatPath != ''):
    conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.ds_dir, conf.frameDir,
                                  conf.labelMatPath)

conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                   conf.feats_dir)

#Enable steps
comp_feats = False
comp_superpix = False
comp_pom_fg = True
comp_pom_bg = True

# ---------- Descriptors/superpixel costs
calc_features.run(conf, comp_feats, comp_superpix, comp_pom_fg, comp_pom_bg)

print('loading data (labels,descriptors,...)')
centroids_loc = pd.read_pickle(
    os.path.join(conf.precomp_desc_path, 'centroids_loc_df.p'))
labels = scipy.io.loadmat(
    os.path.join(conf.labelMatPath, 'sp_labels_tsp.mat'))['sp_labels'][:, :, np.arange(conf.seqStart, conf.seqEnd+1)]

npzfile = np.load(
    os.path.join(conf.precomp_desc_path, 'sp_labels_tsp_contours.npz'),
    fix_imports=True,
    encoding='bytes')
labelContourMask = npzfile['labelContourMask']
print("Loading seen descriptors")
seen_feats_df = pd.read_pickle(
    os.path.join(conf.precomp_desc_path, 'seen_feats_df.p')).T
print("Loading superpixels descriptors")
sp_desc_df = pd.read_pickle(os.path.join(conf.precomp_desc_path, 'sp_desc_df.p'))
print("Loading link data")
sp_link_df = pd.read_pickle(os.path.join(conf.precomp_desc_path, 'sp_link_df.p'))
print("Loading seen-to-sp histogram intersections")
sp_entr_df = pd.read_pickle(os.path.join(conf.precomp_desc_path, 'sp_entr_df.p'))
print("Loading POM")
sp_pm_fg_df = pd.read_pickle(os.path.join(conf.precomp_desc_path, 'sp_pm_fg_df.p'))
sp_pm_bg_df = pd.read_pickle(os.path.join(conf.precomp_desc_path, 'sp_pm_bg_df.p'))
print("Loading POM matrix")
pm_mat_fg = np.load(os.path.join(conf.precomp_desc_path, 'pm_mat_fg.npz'))['pm_mat_fg']
pm_mat_bg = np.load(os.path.join(conf.precomp_desc_path, 'pm_mat_bg.npz'))['pm_mat_bg']

#Test plots
#frameInd = np.arange(0,len(frameFileNames))
#gtFileNames = makeFrameFileNames(conf.frame_prefix,conf.seqStart,conf.seqEnd,conf.frameDigits,conf.truth_dir,
#                                 conf.root_path,conf.ds_dir,conf.frame_extension)
#gt_dir = os.path.join(conf.root_path,conf.ds_dir,conf.truth_dir)


#Extract ground-truth files
gt = np.zeros((len(conf.frameFileNames), labels.shape[0], labels.shape[1]))
for i in range(len(conf.frameFileNames)):
    base, fname = os.path.split(conf.frameFileNames[i])
    this_gt = utls.imread(os.path.join(gt_dir, fname))
    gt[i, :, :] = (this_gt[:, :, 0] > 0)


#Check prob maps
f = 10
im = utls.imread(conf.frameFileNames[f])
im = color.label2rgb(gt[f, :, :], im)
plt.subplot(131)
plt.imshow(pm_mat_fg[:, :, f])
plt.subplot(132)
plt.imshow(pm_mat_bg[:, :, f])
plt.subplot(133)
plt.imshow(im)
plt.show()


#Tracking---------------
totalCost = []
sol = []
ecat = []
v = []
g = []

costs_forward = list()
costs_backward = list()

gaze_points = np.delete(conf.myGaze_fg, (0, 1, 2, 5), axis=1)
conf.sig_r_in = 0.1
conf.sig_a_g = 0.5
conf.sig_a_t = 0.5
conf.norm_neighbor_in = .1
conf.norm_neighbor = .1
#conf.max_paths = 400
print("Making forward graph")
gfor, sourcefor, sinkfor = fn.makeFullGraphSPM(
    sp_entr_df,
    sp_pm_fg_df,
    sp_link_df,
    centroids_loc,
    gaze_points,
    conf.sig_r_in,
    conf.sig_a_t,
    conf.sig_a_g,
    conf.norm_neighbor,
    conf.norm_neighbor_in,
    mode='forward',
    labels=labels)
print("Making backward graph")
gback, sourceback, sinkback = fn.makeFullGraphSPM(
    sp_entr_df,
    sp_pm_fg_df,
    sp_link_df,
    centroids_loc,
    gaze_points,
    conf.sig_r_in,
    conf.sig_a_t,
    conf.sig_a_g,
    conf.norm_neighbor,
    conf.norm_neighbor_in,
    mode='backward',
    labels=labels)

print("Computing KSP on forward graph")
dict_ksp = dict()
gfor_ksp = gksp.GraphKSP(gfor, sourcefor, sinkfor, mode='edge')
gfor_ksp.disjointKSP(conf.max_paths, verbose=True)
dict_ksp['forward_sets'] = gfor_ksp.kspSet
dict_ksp['forward_costs'] = gfor_ksp.costs

print("Computing KSP on backward graph")
gback_ksp = gksp.GraphKSP(gback, sourceback, sinkback, mode='edge')
gback_ksp.disjointKSP(conf.max_paths, verbose=True)
dict_ksp['backward_sets'] = gback_ksp.kspSet
dict_ksp['backward_costs'] = gback_ksp.costs

sets = list()
sets.append(dict_ksp)

#Saving data
fileOut = os.path.join(conf.dataOutDir, 'results.npz')
data = dict()
data['labels'] = labels
data['labelContourMask'] = labelContourMask
data['sets'] = sets
data['myGaze_bg'] = conf.myGaze_bg
data['myGaze_fg'] = conf.myGaze_fg
data['frameFileNames'] = conf.frameFileNames
data['pm_mat_fg'] = pm_mat_fg
data['pm_mat_bg'] = pm_mat_bg
print("Saving stuff to: ", fileOut)
np.savez(fileOut, **data)
print("done")

with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
    yaml.dump(conf, outfile, default_flow_style=True)

# Loading back
fileOut = '/home/laurent.lejeune/medical-labeling/data/Dataset2/results/2017-05-11_12-18-11_exp/results.npz'
#fileOut = '/home/laurent.lejeune/server/medical-labeling/data/Dataset2/results/2017-02-06_11-59-34_exp/results.npz'
#fileOut = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/Dataset9/results/2017-02-11_17-33-52_exp/results.npz'
npz_file = np.load(fileOut)

labels = npz_file['labels']
labelContourMask = npz_file['labelContourMask']
sets = npz_file['sets']
frameFileNames = npz_file['frameFileNames']
pm_mat_fg = npz_file['pm_mat_fg']
pm_mat_bg = npz_file['pm_mat_bg']
gt_dir = os.path.split(frameFileNames[0])[0].replace("input-frames",
                                                     "ground_truth-frames")

scores = utls.get_scores_ksp(
    sets, np.array([0]), np.arange(0,len(conf.frameFileNames)), gt_dir, conf.frameFileNames, labels,Kmax=None)

print(" TPR/FPR: " + str(tpr_ksp[-3]) + "/" + str(fpr_ksp[-3]))

f = 40
im = utls.imread(frameFileNames[f])
idx_contours = np.where(labelContourMask[f, :, :])
im = color.label2rgb(gt[f, :, :], im)
im[idx_contours[0], idx_contours[1], :] = (255, 255, 255)
plt.subplot(131)
plt.imshow(scores[f, :, :])
plt.subplot(132)
plt.imshow(im)
plt.subplot(133)
plt.imshow(pm_mat_fg[:, :, f])
plt.show()

cost = []
for i in range(len(sets[0]['forward_costs'])):
    cost.append(np.sum(sets[0]['forward_costs'][i]))
cost = np.asarray(cost)
plt.plot(cost, 'bo-')
plt.savefig(os.path.join(dataOutDir, 'costs.pdf'))
plt.show()
plt.clf()

#gamma = np.logspace(1,4,4)
#gamma = np.array([5])
#gamma = np.linspace(0.8,1.2,40)
gamma = np.array([0.84])

lambda_ = 50

tpr_gc = []
fpr_gc = []

small = np.finfo(np.float).eps

if (pom_mat.shape[0] != len(frameFileNames)):
    pom_mat = pom_mat.transpose((2, 0, 1))
norm_scores = scores.astype(float) / np.max(scores)
heat_maps = np.zeros(scores.shape)
gc_maps = np.zeros(scores.shape)
gc_scores = []
#with progressbar.ProgressBar(maxval=gamma.shape[0]) as bar:
for i in range(gamma.shape[0]):
    for j in range(len(frameFileNames)):
        #for j in np.asarray([14]):
        print("(j,tau) = (" + str(j + 1) + "/" + str(len(frameFileNames)) +
              ", " + str(gamma[i]) + ")")
        min_p = 0.01
        n = 1000

        this_pom = pom_mat[j, :, :].copy()

        this_fg = 0.5 * np.ones(this_pom.shape)
        this_fg[scores[j, :, :] > 0] = 1

        this_bg = 0.4 * np.ones(this_pom.shape)
        this_bg[this_pom == 0] = 1

        this_pm_fg = np.clip(this_pom, a_max=1.0, a_min=min_p)

        this_fg_costs = -np.log(this_fg + small) / -np.log(
            small) - gamma[i] * np.log(this_pm_fg + small) / -np.log(small)
        this_bg_costs = -np.log(this_bg + small) / -np.log(
            small) - gamma[i] * np.log(1 - this_pm_fg + small) / -np.log(small)

        #Digitize
        bin_min = np.min([this_fg_costs, this_bg_costs])
        bin_max = np.max([this_fg_costs, this_bg_costs])
        bins = np.linspace(bin_min, bin_max, n)

        this_fg_costs = np.digitize(this_fg_costs, bins)
        this_bg_costs = np.digitize(this_bg_costs, bins)
        #this_fg_costs = (n*this_fg_costs).astype(np.int32)
        #this_bg_costs = (n*this_bg_costs).astype(np.int32)

        unaries = ((np.dstack([this_fg_costs,
                               this_bg_costs]).copy("C"))).astype(np.int32)

        # use the gerneral graph algorithm
        # first, we construct the grid graph
        inds = np.arange(scores[j, :, :].size).reshape(scores[j, :, :].shape)
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

        orig_image = color.rgb2gray(utls.imread(frameFileNames[j]))
        dx_im = orig_image[:, 0:-1] - orig_image[:, 1::]
        dy_im = orig_image[0:-1, :] - orig_image[1::, :]
        mean_squared = np.mean(
            np.hstack([(dx_im**2).ravel(), (dy_im**2).ravel()]))
        beta = 1. / (2 * mean_squared)
        horz_weights = np.exp(-dx_im**2 * beta)
        vert_weights = np.exp(-dy_im**2 * beta)

        horz_weights = (n * horz_weights).astype(np.int32)
        vert_weights = (n * vert_weights).astype(np.int32)

        #pairwise = -100* np.eye(2, dtype=np.int32)
        pairwise = 1 - np.eye(2, dtype=np.int32)

        weights = np.vstack([
            lambda_ * horz_weights.reshape(-1, 1),
            lambda_ * vert_weights.reshape(-1, 1)
        ])
        edges_idx = np.vstack([horz, vert]).astype(np.int32)
        edges = np.hstack([edges_idx, weights]).astype(np.int32)

        this_gc = np.logical_not(
            cut_from_graph(edges, unaries.reshape(-1, 2), pairwise).reshape(
                scores[j, :, :].shape))
        gc_maps[j, :, :] = this_gc
    conf_mat = metrics.confusion_matrix(gt.ravel(), gc_maps.ravel())
    precision = float(conf_mat[1, 1]) / float(conf_mat[1, 1] + conf_mat[0, 1])
    recall = float(conf_mat[1, 1]) / float(conf_mat[1, 1] + conf_mat[1, 0])
    f1 = float(2 * conf_mat[1, 1]) / float(
        2 * conf_mat[1, 1] + conf_mat[0, 1] + conf_mat[1, 0])
    gc_scores.append((lambda_, gamma[i], f1, precision, recall))
    print(gc_scores[-1])

ind = 14
plt.subplot(221)
plt.imshow(color.label2rgb(gt[ind, :, :], utls.imread(frameFileNames[ind])))
plt.subplot(222)
plt.imshow(gc_maps[ind, :, :])
plt.title('heat map')
plt.subplot(223)
plt.imshow(scores[ind, :, :])
plt.title('score map')
plt.subplot(224)
plt.imshow(pom_mat[ind, :, :])
plt.title('POM')
plt.show()

#Saving data
fileOut = os.path.join(dataOutDir, 'results_gc.npz')
data = dict()
data['auc_pom'] = auc_pom
data['tpr_pom'] = tpr_pom
data['fpr_pom'] = fpr_pom
data['heat_maps'] = heat_maps
data['tpr_gc_arr'] = tpr_gc_arr
data['fpr_gc_arr'] = fpr_gc_arr
data['auc_gc'] = auc_gc
data['pom_mat'] = pom_mat
print("Saving stuff to: ", fileOut)
np.savez(fileOut, **data)
print("done")

#Load back results
res_dir = '/home/laurent.lejeune/otlShare/laurent.lejeune/medical-labeling/data/Dataset11/results/2017-02-18_11-41-58_exp/'

npz_res = np.load(os.path.join(res_dir, 'results.npz'))
labelContourMask = npz_res['labelContourMask']
labels = npz_res['labels']
frameFileNames = npz_res['frameFileNames']
myGaze = npz_res['myGaze']
#scores = npz_res['scores']

npz_res = np.load(os.path.join(res_dir, 'results_gc.npz'))

print("AUC_pom= " + str(npz_res['auc_pom']))
print("AUC_gc= " + str(npz_res['auc_gc']))

heat_maps = npz_res['heat_maps']
pom_mat = npz_res['pom_mat']

gt_dir = os.path.split(frameFileNames[0])[0].replace("input-frames",
                                                     "ground_truth-frames")
gt = np.zeros((len(frameFileNames), labels.shape[0], labels.shape[1]))
for i in range(len(frameFileNames)):
    base, fname = os.path.split(frameFileNames[i])
    this_gt = my.imread(os.path.join(gt_dir, fname))
    gt[i, :, :] = (this_gt[:, :, 0] > 0)

frames = np.arange(0, len(frameFileNames))
with progressbar.ProgressBar(maxval=frames.shape[0]) as bar:
    for f in frames:
        bar.update(f)
        im = my.imread(frameFileNames[f])
        if (im.shape[2] > 3): im = im[:, :, 0:3]
        cont_gt = segmentation.find_boundaries(gt[f, :, :], mode='thick')
        idx_cont_gt = np.where(cont_gt)
        idx_cont_sp = np.where(labelContourMask[f, :, :])
        im[idx_cont_sp[0], idx_cont_sp[1], :] = (255, 255, 255)
        im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
        im = gaze.drawGazePoint(myGaze, f, im, radius=7)
        plt.subplot(131)
        plt.imshow(im)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(pom_mat[f, :, :], cmap=plt.get_cmap('viridis'))
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(gc_maps[f, :, :], cmap=plt.get_cmap('viridis'))
        plt.axis('off')
        plt.subplots_adjust(wspace=.05)
        plt.savefig(
            os.path.join(dataOutDir,
                         'res_' + os.path.split(frameFileNames[f])[1]),
            dpi=300,
            bbox_inches='tight')

#Cochlea
to_plot = np.array([1, 26, 44, 67, 93])
#to_plot = np.array([1,44,67])
nrows = 3
fig, axes = plt.subplots(nrows, to_plot.size, figsize=(2.8, 1))

for f in range(axes.shape[1]):
    im = my.imread(frameFileNames[to_plot[f]])
    if (im.shape[2] > 3): im = im[:, :, 0:3]
    cont_gt = segmentation.find_boundaries(gt[to_plot[f], :, :], mode='thick')
    idx_cont_gt = np.where(cont_gt)
    idx_cont_sp = np.where(labelContourMask[to_plot[f], :, :])
    im[idx_cont_sp[0], idx_cont_sp[1], :] = (255, 255, 255)
    im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
    im = gaze.drawGazePoint(myGaze, to_plot[f], im, radius=7)

    this_heat_map = heat_maps[to_plot[f], :, :]
    this_heat_map -= np.min(this_heat_map)
    this_heat_map /= np.max(this_heat_map)
    axes[0, f].imshow(im)
    axes[0, f].axis('off')
    axes[1, f].imshow(pom_mat[to_plot[f], :, :], cmap=plt.get_cmap('viridis'))
    axes[1, f].axis('off')
    axes[2, f].imshow(this_heat_map, cmap=plt.get_cmap('viridis'))
    axes[2, f].axis('off')

fig.subplots_adjust(wspace=0.01, hspace=0.01, top=1, bottom=0)
fig.savefig(os.path.join(dataOutDir, 'all.eps'), dpi=800, bbox_inches='tight')
