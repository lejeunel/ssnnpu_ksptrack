from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import glob
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
from ksptrack.cfgs import cfg
from ksptrack.utils import learning_dataset
from ksptrack.utils import superpixel_utils as spix
from ksptrack import sp_manager as spm
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
from ksptrack.utils.lfda import myLFDA
import pandas as pd
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import ndimage
from skimage import (color, io, segmentation, draw)
import shutil as sh
import logging
from skimage import filters
import scipy as sp


data = dict()

arg_cfg = dict()
arg_cfg["ds_dir"] = "Dataset00"  # This is a test dataset
arg_cfg["csvFileName_fg"] = "video1.csv"
arg_cfg["feats_graph"] = "unet_gaze"

#Update config
cfg_dict = cfg.cfg()
arg_cfg['seq_type'] = cfg.datasetdir_to_type(arg_cfg['ds_dir'])
cfg_dict.update(arg_cfg)
conf = cfg.dict_to_munch(cfg_dict)

#Write config to result dir
conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir,
                                        conf.resultDir, conf.out_dir_prefix,
                                        conf.testing)

#Set logger
utls.setup_logging(conf.dataOutDir)

logger = logging.getLogger('iterative_ksp')

logger.info('---------------------------')
logger.info('starting experiment on: ' + conf.ds_dir)
logger.info('type of sequence: ' + conf.seq_type)
logger.info('gaze filename: ' + conf.csvFileName_fg)
logger.info('features type: ' + conf.feat_extr_algorithm)
logger.info('Result dir:')
logger.info(conf.dataOutDir)
logger.info('---------------------------')

#Make frame file names

conf.frameFileNames = utls.get_images(os.path.join(conf.root_path,
                                                   conf.ds_dir,
                                                   conf.frameDir))

conf.myGaze_fg = utls.readCsv(
    os.path.join(conf.root_path,
                 conf.ds_dir,
                 conf.locs_dir,
                 conf.csvFileName_fg))

if (conf.labelMatPath != ''):
    conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                     conf.frameDir, conf.labelMatPath)

conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                      conf.feats_dir)

# ---------- Descriptors/superpixel costs
my_dataset = DataManager(conf)
if (conf.calc_superpix): my_dataset.calc_superpix(save=True)

my_dataset.load_superpix_from_file()
#my_dataset.relabel(save=True,who=conf.relabel_who)

from scipy.spatial.distance import mahalanobis
from metric_learn import LFDA
from sklearn.decomposition import PCA

#Calculate covariance matrix
descs = my_dataset.sp_desc_df

descs_cat = np.vstack(descs['desc'].values)
labels = my_dataset.labels
my_dataset.load_all_from_file()
pm = my_dataset.fg_pm_df

my_thresh = 0.8
lfda_n_samps = 1000

frame_1 = 62

label_2 = 268

gaze_1 = conf.myGaze_fg[conf.myGaze_fg[:, 0] == frame_1, 3:5]
g1_i, g1_j = utls.norm_to_pix(
    gaze_1[0, 0],
    gaze_1[0, 1],
    labels[..., 0].shape[1],
    labels[..., 0].shape[0])
label_1 = labels[g1_i, g1_j, frame_1]

frame_2 = 63

pm_scores_fg = my_dataset.get_pm_array(mode='foreground', frames=[frame_2])

df = descs.loc[descs['frame'] == frame_2]
feat_1 = descs.loc[(descs['frame'] == frame_1) &
                    (descs['sp_label'] == label_1), 'desc'].as_matrix()[0]


# feat_2 = descs.loc[(descs['frame'] == frame_2) &
#                     (descs['sp_label'] == label_2), 'desc'].as_matrix()[0]

descs_from = [feat_1] * df.shape[0]
df.loc[:, 'descs_from'] = pd.Series(descs_from, index=df.index)
descs_2 = utls.concat_arr(df['desc'].as_matrix())

pm = my_dataset.fg_pm_df
y = pm['proba'].values

lfda = myLFDA(num_dims=5, k=7, embedding_type='orthonormalized')

n_comps_pca = 5

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA
# lda = LinearDiscriminantAnalysis(n_components=25)
pca = PCA(n_components=n_comps_pca, whiten=True)

# descs_cat = utls.concat_arr(descs['desc'])
# rand_idx_pos = np.random.choice(np.where(y > 0)[0], size=lfda_n_samps)
# rand_idx_neg = np.random.choice(np.where(y == 0)[0], size=lfda_n_samps)
# descs_cat = utls.concat_arr(descs['desc'])
# rand_descs_pos = descs_cat[rand_idx_pos, :]
# rand_descs_neg = descs_cat[rand_idx_neg, :]
# rand_y_pos = y[rand_idx_pos]
# rand_y_neg = y[rand_idx_neg]
# rand_descs = np.concatenate((rand_descs_pos, rand_descs_neg), axis=0)
# rand_y = np.concatenate((rand_y_pos, rand_y_neg), axis=0)

# lfda.fit(descs_cat, y, my_thresh, lfda_n_samps)
#lda.fit(rand_descs, rand_y)
pca.fit(descs_cat)

# f1 = lfda.transform(feat_1.reshape(1, -1))
# f2 = lfda.transform(descs_2)
#f1 = lda.transform(feat_1.reshape(1,-1))
#f2 = lda.transform(descs_2)
f1 = pca.transform(feat_1.reshape(1, -1))
f2 = pca.transform(descs_2)
diff_norm = np.linalg.norm(f2 - np.tile(f1, (f2.shape[0], 1)), axis=1)

dists = np.zeros(labels[..., frame_2].shape)
for i, l in enumerate(np.unique(labels[..., frame_2])):
    dists[labels[..., frame_2] == l] = np.exp(-diff_norm[i]**2)
    #dists[labels[...,frame_2]==l] = -diff_norm[l]

im1 = utls.imread(conf.frameFileNames[frame_1])
label_cont = segmentation.find_boundaries(
    labels[..., frame_1], mode='thick')
aimed_cont = segmentation.find_boundaries(
    labels[..., frame_1] == label_1, mode='thick')

label_cont_im = np.zeros(im1.shape, dtype=np.uint8)
label_cont_i, label_cont_j = np.where(label_cont)
label_cont_im[label_cont_i, label_cont_j, :] = 255

io.imsave('conts.png', label_cont_im)

rr, cc = draw.circle_perimeter(g1_i, g1_j,
                                int(conf.norm_neighbor_in * im1.shape[1]))

im1[rr, cc, 0] = 0
im1[rr, cc, 1] = 255
im1[rr, cc, 2] = 0

im1[aimed_cont, :] = (255, 0, 0)

entr_labels_1 = []
centroids = spix.getLabelCentroids(labels[..., frame_1][..., np.newaxis])
#for l in np.unique(labels[...,frame_1]):
#    centroid =

im1 = csv.draw2DPoint(conf.myGaze_fg, frame_1, im1, radius=7)

im2 = utls.imread(conf.frameFileNames[frame_2])
label_cont = segmentation.find_boundaries(
    labels[..., frame_1], mode='thick')
im2[label_cont, :] = (255, 255, 255)

#plt.imshow(im1); plt.show()
plt.subplot(321)
plt.imshow(im1)
plt.title('frame_1. ind: ' + str(frame_1))
plt.subplot(322)
plt.imshow(im2)
plt.title('frame_2. ind: ' + str(frame_2))
plt.subplot(323)
plt.imshow(labels[..., frame_1])
plt.title('labels frame_1')
plt.subplot(324)
plt.imshow(labels[..., frame_2])
plt.title('labels frame_2')
plt.subplot(325)
plt.imshow(dists)
plt.title('proba trans')
plt.subplot(326)
plt.imshow(pm_scores_fg[...,frame_2] > my_thresh)
# plt.imshow(pm_scores_fg[..., frame_2] )
plt.title('f2. pm > thresh (' + str(my_thresh) + ')')
plt.show()

# plt.subplot(131)
# plt.stem(feat_1)
# plt.subplot(132)
# plt.stem(feat_2)
# plt.subplot(133)
# plt.stem(feat_1 - feat_2)
# plt.show()
