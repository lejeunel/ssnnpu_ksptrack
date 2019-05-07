from sklearn.metrics import f1_score
import glob
from pygco import cut_from_graph
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib, json
import cfg
import pandas as pd
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import superPixels as spix
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import my_utils as utls
import graph_cut as gc
import graph_data as my_graph_data
import gazeCsv as gaze

#Load back stuff from ksp experiment
#dir_in = '/home/laurent.lejeune/medical-labeling/Dataset3/results/2017-06-14_14-38-39_exp'
dir_in = '/home/laurent.lejeune/medical-labeling/Dataset2/results/2017-06-14_22-26-11_exp'
with open(os.path.join(dir_in, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)

the_gaze = conf.myGaze_fg

my_graph_data = my_graph_data.Graph_data(conf)
my_graph_data.load_labels_if_not_exist()

for iter_ in np.array([0,3,6,9]):
    print('iter: ' + str(iter_))

    files = sorted(glob.glob(os.path.join(dir_in,'pm_scores_iter*')))[iter_]
    print(files)
    npz_file = np.load(files)
    pm_scores = npz_file['pm_scores_fg']
    ksp_scores = npz_file['ksp_scores_mat'].transpose((1,2,0))

    if(not os.path.exists(os.path.join(conf.dataOutDir, 'pm_scores_bg_iter_' + str(iter_) + '.npz'))):
        ksp_scores_compl = utls.get_complement_arrays(ksp_scores)

        #Compute PM on complement

        #Get array of marked SPs of complement
        marked_bg = utls.array_to_marked(ksp_scores_compl,my_graph_data.labels)

        my_graph_data.calc_pm(marked_bg,
                                save=False,
                                marked_feats=None,
                                all_feats_df=my_graph_data.sp_desc_df,
                                in_type='arr',
                                mode='foreground',
                                feat_fields=['desc','hsv_hist'],
                                P_ratio = 0.05)

        pm_scores_bg = my_graph_data.get_pm_array(mode='foreground')

        fileOut = os.path.join(conf.dataOutDir, 'pm_scores_bg_iter_' + str(iter_) + '.npz')
        data = dict()
        data['pm_scores_bg'] = pm_scores_bg
        np.savez(fileOut, **data)

    else:
        npz = np.load(os.path.join(conf.dataOutDir, 'pm_scores_bg_iter_' + str(iter_) + '.npz'))
        pm_scores_bg = npz['pm_scores_bg']


#Extract ground-truth files
gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
gtFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.frameDigits, conf.truth_dir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

gt_positives = utls.getPositives(gtFileNames)

gamma = np.linspace(8,25,3)
lambda_ = np.linspace(0.8,4,3)
#gamma = np.array([30])
#lambda_ = np.array([2])
conf_mat = []
f1 = []

max_n_frames = 50
f_ind = np.linspace(0,len(conf.frameFileNames)-1,np.min((max_n_frames,len(conf.frameFileNames)))).astype(int)

for this_iter in np.array([0,6,9]):
    print('Iteration: ' + str(this_iter))
    conf_mat.append([])
    f1.append([])
    for g in gamma:
        conf_mat[-1].append([])
        f1[-1].append([])
        for l in lambda_:
            print('gamma: ' + str(g))
            print('lambda_: ' + str(l))
            my_gc = gc.graph_cut(conf,
                                pm_scores,
                                pm_scores_bg,
                                ksp_scores,
                                my_graph_data.labels,
                                gt_positives,
                                gamma = g,
                                lambda_= l)

            my_gc.run()
            import pdb; pdb.set_trace()
            this_conf_mat,this_f1 = my_gc.get_scores()
            conf_mat[-1][-1].append(this_conf_mat)
            f1[-1][-1].append(this_f1)

            files = sorted(glob.glob(os.path.join(dir_in,'pm_scores_iter*')))[this_iter]
            npz_file = np.load(files)
            pm_scores = npz_file['pm_scores_fg']
            ksp_scores = npz_file['ksp_scores_mat'].transpose((1,2,0))
            for f in f_ind:
                plt.subplot(131)
                plt.imshow(ksp_scores[f,...])
                plt.title('KSP scores. Iter.: ' + str(this_iter))
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(pm_scores[...,f])
                plt.title('PM before Iter.: ' + str(this_iter))
                plt.axis('off')
                this_im = utls.imread(conf.frameFileNames[f])
                this_im =  gaze.drawGazePoint(the_gaze,f,this_im,radius=12)
                plt.subplot(133)
                plt.imshow(this_im)
                plt.suptitle('frame ' + str(f+1) + '/' + str(len(conf.frameFileNames)) +
                             '. (gamma, lambda): ' +
                             '(' + str(g) + ',' + str(l) + ')')

                this_fname = os.path.split(conf.frameFileNames[f])[1]
                #plt.show()
                this_subdir = 'gc_frames_iter_' + str(this_iter)
                this_full_path = os.path.join(conf.dataOutDir,this_subdir,'gc_iter_' + str(this_iter) + '_' + this_fname)
                if(not os.path.exists(os.path.join(conf.dataOutDir,this_subdir))):
                    os.mkdir(os.path.join(conf.dataOutDIr,this_subdir))
                plt.savefig(os.path.join(this_full_path),dpi=100)


fileOut = os.path.join(conf.dataOutDir, 'search_gc_iter_' + str(iter_) + '.npz')
data = dict()
data['conf_mat'] = conf_mat
data['f1'] = f1
data['gamma'] = gamma
data['lambda_'] = lambda_
np.savez(fileOut, **data)

my_gc_test = gc.graph_cut(conf,
                    pm_scores,
                    pm_scores_bg,
                    ksp_scores,
                    my_graph_data.labels,
                    gt_positives,
                    gamma = 30,
                    lambda_= 2)
my_gc_test.run()
this_conf_mat,this_f1 = my_gc_test.get_scores()
print(this_f1)

for f in range(len(conf.frameFileNames)):
    this_im = utls.imread(conf.frameFileNames[f])
    this_im = gaze.drawGazePoint(the_gaze,f,this_im,radius=12)
    plt.subplot(231)
    plt.imshow(my_gc.gc_maps[f,...]);
    plt.title('Graph-cut result')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(ksp_scores[...,f]);
    plt.title('KSP')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(this_im)
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(pm_scores[...,f])
    plt.title('PM foreground')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(pm_scores_bg[...,f])
    plt.title('PM background')
    plt.axis('off')
    plt.suptitle("(frame,gamma,lambda_) = (" + str(f + 1) + "/" + str(len(conf.frameFileNames)) +
            ", " + str(my_gc.gamma) + ',' + str(my_gc.lambda_) + ")")
    this_fname = os.path.split(conf.frameFileNames[f])[1]
    plt.savefig(os.path.join(conf.dataOutDir,'all_iter_' + str(iter_) + '_' + this_fname),dpi=200)
#plt.show()

#Saving data
fileOut = os.path.join(conf.dataOutDir, 'results_gc.npz')
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
