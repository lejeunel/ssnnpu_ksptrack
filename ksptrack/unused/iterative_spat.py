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
import scipy.io
from scipy import ndimage
import skimage.segmentation
from skimage import (color, io, segmentation)
import graphtracking as gtrack
import my_utils as utls
import graph_cut as gc
import dataset as ds
import selective_search as ss

extra_cfg = dict()

extra_cfg['calc_superpix'] = True
extra_cfg['calc_sp_feats'] = False
extra_cfg['calc_entrance'] = False
extra_cfg['calc_linking'] = True
extra_cfg['calc_pm'] = True
extra_cfg['calc_seen_feats'] = True
extra_cfg['calc_ss'] = False
extra_cfg['calc_desc_means'] = False
extra_cfg['n_iters_ksp'] = 10


extra_cfg['ds_dir'] = 'Dataset3'
data = dict()


cfg_dict = cfg.cfg()
cfg_dict.update(extra_cfg)
conf = cfg.Bunch(cfg_dict)

#Write config to result dir
conf.dataOutDir = utls.getDataOutDir(conf.dataOutRoot, conf.ds_dir, conf.resultDir,
                                conf.out_dir_prefix, conf.testing)
print('starting experiment on: ' + conf.ds_dir)
print('Result dir:')
print(conf.dataOutDir)

#Make frame file names
gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
gtFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.frameDigits, conf.truth_dir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

conf.frameFileNames = utls.makeFrameFileNames(
    conf.frame_prefix, conf.frameDigits, conf.frameDir,
    conf.root_path, conf.ds_dir, conf.frame_extension)

#conf.myGaze_fg = utls.readCsv(conf.csvName_fg)
conf.myGaze_fg = utls.readCsv(os.path.join(conf.root_path,conf.ds_dir,conf.locs_dir,conf.csvFileName_fg))

#conf.myGaze_bg = utls.readCsv(conf.csvName_bg)
gt_positives = utls.getPositives(gtFileNames)

if (conf.labelMatPath != ''):
    conf.labelMatPath = os.path.join(conf.dataOutRoot, conf.ds_dir, conf.frameDir,
                                conf.labelMatPath)

conf.precomp_desc_path = os.path.join(conf.dataOutRoot, conf.ds_dir,
                                conf.feats_dir)


# ---------- Descriptors/superpixel costs
my_dataset = ds.Dataset(conf)


if(conf.calc_superpix): my_dataset.calc_superpix(save=True)
if(conf.calc_sp_feats): my_dataset.calc_sp_feats(save=True)
if(conf.calc_linking): my_dataset.calc_linking(save=True)
if(conf.calc_seen_feats): my_dataset.calc_seen_feats(save=True)
if(conf.calc_entrance): my_dataset.calc_entrance(save=True)
if(conf.calc_pm): my_dataset.calc_pm(conf.myGaze_fg,
                    save=True,
                    marked_feats=None,
                    all_feats_df=my_dataset.sp_desc_df,
                    in_type='csv_normalized',
                    mode='foreground',
                    #feat_fields=['desc','hsv_hist'])
                    feat_fields=['desc'])
if(conf.calc_ss): my_dataset.calc_sel_search(save=True)
if(conf.calc_desc_means): my_dataset.calc_sp_feats_means(save=True)

my_dataset.load_all_from_file()

#sp_desc = my_dataset.get_sp_desc_from_file()

#s = np.asarray([sp_desc.ix[i]['desc'].shape for i in range(sp_desc.shape[0])])


#Tracking with KSP---------------
totalCost = []
g = []

costs_forward = list()
costs_backward = list()

gaze_points = np.delete(conf.myGaze_fg, (0, 1, 2, 5), axis=1)

list_ksp = []
pm_scores_mat = []
ksp_scores_mat = []

g_for = gtrack.GraphTracking(tol=conf.ksp_tol,mode='edge')
g_back = gtrack.GraphTracking(tol=conf.ksp_tol,mode='edge')


find_new_forward = True
find_new_backward = True
i = 0

pos_sp_for = []
pos_sp_back = []

#pm_scores_fg = my_dataset.get_pm_array(mode='foreground')


while((find_new_forward or find_new_backward) and (i<conf.n_iters_ksp)):
    dict_ksp = dict()

    print("i: " + str(i+1))

    if(i == 0):
        #Make backward graph
        #tmp_graph = np.load('tmp_graphs.npz')
        if(find_new_backward):
            #g_back = tmp_graph['g_back'][()]
            g_back.makeFullGraph(
                my_dataset.sp_desc_df,
                my_dataset.sp_desc_means,
                my_dataset.sp_entr_df,
                my_dataset.fg_pm_df,
                my_dataset.sp_link_df,
                my_dataset.centroids_loc,
                my_dataset.conf.myGaze_fg,
                my_dataset.conf.norm_neighbor,
                my_dataset.conf.norm_neighbor_in,
                my_dataset.conf.thresh_aux,
                my_dataset.conf.hoof_tau_u,
                direction='backward',
                labels=my_dataset.labels)

        #Make forward graph
        if(find_new_forward):
            g_for = tmp_graph['g_for'][()]
            g_for.makeFullGraph(
                my_dataset.sp_desc_df,
                my_dataset.sp_desc_means,
                my_dataset.sp_entr_df,
                my_dataset.fg_pm_df,
                my_dataset.sp_link_df,
                my_dataset.centroids_loc,
                my_dataset.conf.myGaze_fg,
                my_dataset.conf.norm_neighbor,
                my_dataset.conf.norm_neighbor_in,
                my_dataset.conf.thresh_aux,
                my_dataset.conf.hoof_tau_u,
                labels=my_dataset.labels)
    else:
        g_for.merge_tracklets_temporally(my_dataset.centroids_loc,
                                        my_dataset.fg_pm_df,
                                        my_dataset.sp_desc_df,
                                        my_dataset.conf.myGaze_fg,
                                        my_dataset.conf.norm_neighbor_in,
                                        my_dataset.conf.thresh_aux,
                                        my_dataset.get_labels())
        g_for.reset_paths()

        g_back.merge_tracklets_temporally(my_dataset.centroids_loc,
                                        my_dataset.fg_pm_df,
                                        my_dataset.sp_desc_df,
                                        my_dataset.conf.myGaze_fg,
                                        my_dataset.conf.norm_neighbor_in,
                                        my_dataset.conf.thresh_aux,
                                        my_dataset.get_labels())
        g_back.reset_paths()

    print("Computing KSP on backward graph. (i: " + str(i+1) + ")")
    if(i==0):
        dict_ksp['backward_sets'] = tmp_graph['g_back'][()].kspSet
        dict_ksp['backward_tracklets'] = tmp_graph['g_back'][()].tracklets
        dict_ksp['backward_costs'] = tmp_graph['g_back'][()].costs
    else:
        find_new_backward = g_back.disjointKSP(conf.max_paths, verbose=True)
        dict_ksp['backward_sets'] = g_back.kspSet
        dict_ksp['backward_tracklets'] = g_back.tracklets
        dict_ksp['backward_costs'] = g_back.costs


    print("Computing KSP on forward graph. (i: " + str(i+1) + ")")
    if(i==0):
        dict_ksp['forward_sets'] = tmp_graph['g_for'][()].kspSet
        dict_ksp['forward_tracklets'] = tmp_graph['g_for'][()].tracklets
        dict_ksp['forward_costs'] = tmp_graph['g_for'][()].costs
    else:
        find_new_forward = g_for.disjointKSP(conf.max_paths, verbose=True)
        dict_ksp['forward_sets'] = g_for.kspSet
        dict_ksp['forward_tracklets'] = g_for.tracklets
        dict_ksp['forward_costs'] = g_for.costs


    if((find_new_forward or find_new_backward)):
        list_ksp.append(dict_ksp)

        ksp_scores_mat = utls.get_scores_ksp_tracklets(
            dict_ksp, np.array([0]), np.arange(0,len(conf.frameFileNames)),
            gt_dir, conf.frameFileNames, my_dataset.labels,Kmax=None)

        #Update marked superpixels if graph is not "finished"
        if(find_new_forward):
            this_marked_for = utls.ksp2array_tracklets(dict_ksp,
                                        my_dataset.labels,arg_direction=['forward_sets'])
            my_dataset.update_marked_sp(this_marked_for,mode='foreground')


            pos_sp_for.append(this_marked_for.shape[0])

            print("Forward graph. Number of positive sps of ksp at iteration " + str(i+1) + ": " + str(this_marked_for.shape[0]))
            if(i>0):
                if(pos_sp_for[-1] == pos_sp_for[-2]):
                    find_new_forward = False

        if(find_new_backward):
            this_marked_back = utls.ksp2array_tracklets(dict_ksp,
                                        my_dataset.labels,arg_direction=['backward_sets'])
            my_dataset.update_marked_sp(this_marked_back,mode='foreground')
            pos_sp_back.append(this_marked_back.shape[0])

            print("Backward graph. Number of positive sps of ksp at iteration " + str(i+1) + ": " + str(this_marked_back.shape[0]) )
            if(i>0):
                if(pos_sp_back[-1] == pos_sp_back[-2]):
                    find_new_backward = False


        n_pix_ksp = np.sum((ksp_scores_mat > 0).ravel())
        print("Number hit pixels of ksp at iteration " + str(i+1) + ": " + str(n_pix_ksp))

        #print('Generating PM array. (i: ' + str(i+1) + ')')
        #pm_scores_fg = my_dataset.get_pm_array(mode='foreground')

        fileOut = os.path.join(conf.dataOutDir, 'pm_scores_iter_' + str(i) + '.npz')
        data = dict()
        #data['pm_scores_fg'] = pm_scores_fg
        data['ksp_scores_mat'] = ksp_scores_mat
        np.savez(fileOut, **data)

        #Recompute PM values
        if(i+1 < conf.n_iters_ksp):
            pass
            my_dataset.calc_pm(my_dataset.fg_marked,
                                save=False,
                                marked_feats=None,
                                all_feats_df=my_dataset.sp_desc_df,
                                in_type='not csv',
                                mode='foreground',
                                feat_fields=['desc'])

        i += 1


###Saving KSP-------------------------------------------------
fileOut = os.path.join(conf.dataOutDir, 'results.npz')
data = dict()
data['frameFileNames'] = conf.frameFileNames
data['n_iters_ksp'] = conf.n_iters_ksp
data['ksp_scores_mat'] = ksp_scores_mat
data['list_ksp'] = list_ksp
print("Saving stuff to: ", fileOut)
np.savez(fileOut, **data)
print("done")

with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
    yaml.dump(conf, outfile, default_flow_style=True)



import pdb; pdb.set_trace()
#print('Merging KSP hits with SS. tau:' + str(conf.ss_thr))
#my_dataset.load_ss_from_file()
#frames_idx = np.arange(0,len(conf.frameFileNames))
#merged_sps = []
#with progressbar.ProgressBar(maxval=frames_idx.shape[0]) as bar:
#    for f in frames_idx:
#        candidates = my_dataset.fg_marked[my_dataset.fg_marked[:,0] == f,1].astype(int).tolist()
#        merged_parents, merged_children, leftovers = ss.get_merge_candidates(my_dataset.g_ss[f],candidates,conf.ss_thr)

print('Calculating final PM')
pm_scores_fg = my_dataset.get_pm_array(mode='foreground')

ksp_scores = utls.get_scores_from_sps(my_dataset.fg_marked.astype(int),
                                            my_dataset.labels)

print('Saving KSP, PM and SS merged frames...')
os.mkdir(conf.dataOutDir,'ksp_pm_frames')
with progressbar.ProgressBar(maxval=frames_idx.shape[0]) as bar:
    for f in frames_idx:
        bar.update(f)
        plt.subplot(221)
        plt.imshow(lp_scores[...,f]); plt.title('LP')
        plt.subplot(222)
        plt.imshow(ksp_scores[...,f]); plt.title('KSP')
        plt.subplot(223)
        plt.imshow(ksp_scores[...,f]); plt.title('KSP')
        plt.subplot(224)
        plt.imshow(utls.imread(conf.frameFileNames[f])); plt.title('image')
        plt.suptitle('frame: ' + str(f))
        plt.savefig(os.path.join(os.path.join(conf.dataOutDir,
                                            'ksp_pm_frames'),'f_'+str(f)+'.png'),
                    dpi=200)
#plt.show()


#Tracking scale-wise-------------------------------------------------
dict_ksp = list_ksp[-1]
find_new_forward = True
find_new_backward = True

with open(os.path.join(dir_in, 'cfg.yml'), 'r') as outfile:
    conf = yaml.load(outfile)

conf.n_iter_lp_gd = 10

my_dataset = ds.Dataset(conf)
my_dataset.load_all_from_file()

#Load graph selective search
npzfile = np.load(os.path.join(conf.precomp_desc_path,'g_ss.npz'))
g_ss = npzfile['g_ss'][()]

marked_for = utls.ksp2array_tracklets(dict_ksp,
                        my_dataset.labels,arg_direction=['forward_sets'])
marked_back = utls.ksp2array_tracklets(dict_ksp,
                        my_dataset.labels,arg_direction=['backward_sets'])

my_dataset.update_marked_sp(marked_back,mode='foreground')
my_dataset.update_marked_sp(marked_for,mode='foreground')


g_for = gtrack.GraphTracking()
g_back = gtrack.GraphTracking()

g_for.makeFullGraphSPM(
    my_dataset.sp_desc_df,
    my_dataset.sp_entr_df,
    my_dataset.fg_pm_df,
    my_dataset.sp_link_df,
    my_dataset.centroids_loc,
    my_dataset.conf.myGaze_fg,
    my_dataset.conf.norm_neighbor,
    my_dataset.conf.norm_neighbor_in,
    my_dataset.conf.thresh_aux,
    my_dataset.conf.hoof_tau_u,
    direction='forward',
    labels=my_dataset.labels)


g_back.makeFullGraphSPM(
    my_dataset.sp_desc_df,
    my_dataset.sp_entr_df,
    my_dataset.fg_pm_df,
    my_dataset.sp_link_df,
    my_dataset.centroids_loc,
    my_dataset.conf.myGaze_fg,
    my_dataset.conf.norm_neighbor,
    my_dataset.conf.norm_neighbor_in,
    my_dataset.conf.thresh_aux,
    my_dataset.conf.hoof_tau_u,
    direction='backward',
    labels=my_dataset.labels)

list_lp = []

init_flow_for = len(dict_ksp['forward_sets'][-1])
init_flow_back = len(dict_ksp['backward_sets'][-1])

i = 0

while((find_new_forward or find_new_backward) and (i<conf.n_iter_lp)):

    dict_lp = dict()

    print("i: " + str(i+1))
    #Make forward graph
    if(find_new_forward):

        g_for.mark_tracklets_for_ss(my_dataset.fg_marked)
        g_for.merge_tracklets_spatially(g_ss,conf.ss_thr)

        g_for.makeFullGraphSPM(
            my_dataset.sp_desc_df,
            my_dataset.sp_entr_df,
            my_dataset.fg_pm_df,
            my_dataset.sp_link_df,
            my_dataset.centroids_loc,
            my_dataset.conf.myGaze_fg,
            my_dataset.conf.norm_neighbor,
            my_dataset.conf.norm_neighbor_in,
            my_dataset.conf.thresh_aux,
            my_dataset.conf.hoof_tau_u,
            direction='forward',
            labels=my_dataset.labels)


        print("Computing LP on forward graph. (i: " + str(i+1) + ")")
        find_new_forward = g_for.lp_min_cost_flow(init_flow=init_flow_for,gamma_0=0.15,max_iter=conf.n_iter_lp_gd)
        dict_lp['forward_lp_sps'] = utls.lp_sol_to_sps(g_for.g,
                                                       g_for.tracklets,
                                                       g_ss,
                                                       g_for.lp_sols[-1]['x'],
                                                       labels=my_dataset.labels)
        dict_lp['forward_costs'] = g_for.lp_costs

        lp_scores = utls.get_scores_from_sps(dict_lp['forward_lp_sps'],
                                                  my_dataset.labels)
        ksp_scores = utls.get_scores_from_sps(my_dataset.fg_marked.astype(int),
                                                  my_dataset.labels)

        #f = 116
        #t_id = [t.id_ for t in g_for.tracklets if(t.get_in_frame() == f)]
        #e_in = [e for e in g_for.g.edges() if((e[1][0] in t_id) and (e[0] == 's'))]
        #ids_from_s = [e[1][0] for e in e_in]
        #labelsin = [t.get_in_label() for t in g_for.tracklets if(t.id_ in ids_from_s)]
        #labels72 = np.zeros(my_dataset.labels[...,f].shape)
        ##labels72 += my_dataset.labels[...,f] == 257
        #labels72 += my_dataset.labels[...,f] == 33
        #labels72 += my_dataset.labels[...,f] == 171
        #labels72 += my_dataset.labels[...,f] == 45
        #labels72 += my_dataset.labels[...,f] == 16
        #labels72 += my_dataset.labels[...,f] == 217
        #plt.imshow(labels72); plt.show()


        #f = 80
        #for f in np.arange(0,120):
        #    plt.subplot(131)
        #    plt.imshow(lp_scores[...,f]); plt.title('LP')
        #    plt.subplot(132)
        #    plt.imshow(ksp_scores[...,f]); plt.title('KSP')
        #    plt.subplot(133)
        #    plt.imshow(utls.imread(conf.frameFileNames[f])); plt.title('image')
        #    plt.suptitle('frame: ' + str(f))
        #    plt.savefig(os.path.join(dir_in,'f_'+str(f)+'.png'),dpi=100)
        #plt.show()


    #Make backward graph
    if(find_new_backward):

        g_back.mark_tracklets_for_ss(my_dataset.fg_marked)
        g_back.merge_tracklets_spatially(g_ss,conf.ss_thr)

        g_back.makeFullGraphSPM(
            my_dataset.sp_desc_df,
            my_dataset.sp_entr_df,
            my_dataset.fg_pm_df,
            my_dataset.sp_link_df,
            my_dataset.centroids_loc,
            my_dataset.conf.myGaze_fg,
            my_dataset.conf.norm_neighbor,
            my_dataset.conf.norm_neighbor_in,
            my_dataset.conf.thresh_aux,
            my_dataset.conf.hoof_tau_u,
            direction='backward',
            labels=my_dataset.labels)

        print("Computing LP on backward graph. (i: " + str(i+1) + ")")
        find_new_backward = g_back.lp_min_cost_flow(init_flow=init_flow_back,gamma_0=0.15,max_iter=conf.n_iter_lp_gd)
        dict_lp['backward_lp_sps'] = utls.lp_sol_to_sps(g_back.g,
                                                       g_back.tracklets,
                                                       g_ss,
                                                       g_back.lp_sols[-1]['x'],
                                                       labels=my_dataset.labels)
        dict_lp['backward_costs'] = g_back.lp_costs


        my_dataset.update_marked_sp(dict_lp['backward_lp_sps'],mode='foreground')
        my_dataset.update_marked_sp(dict_lp['forward_lp_sps'],mode='foreground')

        list_lp.append(dict_lp)

        #Recompute PM values
        if(i+1 < conf.n_iter_lp):
            my_dataset.calc_pm(my_dataset.fg_marked,
                                save=False,
                                marked_feats=None,
                                all_feats_df=my_dataset.sp_desc_df,
                                in_type='not csv',
                                mode='foreground',
                                feat_fields=['desc','hsv_hist'])

        i += 1


#Saving data
if(conf.n_iters_ksp > 0):
    fileOut = os.path.join(conf.dataOutDir, 'results.npz')
    data = dict()
    data['labels'] = my_dataset.labels
    data['labelContourMask'] = my_dataset.labelContourMask
    data['myGaze_fg'] = conf.myGaze_fg
    data['frameFileNames'] = conf.frameFileNames
    data['n_iters_ksp'] = conf.n_iters_ksp
    data['ksp_scores_mat'] = ksp_scores_mat
    data['pm_scores_fg'] = pm_scores_fg
    data['list_ksp'] = list_ksp
    data['list_lp'] = list_ksp
    print("Saving stuff to: ", fileOut)
    np.savez(fileOut, **data)
    print("done")

    with open(os.path.join(conf.dataOutDir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=True)
