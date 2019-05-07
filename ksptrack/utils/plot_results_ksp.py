from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import glob
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
import dataset as ds
import selective_search as ss
import shutil as sh
import learning_dataset
from itertools import cycle
import logging


def main(conf,logger=None):

    logger = logging.getLogger('plot_results_ksp')

    logger.info('--------')
    logger.info('Self-learning on: ' + conf.dataOutDir)
    logger.info('--------')

    conf.pm_thr = 0.8

    if(not os.path.exists(os.path.join(conf.dataOutDir,'metrics.npz'))):

        list_ksp = np.load(os.path.join(conf.dataOutDir,'results.npz'))['list_ksp']
        gt_dir = os.path.join(conf.root_path, conf.ds_dir, conf.truth_dir)
        my_dataset = ds.Dataset(conf)
        my_dataset.load_labels_if_not_exist()
        my_dataset.load_pm_fg_from_file()
        my_dataset.load_ss_from_file()

        l_dataset = learning_dataset.LearningDataset(conf,pos_thr=0.5)
        l_dataset.make_y_array_true(l_dataset.gt)

        logger.info('[1/8] Calculating metrics on KSP+SS PM... ')
        #probas_thr = np.linspace(0,1,20)
        seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
        new_seeds = ss.thr_all_graphs(my_dataset.g_ss,seeds,conf.ss_thr)
        my_dataset.fg_marked = new_seeds
        l_dataset.set_seeds(new_seeds)
        l_dataset.make_y_array(l_dataset.seeds)
        my_dataset.calc_pm(my_dataset.fg_marked,
                           save=False,
                           marked_feats=None,
                           all_feats_df=my_dataset.sp_desc_df,
                           in_type='not csv',
                           mode='foreground',
                           feat_fields=['desc'],
                           T = conf.T,
                           bag_max_depth=conf.bag_max_depth,
                           bag_n_feats=conf.bag_n_feats)

        probas_ksp_ss_pm = my_dataset.fg_pm_df['proba'].as_matrix()
        fpr_pm_ss, tpr_pm_ss, _ = roc_curve(l_dataset.y_true[:,2],
                                            probas_ksp_ss_pm)
        pr_pm_ss, rc_pm_ss, _ = precision_recall_curve(l_dataset.y_true[:,2],
                                                        probas_ksp_ss_pm)
        probas_thr = np.unique(probas_ksp_ss_pm)
        f1_pm_ss = [f1_score(l_dataset.y_true[:,2],probas_ksp_ss_pm > p) for p in probas_thr]

        fpr_ksp = [0.]
        tpr_ksp = [0.]
        pr_ksp = [1.]
        rc_ksp = [0.]
        f1_ksp = []

        logger.info('[2/8] Calculating metrics on KSP... ')
        for i in range(len(list_ksp)):
            logger.info('iter: ' + str(i+1) + '/' + str(len(list_ksp)))

            seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[i]))
            l_dataset.set_seeds(seeds)
            l_dataset.make_y_array(l_dataset.seeds)

            fpr, tpr, _ = roc_curve(l_dataset.y_true[:,2], l_dataset.y[:,2])
            precision, recall, _ = precision_recall_curve(l_dataset.y_true[:,2], l_dataset.y[:,2])
            f1_ksp.append(f1_score(l_dataset.y_true[:,2],l_dataset.y[:,2]))
            fpr_ksp.append(fpr[1])
            tpr_ksp.append(tpr[1])
            pr_ksp.append(precision[1])
            rc_ksp.append(recall[1])

        fpr_ksp.append(1.)
        tpr_ksp.append(1.)
        pr_ksp.append(0.)
        rc_ksp.append(1.)

        fpr_ksp_ss = [0.]
        tpr_ksp_ss = [0.]
        pr_ksp_ss = [1.]
        rc_ksp_ss = [0.]
        f1_ksp_ss = []

        logger.info('[3/8] Calculating metrics on KSP+SS... ')
        for i in range(len(list_ksp)):
            logger.info('iter: ' + str(i+1) + '/' + str(len(list_ksp)))

            seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[i]))
            new_seeds = ss.thr_all_graphs(my_dataset.g_ss,seeds,conf.ss_thr)
            l_dataset.set_seeds(new_seeds)
            l_dataset.make_y_array(l_dataset.seeds)

            fpr, tpr, _ = roc_curve(l_dataset.y_true[:,2], l_dataset.y[:,2])
            precision, recall, _ = precision_recall_curve(l_dataset.y_true[:,2], l_dataset.y[:,2])
            f1_ksp_ss.append(f1_score(l_dataset.y_true[:,2],l_dataset.y[:,2]))
            fpr_ksp_ss.append(fpr[1])
            tpr_ksp_ss.append(tpr[1])
            pr_ksp_ss.append(precision[1])
            rc_ksp_ss.append(recall[1])

        fpr_ksp_ss.append(1.)
        tpr_ksp_ss.append(1.)
        pr_ksp_ss.append(0.)
        rc_ksp_ss.append(1.)

        #Will append thresholded values to old
        fpr_ksp_ss_thr = list(fpr_ksp_ss)
        tpr_ksp_ss_thr = list(tpr_ksp_ss)
        pr_ksp_ss_thr = list(pr_ksp_ss)
        rc_ksp_ss_thr = list(rc_ksp_ss)
        f1_ksp_ss_thr = list(f1_ksp_ss)
        #probas_ksp_ss_pm = my_dataset.fg_pm_df['proba'].as_matrix()

        logger.info('[4/8] Calculating metrics on KSP+SS thresholded... ')
        y_ksp_ss_thr = probas_ksp_ss_pm > conf.pm_thr


        fpr, tpr, _ = roc_curve(l_dataset.y_true[:,2], y_ksp_ss_thr.astype(float))
        precision, recall, _ = precision_recall_curve(l_dataset.y_true[:,2], y_ksp_ss_thr.astype(float))
        f1_ksp_ss_thr.append(f1_score(l_dataset.y_true[:,2],y_ksp_ss_thr.astype(float)))
        fpr_ksp_ss_thr.append(fpr[1])
        tpr_ksp_ss_thr.append(tpr[1])
        rc_ksp_ss_thr.append(recall[1])
        pr_ksp_ss_thr.append(precision[1])

        logger.info('[5/8] Calculating metrics on PM... ')
        #probas_thr = np.linspace(0,1,20)
        fpr_pm = []
        tpr_pm = []
        pr_pm = []
        rc_pm = []
        f1_pm = []
        for i in range(len(list_ksp)):
            logger.info('iter: ' + str(i+1) + '/' + str(len(list_ksp)))
            seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[i]))
            my_dataset.fg_marked = seeds
            my_dataset.calc_pm(my_dataset.fg_marked,
                                save=False,
                                marked_feats=None,
                                all_feats_df=my_dataset.sp_desc_df,
                                in_type='not csv',
                                mode='foreground',
                        bag_n_feats=conf.max_feats_ratio,
                                feat_fields=['desc'])

            probas = my_dataset.fg_pm_df['proba'].as_matrix()
            fpr, tpr, _ = roc_curve(l_dataset.y_true[:,2],
                                    probas)
            precision, recall, _ = precision_recall_curve(l_dataset.y_true[:,2],
                                                        probas)
            probas_thr = np.unique(probas)
            f1_pm_ = [f1_score(l_dataset.y_true[:,2],probas > p) for p in probas_thr]
            f1_pm.append(f1_pm_)
            fpr_pm.append(fpr)
            tpr_pm.append(tpr)
            pr_pm.append(precision)
            rc_pm.append(recall)


        logger.info('[6/8] Calculating metrics on true ground-truth... ')
        seeds_gt = l_dataset.y_true[l_dataset.y_true[:,2]==1,:]
        my_dataset.fg_marked = seeds_gt
        my_dataset.calc_pm(my_dataset.fg_marked,
                            save=False,
                            marked_feats=None,
                            all_feats_df=my_dataset.sp_desc_df,
                            in_type='not csv',
                            mode='foreground',
                        bag_n_feats=conf.max_feats_ratio,
                            feat_fields=['desc'])

        probas = my_dataset.fg_pm_df['proba'].as_matrix()
        fpr_gt, tpr_gt, _ = roc_curve(l_dataset.y_true[:,2],
                                probas)
        pr_gt, rc_gt , _ = precision_recall_curve(l_dataset.y_true[:,2],
                                                        probas)
        probas_thr = np.unique(probas)
        f1_gt = [f1_score(l_dataset.y_true[:,2],probas > p) for p in probas_thr]

        #Make PM and KSP frames on SS
        logger.info('[7/8] Making prediction maps of KSP and KSP+SS PM... ')
        seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
        ksp_scores = utls.get_scores_from_sps(seeds,my_dataset.labels)
        new_seeds = ss.thr_all_graphs(my_dataset.g_ss,seeds,conf.ss_thr)
        ksp_ss_scores = utls.get_scores_from_sps(new_seeds,my_dataset.labels)


        my_dataset.fg_marked = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
        my_dataset.calc_pm(my_dataset.fg_marked,
                            save=False,
                            marked_feats=None,
                            all_feats_df=my_dataset.sp_desc_df,
                            in_type='not csv',
                            mode='foreground',
                        bag_n_feats=conf.max_feats_ratio,
                            feat_fields=['desc'])
        pm_ksp = my_dataset.get_pm_array(mode='foreground')
        my_dataset.fg_marked = new_seeds
        my_dataset.calc_pm(my_dataset.fg_marked,
                            save=False,
                            marked_feats=None,
                            all_feats_df=my_dataset.sp_desc_df,
                            in_type='not csv',
                            mode='foreground',
                        bag_n_feats=conf.max_feats_ratio,
                            feat_fields=['desc'])
        pm_ksp_ss = my_dataset.get_pm_array(mode='foreground')

        #Make PM and KSP frames on SS
        f1_pm_thr = []
        fpr_pm_thr = []
        tpr_pm_thr = []
        rc_pm_thr = []
        pr_pm_thr = []
        logger.info('[8/8] Making prediction maps and metrics of KSP+SS PM thresholded... ')
        new_seeds_thr_frames = my_dataset.fg_pm_df.loc[y_ksp_ss_thr,'frame'].as_matrix()
        new_seeds_thr_labels = my_dataset.fg_pm_df.loc[y_ksp_ss_thr,'sp_label'].as_matrix()
        new_seeds_thr = np.concatenate((new_seeds_thr_frames.reshape(-1,1),new_seeds_thr_labels.reshape(-1,1)),axis=1)
        ksp_ss_thr_scores = utls.get_scores_from_sps(new_seeds_thr,my_dataset.labels)

        my_dataset.fg_marked = new_seeds_thr
        my_dataset.calc_pm(my_dataset.fg_marked,
                            save=False,
                            marked_feats=None,
                            all_feats_df=my_dataset.sp_desc_df,
                            in_type='not csv',
                            mode='foreground',
                        bag_n_feats=conf.max_feats_ratio,
                            feat_fields=['desc'])
        pm_ksp_ss_thr = my_dataset.get_pm_array(mode='foreground')

        probas = my_dataset.fg_pm_df['proba'].as_matrix()
        fpr, tpr, _ = roc_curve(l_dataset.y_true[:,2],
                                probas)
        precision, recall, _ = precision_recall_curve(l_dataset.y_true[:,2],
                                                    probas)
        probas_thr = np.unique(probas)
        f1_pm_thr_ = [f1_score(l_dataset.y_true[:,2],probas > p) for p in probas_thr]
        f1_pm_thr.append(f1_pm_thr_)
        fpr_pm_thr.append(fpr)
        tpr_pm_thr.append(tpr)
        pr_pm_thr.append(precision)
        rc_pm_thr.append(recall)

        ##Saving metrics
        data = dict()
        data['probas_thr'] = probas_thr
        data['fpr_pm'] = fpr_pm
        data['tpr_pm'] = tpr_pm
        data['pr_pm'] = pr_pm
        data['rc_pm'] = rc_pm
        data['f1_pm'] = f1_pm

        data['fpr_pm_thr'] = fpr_pm_thr
        data['tpr_pm_thr'] = tpr_pm_thr
        data['pr_pm_thr'] = pr_pm_thr
        data['rc_pm_thr'] = rc_pm_thr
        data['f1_pm_thr'] = f1_pm_thr

        data['fpr_ksp'] = fpr_ksp
        data['tpr_ksp'] = tpr_ksp
        data['pr_ksp'] = pr_ksp
        data['rc_ksp'] = rc_ksp
        data['f1_ksp'] = f1_ksp

        data['fpr_ksp_ss'] = fpr_ksp_ss
        data['tpr_ksp_ss'] = tpr_ksp_ss
        data['pr_ksp_ss'] = pr_ksp_ss
        data['rc_ksp_ss'] = rc_ksp_ss
        data['f1_ksp_ss'] = f1_ksp_ss

        data['fpr_ksp_ss_thr'] = fpr_ksp_ss_thr
        data['tpr_ksp_ss_thr'] = tpr_ksp_ss_thr
        data['pr_ksp_ss_thr'] = pr_ksp_ss_thr
        data['rc_ksp_ss_thr'] = rc_ksp_ss_thr
        data['f1_ksp_ss_thr'] = f1_ksp_ss_thr

        data['fpr_pm_ss'] = fpr_pm_ss
        data['tpr_pm_ss'] = tpr_pm_ss
        data['pr_pm_ss'] = pr_pm_ss
        data['rc_pm_ss'] = rc_pm_ss
        data['f1_pm_ss'] = f1_pm_ss

        data['fpr_gt'] = fpr_gt
        data['tpr_gt'] = tpr_gt
        data['pr_gt'] = pr_gt
        data['rc_gt'] = rc_gt
        data['f1_gt'] = f1_gt

        #ksp_ss_thr_scores
        data['seeds'] = seeds
        data['ksp_scores'] = ksp_scores
        data['new_seeds'] = seeds_gt
        data['ksp_ss_scores'] = ksp_ss_scores
        data['ksp_ss_thr_scores'] = ksp_ss_thr_scores
        data['pm_ksp'] = pm_ksp
        data['pm_ksp_ss'] = pm_ksp_ss
        data['pm_ksp_ss_thr'] = pm_ksp_ss_thr
        np.savez(os.path.join(conf.dataOutDir,'metrics.npz'),**data)
    else:
        logger.info('Loading metrics.npz...')
        metrics = np.load(os.path.join(conf.dataOutDir,'metrics.npz'))
        probas_thr = metrics['probas_thr']
        fpr_pm = metrics['fpr_pm']
        tpr_pm = metrics['tpr_pm']
        pr_pm = metrics['pr_pm']
        rc_pm = metrics['rc_pm']
        f1_pm = metrics['f1_pm']

        fpr_pm_thr = metrics['fpr_pm_thr']
        tpr_pm_thr = metrics['tpr_pm_thr']
        pr_pm_thr = metrics['pr_pm_thr']
        rc_pm_thr = metrics['rc_pm_thr']
        f1_pm_thr = metrics['f1_pm_thr']

        fpr_ksp = metrics['fpr_ksp']
        tpr_ksp = metrics['tpr_ksp']
        pr_ksp = metrics['pr_ksp']
        rc_ksp = metrics['rc_ksp']
        f1_ksp = metrics['f1_ksp']

        fpr_ksp_ss = metrics['fpr_ksp_ss']
        tpr_ksp_ss = metrics['tpr_ksp_ss']
        pr_ksp_ss = metrics['pr_ksp_ss']
        rc_ksp_ss = metrics['rc_ksp_ss']
        f1_ksp_ss = metrics['f1_ksp_ss']

        fpr_ksp_ss_thr = metrics['fpr_ksp_ss_thr']
        tpr_ksp_ss_thr = metrics['tpr_ksp_ss_thr']
        pr_ksp_ss_thr = metrics['pr_ksp_ss_thr']
        rc_ksp_ss_thr = metrics['rc_ksp_ss_thr']
        f1_ksp_ss_thr = metrics['f1_ksp_ss_thr']

        fpr_pm_ss = metrics['fpr_pm_ss']
        tpr_pm_ss = metrics['tpr_pm_ss']
        pr_pm_ss = metrics['pr_pm_ss']
        rc_pm_ss = metrics['rc_pm_ss']
        f1_pm_ss = metrics['f1_pm_ss']

        fpr_gt = metrics['fpr_gt']
        tpr_gt = metrics['tpr_gt']
        pr_gt = metrics['pr_gt']
        rc_gt = metrics['rc_gt']
        f1_gt = metrics['f1_gt']

        seeds = metrics['seeds']
        ksp_scores = metrics['ksp_scores']
        seeds_gt = metrics['new_seeds']
        ksp_ss_scores = metrics['ksp_ss_scores']
        ksp_ss_thr_scores = metrics['ksp_ss_thr_scores']
        pm_ksp = metrics['pm_ksp']
        pm_ksp_ss = metrics['pm_ksp_ss']
        pm_ksp_ss_thr = metrics['pm_ksp_ss_thr']

        my_dataset = ds.Dataset(conf)
        my_dataset.load_labels_if_not_exist()
        my_dataset.load_pm_fg_from_file()
        my_dataset.load_ss_from_file()
        l_dataset = learning_dataset.LearningDataset(conf,pos_thr=0.5)
        list_ksp = np.load(os.path.join(conf.dataOutDir,'results.npz'))['list_ksp']

    #Plot all iterations of PM
    plt.clf()
    conf.roc_xlim = [0,0.4]
    conf.pr_rc_xlim = [0.6,1.]

    colors = cycle(['brown', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','slateblue','lightpink','darkmagenta'])
    lw = 1
    #PM curves
    for i, color in zip(range(len(tpr_pm)), colors):
        auc_ = auc(fpr_pm[i], tpr_pm[i])
        max_f1 = np.max(f1_pm[i])

        plt.subplot(121)
        plt.plot(fpr_pm[i], tpr_pm[i],'-', lw=lw, color=color,
                 label='KSP/PM iter. %d (area = %0.4f, max(F1) = %0.4f)' % (i+1, auc_,max_f1))

        auc_ = auc(rc_pm[i], pr_pm[i])
        plt.subplot(122)
        plt.plot(rc_pm[i], pr_pm[i],'-', lw=lw, color=color,
                 label='KSP/PM iter. %d (area = %0.4f, max(F1) = %0.4f)' % (i+1, auc_,max_f1))

    #Plot true groundtruth
    #auc_ = auc(fpr_gt, tpr_gt)
    #max_f1 = np.max(f1_gt)
    #plt.subplot(121)
    #plt.plot(fpr_gt, tpr_gt,'r-', lw=lw,
    #            label='GT (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))
    #plt.subplot(122)
    #auc_ = auc(rc_gt, pr_gt)
    #plt.plot(rc_gt, pr_gt,'r-', lw=lw,
    #            label='GT (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))

    #Plot KSP
    auc_ = auc(fpr_ksp, tpr_ksp,reorder=True)
    max_f1 = np.max(f1_ksp)
    plt.subplot(121)
    plt.plot(fpr_ksp, tpr_ksp,'go--', lw=lw,
                label='KSP (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))
    plt.subplot(122)
    auc_ = auc(rc_ksp, pr_ksp,reorder=True)
    plt.plot(rc_ksp, pr_ksp,'go--', lw=lw,
                label='KSP (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))

    #Plot KSP+SS
    auc_ = auc(fpr_ksp_ss, tpr_ksp_ss,reorder=True)
    max_f1 = np.max(f1_ksp_ss)
    plt.subplot(121)
    plt.plot(fpr_ksp_ss, tpr_ksp_ss,'ro--', lw=lw,
                label='KSP+SS (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))
    plt.subplot(122)
    auc_ = auc(rc_ksp_ss, pr_ksp_ss,reorder=True)
    plt.plot(rc_ksp_ss, pr_ksp_ss,'ro--', lw=lw,
                label='KSP+SS (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))

    #Plot KSP+SS thresholded
    auc_ = auc(fpr_ksp_ss_thr, tpr_ksp_ss_thr,reorder=True)
    max_f1 = np.max(f1_ksp_ss_thr)
    plt.subplot(121)
    plt.plot(np.sort(fpr_ksp_ss_thr), np.sort(tpr_ksp_ss_thr),'ko--', lw=lw,
                label='KSP+SS (thr = %0.2f) (area = %0.4f, max(F1) = %0.4f)' % (conf.pm_thr,auc_,max_f1))
    plt.subplot(122)
    auc_ = auc(np.asarray(rc_ksp_ss_thr).ravel(), np.asarray(pr_ksp_ss_thr).ravel(),reorder=True)
    plt.plot(np.sort(rc_ksp_ss_thr)[::-1], np.sort(pr_ksp_ss_thr[::-1]),'ko--', lw=lw,
                label='KSP+SS (thr = %0.2f) (area = %0.4f, max(F1) = %0.4f)' % (conf.pm_thr,auc_,max_f1))

    #Plot KSP+SS PM
    auc_ = auc(fpr_pm_ss, tpr_pm_ss)
    max_f1 = np.max(f1_pm_ss)
    plt.subplot(121)
    plt.plot(np.asarray(fpr_pm_ss).ravel(), np.asarray(tpr_pm_ss).ravel(),'m-', lw=lw,
                label='KSP+SS/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))
    plt.subplot(122)
    auc_ = auc(rc_pm_ss, pr_pm_ss)
    plt.plot(rc_pm_ss, pr_pm_ss,'m-', lw=lw,
                label='KSP+SS/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_,max_f1))

    #Plot KSP+SS PM thresholded
    auc_ = auc(np.asarray(fpr_pm_thr).ravel(), np.asarray(tpr_pm_thr).ravel())
    max_f1 = np.max(f1_pm_thr)
    plt.subplot(121)
    plt.plot(np.asarray(fpr_pm_thr).ravel(), np.asarray(tpr_pm_thr).ravel(),'c-', lw=lw,
                label='KSP+SS/PM (thr = %0.2f)/PM (area = %0.4f, max(F1) = %0.4f)' % (conf.pm_thr,auc_,max_f1))
    plt.subplot(122)
    auc_ = auc(np.asarray(rc_pm_thr).ravel(), np.asarray(pr_pm_thr).ravel())
    plt.plot(np.asarray(rc_pm_thr).ravel(), np.asarray(pr_pm_thr).ravel(),'c-', lw=lw,
                label='KSP+SS/PM (thr = %0.2f) (area = %0.4f, max(F1) = %0.4f)' % (conf.pm_thr,auc_,max_f1))

    plt.subplot(121)
    plt.legend()
    plt.xlim(conf.roc_xlim)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.subplot(122)
    plt.legend()
    plt.xlim(conf.pr_rc_xlim)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.suptitle(conf.seq_type + ', ' + conf.ds_dir + '\n' + 'T: ' + str(conf.T))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(conf.dataOutDir,'metrics.eps'),dpi=200)


    ###Make plots
    logger.info('Saving KSP, PM and SS merged frames...')
    gt = l_dataset.gt
    frame_dir = 'ksp_pm_frames'
    frame_path = os.path.join(conf.dataOutDir,frame_dir)
    if(os.path.exists(frame_path)):
        logger.info('[!!!] Frame dir: ' + frame_path + ' exists. Delete to rerun.')
    else:
        n_iters_ksp = len(list_ksp)
        os.mkdir(frame_path)
        with progressbar.ProgressBar(maxval=len(conf.frameFileNames)) as bar:
            for f in range(len(conf.frameFileNames)):
                cont_gt = segmentation.find_boundaries(gt[...,f],mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(conf.frameFileNames[f])
                im[idx_cont_gt[0],idx_cont_gt[1],:] = (255,0,0)
                im = gaze.drawGazePoint(conf.myGaze_fg,f,im,radius=7)

                bar.update(f)
                plt.subplot(231)
                plt.imshow(ksp_scores[...,f]); plt.title('KSP')
                plt.subplot(232)
                plt.imshow(pm_ksp[...,f]); plt.title('KSP -> PM')
                plt.subplot(233)
                plt.imshow(ksp_ss_scores[...,f]); plt.title('KSP+SS')
                plt.subplot(234)
                plt.imshow(ksp_ss_thr_scores[...,f]); plt.title('KSP+SS -> PM -> (thr = %0.2f)'%(conf.pm_thr))
                plt.subplot(235)
                plt.imshow(pm_ksp_ss_thr[...,f]); plt.title('KSP+SS -> PM -> (thr = %0.2f) -> PM'%(conf.pm_thr))
                plt.subplot(236)
                plt.imshow(im); plt.title('image')
                plt.suptitle('frame: ' + str(f) + ', n_iters_ksp: ' + str(n_iters_ksp))
                plt.savefig(os.path.join(frame_path,'f_' + str(f) + '.png'),
                            dpi=200)


    logger.info('Saving SPs per iterations plot...')
    n_sps = []
    for i in range(len(list_ksp)):
        n = np.asarray(utls.get_node_list_tracklets(list_ksp[i])).shape[0]
        n_sps.append((i+1,n))

    seeds = np.asarray(utls.get_node_list_tracklets(list_ksp[-1]))
    n = ss.thr_all_graphs(my_dataset.g_ss,seeds,conf.ss_thr).shape[0]
    n_sps.append((len(list_ksp)+1,n))
    n_sps = np.asarray(n_sps)

    plt.clf()
    plt.plot(n_sps[:,0],n_sps[:,1],'bo-')
    plt.plot(n_sps[-1,0],n_sps[-1,1],'ro')
    plt.xlabel('iterations')
    plt.ylabel('num. of superpixels')
    plt.title('num of superpixels vs. iterations. SS threshold: ' + str(conf.ss_thr))
    plt.savefig(os.path.join(conf.dataOutDir,'sps_iters.eps'),
                dpi=200)


if __name__ == "__main__":
    main(sys.argv)
