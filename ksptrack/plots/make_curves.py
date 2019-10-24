from sklearn.metrics import (precision_recall_curve, roc_curve, auc)
from skimage import (color, segmentation)
import os
import datetime
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from labeling.utils import my_utils as utls
from labeling.utils import learning_dataset
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd
from labeling.cfgs import cfg
import pickle
import glob
"""
Calculate (F1, PR, RC) in all results dirs and write score csv file
"""

def downsample(s, n_samps):

    if (s.size == 3):  # binary case
        return s[1]

    if (s.size > n_samps):
        ratio = s.size // n_samps
        return s[0::ratio]

    return s

def make_res_dict(y, y_true, method_name=None, bag_max_samples=2000, store_y=True):

    out = dict()
    pr, rc, _ = precision_recall_curve(y_true.ravel(), y.ravel())
    pr = downsample(pr, bag_max_samples)
    rc = downsample(rc, bag_max_samples)
    tpr, fpr, _ = roc_curve(y_true.ravel(), y.ravel())
    tpr = downsample(tpr, bag_max_samples)
    fpr = downsample(fpr, bag_max_samples)
    if ((tpr.size == 1) and (fpr.size == 1)):  # binary case
        auc_ = None
    else:
        auc_ = auc(tpr, fpr)
    out['pr'] = pr
    out['rc'] = rc
    num = pr * rc
    denum = pr + rc
    f1 = np.nan_to_num(2 * (num) / (denum))
    max_f1 = np.max(f1)
    max_ind_f1 = np.argmax(f1)
    out['max_ind_f1'] = max_ind_f1
    out['max_f1'] = max_f1
    out['tpr'] = tpr
    out['fpr'] = fpr
    out['auc'] = auc_
    if (store_y):
        out['preds'] = y
        out['gt'] = gt
    out['method_name'] = method_name

    return out

def pop_keys(a, keys):

    if (isinstance(a, dict)):
        for k in keys:
            del a[k]
            #a.pop(k, None)
    elif (isinstance(a, list)):
        for l in a:
            for k in keys:
                del l[k]
                #l.pop(k, None)

    return a


n_decs = 2
n_sets_per_type = 1

save_root_dir = os.path.join(rd.root_dir, 'plots_results', 'curves')
if (not os.path.exists(save_root_dir)):
    os.mkdir(save_root_dir)

for key in rd.types:

    file_out = os.path.join(save_root_dir, key + '_wtp.p')
    if (not os.path.exists(file_out)):
        wtp_list = list()
        for i in range(4):

            wtp_dict = dict()

            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_wtp[key][i])
            file_ = os.path.join(dir_, 'preds.npz')

            # Get config
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))
            conf.root_path = rd.root_dir
            conf.dataOutRoot = rd.root_dir
            conf.precomp_desc_path = os.path.join(conf.dataOutRoot,
                                                  'precomp_desc_path')

            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_)
            np_ksp = np.load(file_)

            print('WTP')
            preds = np.load(file_)['preds']
            wtp_dict = make_res_dict(preds, gt, method_name='DL-prior')
            wtp_list.append(wtp_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in wtp_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in wtp_list])
        wtp_all_dict = make_res_dict(
            preds, gts, method_name='DL-prior', store_y=False)
        wtp_list = pop_keys(wtp_list, ['preds', 'gt'])

        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': wtp_all_dict,
            'list': wtp_list
        }, open(file_out, 'wb'))

        del wtp_all_dict, wtp_list

    file_out = os.path.join(save_root_dir, key + '_ksp.p')
    if (not os.path.exists(file_out)):
        # Get first gaze-set of every dataset
        ksp_list = list()
        for i in range(4):

            ksp_dict = dict()

            #My model
            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp[key][i][0])

            file_ksp = os.path.join(dir_, 'metrics.npz')

            # Get config
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))

            conf.root_path = rd.root_dir
            conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_ksp)
            np_ksp = np.load(file_ksp)

            print('KSP')
            ksp_scores = np_ksp['ksp_scores']
            ksp_dict = make_res_dict(ksp_scores, gt, method_name='KSPTrack')

            ksp_list.append(ksp_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in ksp_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in ksp_list])
        ksp_all_dict = make_res_dict(
            preds, gts, method_name='KSPTrack', store_y=False)

        ksp_list = pop_keys(ksp_list, ['preds', 'gt'])

        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': ksp_all_dict,
            'list': ksp_list
        }, open(file_out, 'wb'))

        del ksp_all_dict, ksp_list

    file_out = os.path.join(save_root_dir, key + '_pm.p')
    if (not os.path.exists(file_out)):
        # Get first gaze-set of every dataset
        pm_list = list()
        for i in range(4):

            pm_dict = dict()

            #My model
            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_ksp[key][i][0])

            file_ksp = os.path.join(dir_, 'metrics.npz')

            # Get config
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))

            conf.root_path = rd.root_dir
            conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_ksp)
            np_ksp = np.load(file_ksp)

            print('PM')
            pm_scores = np_ksp['pm_ksp']
            pm_dict = make_res_dict(
                pm_scores, gt, method_name='KSPTrack^{opt}')

            pm_list.append(pm_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in pm_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in pm_list])
        pm_all_dict = make_res_dict(
            preds, gts, method_name='KSPTrack^{opt}', store_y=False)

        pm_list = pop_keys(pm_list, ['preds', 'gt'])

        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': pm_all_dict,
            'list': pm_list
        }, open(file_out, 'wb'))

        del pm_all_dict, pm_list

    file_out = os.path.join(save_root_dir, key + '_vilar.p')
    if (not os.path.exists(file_out)):
        vilar_list = list()
        for i in range(4):

            vilar_dict = dict()

            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_vilar[key][i])
            file_ = os.path.join(dir_, 'preds.npz')

            # Get config
            #conf_dir = os.path.join(rd.root_dir,
            #                    rd.res_dirs_dict_ksp[key][i][0])
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))

            conf.root_path = rd.root_dir
            conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_)
            np_ksp = np.load(file_)

            print('Vilar')
            preds = np.load(file_)['preds']
            vilar_dict = make_res_dict(preds, gt, method_name='P-SVM')
            vilar_list.append(vilar_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in vilar_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in vilar_list])
        vilar_all_dict = make_res_dict(
            preds, gts, method_name='P-SVM', store_y=False)
        vilar_list = pop_keys(vilar_list, ['preds', 'gt'])

        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': vilar_all_dict,
            'list': vilar_list
        }, open(file_out, 'wb'))

        del vilar_all_dict, vilar_list

    file_out = os.path.join(save_root_dir, key + '_mic17.p')
    if (not os.path.exists(file_out)):
        mic17_list = list()
        for i in range(4):

            mic17_dict = dict()

            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_mic17[key][i])
            file_ = os.path.join(dir_, 'preds.npz')

            # Get config
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))

            conf.root_path = rd.root_dir
            conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_)
            np_ksp = np.load(file_)

            print('MIC17')
            preds = np.load(file_)['preds']
            mic17_dict = make_res_dict(preds, gt, method_name='EEL')
            mic17_list.append(mic17_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in mic17_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in mic17_list])
        mic17_all_dict = make_res_dict(
            preds, gts, method_name='EEL', store_y=False)
        mic17_list = pop_keys(mic17_list, ['preds', 'gt'])

        # Save
        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': mic17_all_dict,
            'list': mic17_list
        }, open(file_out, 'wb'))

        del mic17_all_dict, mic17_list
#
    file_out = os.path.join(save_root_dir, key + '_g2s.p')
    if (not os.path.exists(file_out)):
        g2s_list = list()
        for i in range(4):

            g2s_dict = dict()

            dir_ = os.path.join(rd.root_dir, rd.res_dirs_dict_g2s[key][i])
            file_ = os.path.join(dir_, 'preds.npz')

            # Get config
            conf = cfg.load_and_convert(os.path.join(dir_, 'cfg.yml'))

            conf.root_path = rd.root_dir
            conf.dataOutDir = os.path.join(rd.root_dir, conf.dataOutDir)
            l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
            gt = l_dataset.gt

            print('Loading: ' + file_)
            np_ksp = np.load(file_)

            print('G2S')
            preds = np.load(file_)['preds']
            g2s_dict = make_res_dict(preds, gt, method_name='gaze2Segment')
            g2s_list.append(g2s_dict)

        # Compute scores on all sequences
        preds = np.concatenate([d_['preds'].ravel() for d_ in g2s_list])
        preds = preds.astype(np.float32)
        gts = np.concatenate([d_['gt'].ravel() for d_ in g2s_list])
        g2s_all_dict = make_res_dict(
            preds, gts, method_name='gaze2Segment', store_y=False)
        g2s_list = pop_keys(g2s_list, ['preds', 'gt'])

        # Save
        print('Saving to {}'.format(file_out))
        pickle.dump({
            'all_dict': g2s_all_dict,
            'list': g2s_list
        }, open(file_out, 'wb'))

        del g2s_all_dict, g2s_list

# Make curves
objs = list()

def method_name_to_style(method_name, style, query):

    ind = [i for i in range(len(method_name)) if (method_name[i] == query)]

    return style[ind[0]]

def reorder_rc_pr(pr, rc):

    if((isinstance(pr, np.ndarray)) and (isinstance(rc, np.ndarray))):
        ind_sorted = np.argsort(pr)
        return pr[ind_sorted], rc
    else:
        return pr, rc

def sort_my_meth_name(paths, names):
    out = list()
    for i in range(len(names)):
        ind = [j for j in range(len(names)) if (names[i] in paths[j])]
        out.append(paths[ind[0]])

    return out


range_roc = [-0.02, 0.25]
method_names = [
    'KSPTrack_opt', 'KSPTrack', 'gaze2Segment', 'P-SVM', 'DL-prior', 'EEL'
]
method_names_legend = [
    r'$\bf{KSPTrack^{opt}}$', 'KSPTrack', 'gaze2Segment', 'P-SVM', 'DL-prior', 'EEL'
]
method_short_names = ['pm', 'ksp', 'g2s', 'vilar', 'wtp', 'mic17' ]
plot_style = ['r-', 'bo', 'go', 'mo', 'c-', 'ko']

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

for t in rd.types:
    handles_roc = list()
    handles_pr_rc = list()
    files = glob.glob(os.path.join(save_root_dir, '{}_*.p'.format(t)))
    files = sort_my_meth_name(files, method_short_names)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for f in files:
        obj = pickle.load(open(f, 'rb'))
        data = [obj['all_dict']] + obj['list']
        for d, i in zip(data, range(len(data))):
            if(d['method_name'] == 'KSPTrack^{opt}'):
                d['method_name'] = 'KSPTrack_opt'

            h_ = ax[0].plot(
                d['tpr'],
                d['fpr'],
                method_name_to_style(method_names,
                                        plot_style,
                                        d['method_name']),
                #label=meth_label,
                alpha=0.2 if (i != 0) else 1)
            if(i == 0):
                handles_roc.append(h_[0])
            ax[0].set_title('ROC')
            pr_sorted, rc_sorted = reorder_rc_pr(d['pr'], d['rc'])
            #plt.subplot(122)
            h_ = ax[1].plot(
                pr_sorted,
                rc_sorted,
                method_name_to_style(method_names,
                                     plot_style,
                                     d['method_name']),
                alpha=0.2 if (i != 0) else 1)
            if(i == 0):
                handles_pr_rc.append(h_[0])

            ax[1].set_title('Precision/Recall')

    ax[0].legend(handles_roc, method_names_legend, loc='lower right')
    ax[0].grid(True)
    ax[0].set_xlim(left=range_roc[0] ,right=range_roc[1])
    #ax[0].set_xticks(np.arange(0, 0.4, 0.1))
    #ax[0].set_yticks(np.arange(0, 1, 0.1))

    ax[1].legend(handles_pr_rc, method_names_legend, loc='lower left')
    ax[1].grid(True)
    #ax[1].set_xticks(np.arange(0, 1, 0.1))
    #ax[1].set_yticks(np.arange(0, 1, 0.1))
    ax[0].set_xlabel('FPR')
    ax[0].set_ylabel('TPR')
    ax[1].set_xlabel('PR')
    ax[1].set_ylabel('RC')
    #plt.suptitle('Dataset type: {}'.format(t))
    #plt.show()
    plt.savefig(os.path.join(save_root_dir, '{}_curve.png'.format(t)),
                dpi=100,bbox_inches='tight')

        #list_ = obj['list']
