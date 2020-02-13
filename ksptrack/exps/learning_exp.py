from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import glob
import datetime
import os
import progressbar
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, io, segmentation)
import my_utils as utls
import gazeCsv as gaze
import learning_dataset
import selective_search
import selective_search as ss
import matplotlib.gridspec as gridspec
import logging
from sklearn.ensemble import RandomForestClassifier
"""
Computes segmentation frames, ROC curves on single sequence
for all iterations (KSP, KSP+SS, etc..)
"""


def get_pm_array(labels, descs, probas, idx=None):

    if (idx is None):
        idx = np.arange(labels.shape[-1])

    descs_aug = descs
    descs_aug['proba'] = pd.Series(probas, index=descs_aug.index)
    scores = labels.copy().astype(float)
    with progressbar.ProgressBar(maxval=scores.shape[2]) as bar:
        for i in idx:
            bar.update(i)
            this_frame_pm_df = descs_aug[descs_aug['frame'] == i]
            dict_keys = this_frame_pm_df['sp_label']
            dict_vals = this_frame_pm_df['proba']
            dict_map = dict(zip(dict_keys, dict_vals))
            for k, v in dict_map.items():
                scores[scores[..., i] == k, i] = v

    return scores


def get_newest_exp_dirs(dir_root, res_dir, dataset_dirs):
    """ Parses directories dir_root/dataset_dirs[i]/results.
    Returns last modified experiment for all i
    """

    all_exp_dirs = []
    for d in dataset_dirs:
        exp_path = os.path.join(dir_root, d, 'results')
        exp_dirs = [os.path.join(exp_path, d) for d in os.listdir(exp_path)]
        latest_exp_dir = max(exp_dirs, key=os.path.getmtime)
        all_exp_dirs.append(latest_exp_dir)

    return all_exp_dirs


def main(confs, out_dir=None):

    alpha = 0.3
    n_points = 2000
    seq_type = confs[0].seq_type

    if (out_dir is None):
        now = datetime.datetime.now()
        dateTime = now.strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(
            confs[0].dataOutRoot, 'learning_exps',
            'learning_' + confs[0].seq_type + '_' + dateTime)

    dir_in = [c.dataOutDir for c in confs]

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    datasets = []
    utls.setup_logging(out_dir)
    logger = logging.getLogger('learning_exp')

    logger.info('Starting learning experiment on:')
    logger.info(dir_in)
    logger.info('Gaze file: ' + str(confs[0].csvFileName_fg))
    logger.info('')

    if (not os.path.exists(os.path.join(out_dir, 'datasets.npz'))):
        logger.info('Building target vectors')
        for i in range(len(dir_in)):
            with open(os.path.join(dir_in[i], 'cfg.yml'), 'r') as outfile:
                conf = yaml.load(outfile)

            logger.info('Dataset: ' + str(i + 1) + '/' + str(len(dir_in)))
            files = sorted(
                glob.glob(os.path.join(dir_in[i], 'pm_scores_iter*')))[-1]

            #logger.info('Init. learner')
            dataset = learning_dataset.LearningDataset(conf)

            npz_file = np.load(os.path.join(dir_in[i], 'results.npz'))

            seeds = np.asarray(
                utls.get_node_list_tracklets(npz_file['list_ksp'][-1]))
            if (confs[0].use_ss):
                dataset.load_ss_from_file()
                seeds = ss.thr_all_graphs(dataset.g_ss, seeds, conf.ss_thr)
            dataset.set_seeds(seeds)

            dataset.make_y_array(seeds)
            dataset.make_y_array_true(dataset.gt)

            datasets.append(dataset)

        if (not os.path.exists(out_dir)):
            os.mkdir(out_dir)

        logger.info('saving datasets to: ' + out_dir)
        np.savez(os.path.join(out_dir, 'datasets.npz'),
                 **{'datasets': datasets})
    else:
        logger.info('Loading datasets...')
        datasets = np.load(os.path.join(out_dir, 'datasets.npz'))['datasets']

    n_folds = 4
    fold_ids = np.arange(0, 4)[::-1]
    res_list = []

    n_e = 150

    if (not os.path.exists(os.path.join(out_dir, 'results.npz'))):
        for i in range(n_folds):

            logger.info('-----------------')
            pred_fold = i
            train_folds = np.asarray([
                fold_ids[j] for j in range(n_folds)
                if (fold_ids[j] != pred_fold)
            ])

            logger.info('train_folds: ' + str(train_folds))
            logger.info('pred_folds: ' + str(pred_fold))
            logger.info('-----------------')

            X_train = utls.concat_arr(
                np.concatenate([
                    datasets[train_folds[j]].X
                    for j in range(train_folds.shape[0])
                ]))
            y_train_my = np.concatenate([
                datasets[train_folds[j]].y[:, 2]
                for j in range(train_folds.shape[0])
            ])
            y_train_true = np.concatenate([
                datasets[train_folds[j]].y_true[:, 2]
                for j in range(train_folds.shape[0])
            ])

            logger.info('Extracting X_test')
            X_test = utls.concat_arr(datasets[pred_fold].X)
            logger.info('Extracting y_test')
            y_test = datasets[pred_fold].y_true[:, 2]

            logger.info('Fitting...')
            bag_n_feats = confs[0].bag_n_feats_rf
            bag_max_depth = confs[0].bag_max_depth_rf
            logger.info('bag_n_feats: ' + str(bag_n_feats))
            logger.info('bag_max_depth: ' + str(bag_max_depth))
            n_trees = datasets[0].conf.T
            clf_my = RandomForestClassifier(max_features=bag_n_feats,
                                            class_weight='balanced',
                                            n_estimators=n_trees)
            clf_true = RandomForestClassifier(max_features=bag_n_feats,
                                              class_weight='balanced',
                                              n_estimators=n_trees)
            clf_my.fit(X_train, y_train_my)
            clf_true.fit(X_train, y_train_true)

            logger.info('Predicting...')
            probas_my = clf_my.predict_proba(X_test)[:, 1]
            probas_true = clf_true.predict_proba(X_test)[:, 1]

            #probas_my = rf.run(X_train,y_train_my,X_test,150)
            #probas_true = rf.run(X_train,y_train_true,X_test,150)

            logger.info('Computing ROC curves on true model')
            fpr_true, tpr_true, thresholds_true = roc_curve(
                y_test, probas_true)

            auc_true = auc(fpr_true, tpr_true)
            logger.info('auc_true: ' + str(auc_true))
            logger.info('Computing ROC curves on my model')
            fpr_my, tpr_my, thresholds_my = roc_curve(y_test,
                                                      probas_my,
                                                      pos_label=1)
            auc_my = auc(fpr_my, tpr_my)
            logger.info('auc_my: ' + str(auc_my))

            logger.info('Computing prec-recall curves on true model')
            precision_true, recall_true, _ = precision_recall_curve(
                y_test, probas_true)
            logger.info('Computing prec-recall curves on my model')
            precision_my, recall_my, _ = precision_recall_curve(
                y_test, probas_my)

            dict_ = dict()
            dict_['train_folds'] = train_folds
            dict_['pred_fold'] = pred_fold
            dict_['n_estimators'] = n_e
            dict_['fpr_true'] = fpr_true
            dict_['tpr_true'] = tpr_true
            dict_['fpr_my'] = fpr_my
            dict_['tpr_my'] = tpr_my
            dict_['auc_true'] = auc_true
            dict_['precision_true'] = precision_true
            dict_['recall_true'] = recall_true
            dict_['auc_true'] = auc_true
            dict_['auc_my'] = auc_my
            dict_['precision_my'] = precision_my
            dict_['recall_my'] = recall_my
            dict_['probas_my'] = probas_my
            dict_['probas_true'] = probas_true
            dict_['y_test'] = y_test

            res_list.append(dict_)

        file_out = os.path.join(out_dir, 'results.npz')
        logger.info('Saving metrics to ')
        np.savez(file_out, **{'res_list': res_list})
    else:
        logger.info('Loading results...')
        res_list = np.load(os.path.join(out_dir, 'results.npz'))['res_list']

    #Plot folds
    colors = ['blue', 'darkorange', 'seagreen', 'yellow', 'blue']
    lw = 1
    plt.clf()

    l_fpr_true = []
    l_tpr_true = []
    l_pr_true = []
    l_rc_true = []

    l_fpr_my = []
    l_tpr_my = []
    l_pr_my = []
    l_rc_my = []

    for i in range(len(res_list)):
        fpr_true = res_list[i]['fpr_true']
        tpr_true = res_list[i]['tpr_true']
        #fpr_true, tpr_true = utls.my_interp(fpr_true, tpr_true, n_points)
        l_fpr_true.append(fpr_true)
        l_tpr_true.append(tpr_true)

        fpr_my = res_list[i]['fpr_my']
        tpr_my = res_list[i]['tpr_my']
        #fpr_my, tpr_my = utls.my_interp(fpr_my, tpr_my, n_points)
        l_fpr_my.append(fpr_my)
        l_tpr_my.append(tpr_my)

        pr_true = res_list[i]['precision_true']
        rc_true = res_list[i]['recall_true']
        #rc_true, pr_true = utls.my_interp(rc_true, pr_true, n_points)
        l_rc_true.append(rc_true)
        l_pr_true.append(pr_true)

        pr_my = res_list[i]['precision_my']
        rc_my = res_list[i]['recall_my']
        #rc_my, pr_my = utls.my_interp(rc_my, pr_my, n_points)
        l_rc_my.append(rc_my)
        l_pr_my.append(pr_my)

    rc_range_my = [
        np.min([np.min(l_rc_my[i]) for i in range(len(l_rc_my))]),
        np.max([np.max(l_rc_my[i]) for i in range(len(l_rc_my))])
    ]

    rc_range_true = [
        np.min([np.min(l_rc_true[i]) for i in range(len(l_rc_true))]),
        np.max([np.max(l_rc_true[i]) for i in range(len(l_rc_true))])
    ]

    rc_range = [
        np.min((rc_range_my[0], rc_range_true[0])),
        np.max((rc_range_my[1], rc_range_true[1]))
    ]

    fpr_range_my = [
        np.min([np.min(l_fpr_my[i]) for i in range(len(l_fpr_my))]),
        np.max([np.max(l_fpr_my[i]) for i in range(len(l_fpr_my))])
    ]

    fpr_range_true = [
        np.min([np.min(l_fpr_true[i]) for i in range(len(l_fpr_true))]),
        np.max([np.max(l_fpr_true[i]) for i in range(len(l_fpr_true))])
    ]

    fpr_range = [
        np.min((fpr_range_my[0], fpr_range_true[0])),
        np.max((fpr_range_my[1], fpr_range_true[1]))
    ]

    l_fpr_tpr_my_interp = np.asarray([
        utls.my_interp(l_fpr_my[i], l_tpr_my[i], n_points, fpr_range)
        for i in range(len(l_fpr_my))
    ]).transpose(1, 0, 2)
    l_fpr_my = l_fpr_tpr_my_interp[0, ...]
    l_tpr_my = l_fpr_tpr_my_interp[1, ...]

    l_pr_rc_my_interp = np.asarray([
        utls.my_interp(l_rc_my[i], l_pr_my[i], n_points, rc_range)
        for i in range(len(l_rc_my))
    ]).transpose(1, 0, 2)
    l_rc_my = l_pr_rc_my_interp[0, ...]
    l_pr_my = l_pr_rc_my_interp[1, ...]

    l_fpr_tpr_true_interp = np.asarray([
        utls.my_interp(l_fpr_true[i], l_tpr_true[i], n_points, fpr_range)
        for i in range(len(l_fpr_true))
    ]).transpose(1, 0, 2)
    l_fpr_true = l_fpr_tpr_true_interp[0, ...]
    l_tpr_true = l_fpr_tpr_true_interp[1, ...]

    l_pr_rc_true_interp = np.asarray([
        utls.my_interp(l_rc_true[i], l_pr_true[i], n_points, rc_range)
        for i in range(len(l_rc_true))
    ]).transpose(1, 0, 2)
    l_rc_true = l_pr_rc_true_interp[0, ...]
    l_pr_true = l_pr_rc_true_interp[1, ...]

    roc_xlim = [0, 1]
    pr_rc_xlim = [0, 1]
    logger.info('Concatenating results for scoring')
    all_y_true = np.concatenate([r['y_test'] for r in res_list])
    all_probas_my = np.concatenate([r['probas_my'] for r in res_list])
    all_probas_true = np.concatenate([r['probas_true'] for r in res_list])

    fpr_my_all, tpr_my_all, thresholds_my_all = roc_curve(all_y_true,
                                                          all_probas_my,
                                                          pos_label=1)
    fpr_my_all, tpr_my_all = utls.my_interp(fpr_my_all, tpr_my_all, n_points)

    fpr_true_all, tpr_true_all, thresholds_true_all = roc_curve(
        all_y_true, all_probas_true, pos_label=1)
    fpr_true_all, tpr_true_all = utls.my_interp(fpr_true_all, tpr_true_all,
                                                n_points)
    pr_my_all, rc_my_all, _ = precision_recall_curve(all_y_true, all_probas_my)
    pr_my_all, rc_my_all = utls.my_interp(pr_my_all, rc_my_all, n_points)
    pr_true_all, rc_true_all, _ = precision_recall_curve(
        all_y_true, all_probas_true)
    pr_true_all, rc_true_all = utls.my_interp(pr_true_all, rc_true_all,
                                              n_points)
    auc_my_all = auc(fpr_my_all, tpr_my_all)
    probas_thr = np.linspace(0, 1, n_points)
    f1_my = [f1_score(all_y_true, all_probas_my > p) for p in probas_thr]
    probas_thr = np.linspace(0, 1, 200)
    f1_true = [f1_score(all_y_true, all_probas_true > p) for p in probas_thr]
    auc_true_all = auc(fpr_true_all, tpr_true_all)

    # Plotting
    lw = 3
    plt.figure('tpr')
    plt.plot(l_fpr_true.mean(axis=0),
             l_tpr_true.mean(axis=0),
             '-',
             lw=lw,
             color=colors[0],
             label='all folds (true) (area = %0.4f, max_f1 = %0.4f)' %
             (auc_true_all, np.max(f1_true)))

    plt.fill_between(l_fpr_true.mean(axis=0),
                     l_tpr_true.mean(axis=0) + l_tpr_true.std(axis=0),
                     l_tpr_true.mean(axis=0) - l_tpr_true.std(axis=0),
                     facecolor=colors[0],
                     alpha=alpha)

    plt.plot(l_fpr_my.mean(axis=0),
             l_tpr_my.mean(axis=0),
             '-',
             lw=lw,
             color=colors[1],
             label='all folds (my) (area = %0.4f, max_f1 = %0.4f)' %
             (auc_my_all, np.max(f1_my)))

    plt.fill_between(l_fpr_my.mean(axis=0),
                     l_tpr_my.mean(axis=0) + l_tpr_my.std(axis=0),
                     l_tpr_my.mean(axis=0) - l_tpr_my.std(axis=0),
                     facecolor=colors[1],
                     alpha=alpha)
    plt.legend()
    plt.xlim(roc_xlim)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.suptitle('Sequence: ' + seq_type + '. Gaze: ' +
                 confs[0].csvFileName_fg)
    plt.savefig(os.path.join(out_dir, 'folds_tpr_fpr.pdf'))

    plt.figure('rc')
    plt.plot(l_rc_true.mean(axis=0),
             l_pr_true.mean(axis=0),
             '-',
             lw=lw,
             color=colors[0],
             label='all folds (true)')
    plt.fill_between(l_rc_true.mean(axis=0),
                     l_pr_true.mean(axis=0) + l_pr_true.std(axis=0),
                     l_pr_true.mean(axis=0) - l_pr_true.std(axis=0),
                     facecolor=colors[0],
                     alpha=alpha)
    plt.plot(l_rc_my.mean(axis=0),
             l_pr_my.mean(axis=0),
             '-',
             lw=lw,
             color=colors[1],
             label='all folds (my)')
    plt.fill_between(l_rc_my.mean(axis=0),
                     l_pr_my.mean(axis=0) + l_pr_my.std(axis=0),
                     l_pr_my.mean(axis=0) - l_pr_my.std(axis=0),
                     facecolor=colors[1],
                     alpha=alpha)
    plt.legend()
    plt.xlim(pr_rc_xlim)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.suptitle('Sequence: ' + seq_type + '. Gaze: ' +
                 confs[0].csvFileName_fg)
    #plt.figure('rc').set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(out_dir, 'folds_pr_rc.pdf'))

    min_n_frames = np.min([len(d.conf.frameFileNames) for d in datasets])

    dir_frames = os.path.join(out_dir, 'frames')

    if (not os.path.exists(dir_frames)):
        os.mkdir(dir_frames)
    else:
        logger.info('frames already exist, delete and re-run...')
        #shutil.rmtree(dir_frames)
        #os.mkdir(dir_frames)

    logger.info('Generating prediction frames...')
    #Plot by-frame predictions
    for f in range(min_n_frames):
        my = []
        true = []
        ims = []
        for j in range(len(datasets)):

            y_true = datasets[j].y
            idx_y = np.where(y_true[:, 0] == f)[0]
            y_true = y_true[idx_y]
            probas_true = res_list[j]['probas_true'][idx_y]
            probas_my = res_list[j]['probas_my'][idx_y]

            scores_my = utls.get_scores_from_sps(y_true[:, 0:2],
                                                 datasets[j].get_labels(),
                                                 probas_my)[..., f]
            my.append(scores_my)

            scores_true = utls.get_scores_from_sps(y_true[:, 0:2],
                                                   datasets[j].get_labels(),
                                                   probas_true)[..., f]
            true.append(scores_true)

            cont_gt = segmentation.find_boundaries(datasets[j].gt[..., f],
                                                   mode='thick')
            idx_cont_gt = np.where(cont_gt)
            im = utls.imread(datasets[j].conf.frameFileNames[f])
            im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
            im = gaze.drawGazePoint(datasets[j].conf.myGaze_fg,
                                    f,
                                    im,
                                    radius=7)
            ims.append(im)
        gs = gridspec.GridSpec(3, 4)
        for c in range(4):
            ax = plt.subplot(gs[0, c])
            ax.imshow(true[c])
            plt.title('true')

            ax = plt.subplot(gs[1, c])
            ax.imshow(my[c])
            plt.title('my')

            ax = plt.subplot(gs[2, c])
            ax.imshow(ims[c])
            plt.title('image')

        plt.suptitle('Sequence: ' + seq_type + '. Frame: ' + str(f))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(os.path.join(dir_frames, 'frame_' + str(f) + '.png'))
