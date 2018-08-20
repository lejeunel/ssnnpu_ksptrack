from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import progressbar
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import (color, segmentation)
import logging
import pandas as pd
from ksptrack.utils import my_utils as utls
from ksptrack.utils import learning_dataset
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import data_manager as ds
from ksptrack.exps import results_dirs as rd
from ksptrack.cfgs import cfg
from ksptrack.tr import Tracklet


def main(conf, plot_fname='metrics.pdf', logger=None, skip_frames=False):

    logger = logging.getLogger('plot_results_ksp')

    logger.info('--------')
    logger.info('Self-learning on: ' + conf.dataOutDir)
    logger.info('--------')

    metrics_path = os.path.join(conf.dataOutDir, 'metrics.npz')

    if (not os.path.exists(metrics_path)):

        res = np.load(
            os.path.join(conf.dataOutDir, 'results.npz'))

        list_paths_back = res['list_paths_back']
        list_paths_for = res['list_paths_for']

        my_dataset = ds.DataManager(conf)
        my_dataset.load_labels_if_not_exist()
        #my_dataset.load_pm_fg_from_file()

        l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)
        #l_dataset.make_y_array_true(l_dataset.gt)

        fpr_ksp = [0.]
        tpr_ksp = [0.]
        pr_ksp = [1.]
        rc_ksp = [0.]
        f1_ksp = []

        logger.info('[1/4] Calculating metrics on KSP... ')

        seeds = utls.list_paths_to_seeds(list_paths_for,
                                        list_paths_back)
        l_dataset.set_seeds(seeds)
        #l_dataset.make_y_array(l_dataset.seeds)

        ksp_scores = res['ksp_scores_mat']
        fpr, tpr, _ = roc_curve(l_dataset.gt.ravel(),
                                ksp_scores.ravel())
        precision, recall, _ = precision_recall_curve(l_dataset.gt.ravel(),
                                            ksp_scores.ravel())
        num = precision * recall
        denum = precision + recall
        f1_ = np.nan_to_num(2 * (num) / (denum))
        f1_ksp.append(np.max(f1_))
        fpr_ksp.append(fpr[1])
        tpr_ksp.append(tpr[1])
        pr_ksp.append(precision[1])
        rc_ksp.append(recall[1])

        fpr_ksp.append(1.)
        tpr_ksp.append(1.)
        pr_ksp.append(0.)
        rc_ksp.append(1.)

        logger.info('[2/4] Calculating metrics on PM... ')
        fpr_pm = []
        tpr_pm = []
        pr_pm = []
        rc_pm = []
        f1_pm = []

        seeds = utls.list_paths_to_seeds(list_paths_for,
                                        list_paths_back)
        l_dataset.fg_marked = seeds
        l_dataset.calc_pm(
            l_dataset.fg_marked,
            save=False,
            marked_feats=None,
            all_feats_df=my_dataset.sp_desc_df,
            in_type='not csv',
            mode='foreground',
            max_n_feats=conf.max_n_feats,
            feat_fields=['desc'],
            n_jobs=conf.bagging_jobs)

        pm = l_dataset.get_pm_array()

        fpr, tpr, _ = roc_curve(l_dataset.gt.ravel(),
                                pm.ravel())
        #fpr, tpr, _ = roc_curve(l_dataset.y_true[:, 2], probas)
        precision, recall, _ = precision_recall_curve(l_dataset.gt.ravel(),
                                                      pm.ravel())
        fpr, tpr, _ = roc_curve(l_dataset.gt.ravel(),
                                pm.ravel())
        num = precision * recall
        denum = precision + recall
        f1_ = np.nan_to_num(2 * (num) / (denum))
        f1_pm.append(np.max(f1_))
        fpr_pm.append(fpr)
        tpr_pm.append(tpr)
        pr_pm.append(precision)
        rc_pm.append(recall)


        # Saving metrics
        data = dict()
        data['fpr_pm'] = fpr_pm
        data['tpr_pm'] = tpr_pm
        data['pr_pm'] = pr_pm
        data['rc_pm'] = rc_pm
        data['f1_pm'] = f1_pm

        data['fpr_ksp'] = fpr_ksp
        data['tpr_ksp'] = tpr_ksp
        data['pr_ksp'] = pr_ksp
        data['rc_ksp'] = rc_ksp
        data['f1_ksp'] = f1_ksp

        data['ksp_scores'] = ksp_scores
        data['pm'] = pm
        np.savez(os.path.join(conf.dataOutDir, 'metrics.npz'), **data)

    else:
        logger.info('Loading metrics.npz...')
        metrics = np.load(os.path.join(conf.dataOutDir, 'metrics.npz'))
        fpr_pm = metrics['fpr_pm']
        tpr_pm = metrics['tpr_pm']
        pr_pm = metrics['pr_pm']
        rc_pm = metrics['rc_pm']
        f1_pm = metrics['f1_pm']

        fpr_ksp = metrics['fpr_ksp']
        tpr_ksp = metrics['tpr_ksp']
        pr_ksp = metrics['pr_ksp']
        rc_ksp = metrics['rc_ksp']
        f1_ksp = metrics['f1_ksp']

        ksp_scores = metrics['ksp_scores']
        if('pm' in metrics.keys()):
            pm = metrics['pm']
        else:
            pm = metrics['pm_ksp']

        my_dataset = ds.DataManager(conf)
        my_dataset.load_labels_if_not_exist()
        my_dataset.load_pm_fg_from_file()
        l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)

        res = np.load(os.path.join(conf.dataOutDir, 'results.npz'))
        list_paths_back = res['list_paths_back']
        list_paths_for = res['list_paths_for']

    # Plot all iterations of PM
    conf.roc_xlim = [0, 0.4]
    conf.pr_rc_xlim = [0.6, 1.]

    lw = 1

    # PM curves
    auc_ = auc(np.asarray(fpr_pm[-1]).ravel(), np.asarray(tpr_pm[-1]).ravel())
    max_f1 = np.max(f1_pm[-1])

    plt.subplot(121)
    plt.plot(
        np.asarray(fpr_pm[-1]).ravel(),
        np.asarray(tpr_pm[-1]).ravel(),
        'r-',
        lw=lw,
        label='KSP/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_, max_f1))

    auc_ = auc(np.asarray(rc_pm[-1]).ravel(), np.asarray(pr_pm[-1]).ravel())
    plt.subplot(122)
    plt.plot(
        np.asarray(rc_pm[-1]).ravel(),
        np.asarray(pr_pm[-1]).ravel(),
        'r-',
        lw=lw,
        label='KSP/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_, max_f1))

    # Plot KSP
    auc_ = auc(
        np.asarray(fpr_ksp).ravel(), np.asarray(tpr_ksp).ravel(), reorder=True)
    max_f1 = np.max(f1_ksp)
    plt.subplot(121)
    plt.plot(
        np.asarray(fpr_ksp).ravel(),
        np.asarray(tpr_ksp).ravel(),
        'ro--',
        lw=lw,
        label='KSP (area = %0.4f, max(F1) = %0.4f)' % (auc_, max_f1))
    plt.subplot(122)
    auc_ = auc(
        np.asarray(rc_ksp).ravel(), np.asarray(pr_ksp).ravel(), reorder=True)
    plt.plot(
        np.asarray(rc_ksp).ravel(),
        np.asarray(pr_ksp).ravel(),
        'ro--',
        lw=lw,
        label='KSP (area = %0.4f, max(F1) = %0.4f)' % (auc_, max_f1))

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
    plt.suptitle(conf.seq_type + ', ' + conf.dataSetDir + '\n' + 'T: ' +
                 str(conf.T))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(conf.dataOutDir, plot_fname), dpi=200)

    # Make plots
    gt = l_dataset.gt
    frame_dir = 'ksp_pm_frames'
    frame_path = os.path.join(conf.dataOutDir, frame_dir)
    if(os.path.exists(frame_path)):
        logger.info('[!!!] Frame dir: ' + frame_path +
                    ' exists. Delete to rerun.')
    elif(skip_frames):
        logger.info('[!!!] Skipping saving of frames')
    else:
        logger.info('Saving KSP, PM...')
        n_iter_ksp = len(list_paths_back)

        if(conf.csvFileType == 'pandas'):
            locs2d = pd.read_csv(os.path.join(conf.dataInRoot,
                                                    conf.dataSetDir,
                                                    conf.gazeDir,
                                                    conf.csvFileName_fg))
        elif(conf.csvFileType == 'anna'):
            locs2d = utls.readCsv(os.path.join(conf.dataInRoot,
                                                    conf.dataSetDir,
                                                    conf.gazeDir,
                                                    conf.csvFileName_fg))
        os.mkdir(frame_path)
        with progressbar.ProgressBar(maxval=len(conf.frameFileNames)) as bar:
            for f in range(len(conf.frameFileNames)):
                cont_gt = segmentation.find_boundaries(
                    gt[..., f], mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(conf.frameFileNames[f])
                im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
                im = csv.draw2DPoint(utls.pandas_to_std_csv(locs2d),
                                        f, im, radius=7)

                bar.update(f)
                plt.subplot(221)
                plt.imshow(ksp_scores[..., f])
                plt.title('KSP')
                plt.subplot(222)
                plt.imshow(pm[..., f])
                plt.title('KSP -> PM')
                plt.subplot(223)
                plt.imshow(im)
                plt.title('image')
                plt.suptitle('frame: ' + str(f) + ', n_iter_ksp: ' +
                             str(n_iter_ksp))
                plt.savefig(
                    os.path.join(frame_path, 'f_' + str(f) + '.png'), dpi=200)

    logger.info('Saving SPs per iterations plot...')
    n_sps = []
    for i in range(len(list_paths_back)):

        seeds = utls.list_paths_to_seeds(list_paths_for,
                                         list_paths_back,
                                         iter_=i)
        n = seeds.shape[0]
        n_sps.append((i + 1, n))

    n_sps.append((len(list_paths_back) + 1, n))
    n_sps = np.asarray(n_sps)

    plt.plot(n_sps[:, 0], n_sps[:, 1], 'bo-')
    plt.plot(n_sps[-1, 0], n_sps[-1, 1], 'ro')
    plt.xlabel('iterations')
    plt.ylabel('num. of superpixels')
    plt.title('num of superpixels vs. iterations. SS threshold: ' +
              str(conf.ss_thr))
    plt.savefig(os.path.join(conf.dataOutDir, 'sps_iters.eps'), dpi=200)

    pr_pm, rc_pm, _ = precision_recall_curve(l_dataset.gt.ravel(),pm.ravel())
    ksp_pm_pix_f1 = np.max(2 * (pr_pm * rc_pm) / (pr_pm + rc_pm))
    ksp_pix_f1 = f1_score(l_dataset.gt.ravel(), ksp_scores.ravel())

    file_out = os.path.join(conf.dataOutDir, 'scores.csv')
    logger.info('Saving to {}'.format(file_out))
    C = pd.Index(["F1"], name="columns")
    I = pd.Index(['KSP', 'KSP/PM'], name="Methods")
    data = np.asarray([ksp_pix_f1, ksp_pm_pix_f1]).reshape(2, 1)
    df = pd.DataFrame(data=data, index=I, columns=C)
    df.to_csv(path_or_buf=file_out)


if __name__ == "__main__":
    main(sys.argv)

#dir_ = os.path.join(rd.root_dir,
#                    'Dataset30/results/2017-11-07_14-49-56_exp')
