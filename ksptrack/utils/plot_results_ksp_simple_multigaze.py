from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import os
import progressbar
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
from skimage import (color, segmentation)
import my_utils as utls
import dataset as ds
import selective_search as ss
import learning_dataset
import logging
import plot_results_ksp_simple as pksp
import pandas as pd

"""
Computes segmentation frames, ROC curves on single sequence with multiple gaze
for all iterations (KSP, KSP+SS, etc..)
"""



def plot_curves(out_dir, confs, plot_fname, metrics_fname, logger):
    l_fpr_pm_ss = list()
    l_tpr_pm_ss = list()
    l_pr_pm_ss = list()
    l_rc_pm_ss = list()
    l_f1_pm_ss = list()


    l_fpr_ksp_ss_thr = list()
    l_tpr_ksp_ss_thr = list()
    l_pr_ksp_ss_thr = list()
    l_rc_ksp_ss_thr = list()
    l_f1_ksp_ss_thr = list()

    l_fpr_pm = list()
    l_tpr_pm = list()
    l_pr_pm = list()
    l_rc_pm = list()
    l_f1_pm = list()

    l_pm_ksp_ss = list()
    l_pm_ksp = list()

    l_fpr_pm_thr = list()
    l_tpr_pm_thr = list()
    l_rc_pm_thr = list()
    l_pr_pm_thr = list()
    l_f1_pm_thr = list()
    l_pm_ksp_ss_thr = list()

    l_ksp_scores = list()
    l_ksp_ss_scores = list()
    l_ksp_ss_thr_scores = list()

    met_auc_pm = list()
    met_f1_pm = list()
    met_auc_pm_ss = list()
    met_f1_pm_ss = list()
    met_auc_pm_thr = list()
    met_f1_pm_thr = list()

    for i in range(len(confs)):

        file_ = os.path.join(confs[i].dataOutDir, 'metrics.npz')
        if(not os.path.exists(file_)):
            logger.info(file_ + ' does not exist. Calculating metrics')
            pksp.main(confs[i])
        logger.info('Loading ' + file_)
        npzfile = np.load(file_)

        l_fpr_pm_ss.append(npzfile['fpr_pm_ss'])
        l_tpr_pm_ss.append(npzfile['tpr_pm_ss'])
        l_pr_pm_ss.append(npzfile['pr_pm_ss'])
        l_rc_pm_ss.append(npzfile['rc_pm_ss'])
        l_f1_pm_ss.append(npzfile['f1_pm_ss'])

        l_fpr_ksp_ss_thr.append(npzfile['fpr_ksp_ss_thr'])
        l_tpr_ksp_ss_thr.append(npzfile['tpr_ksp_ss_thr'])
        l_pr_ksp_ss_thr.append(npzfile['pr_ksp_ss_thr'])
        l_rc_ksp_ss_thr.append(npzfile['rc_ksp_ss_thr'])
        l_f1_ksp_ss_thr.append(npzfile['f1_ksp_ss_thr'])

        l_fpr_pm.append(npzfile['fpr_pm'])
        l_tpr_pm.append(npzfile['tpr_pm'])
        l_pr_pm.append(npzfile['pr_pm'])
        l_rc_pm.append(npzfile['rc_pm'])
        l_f1_pm.append(npzfile['f1_pm'])

        l_pm_ksp_ss.append(npzfile['pm_ksp_ss'])
        l_pm_ksp.append(npzfile['pm_ksp'])

        l_fpr_pm_thr.append(npzfile['fpr_pm_thr'])
        l_tpr_pm_thr.append(npzfile['tpr_pm_thr'])
        l_rc_pm_thr.append(npzfile['rc_pm_thr'])
        l_pr_pm_thr.append(npzfile['pr_pm_thr'])
        l_f1_pm_thr.append(npzfile['f1_pm_thr'])
        l_pm_ksp_ss_thr.append(npzfile['pm_ksp_ss_thr'])


    # Make ROC curves
    # Plot all iterations of PM
    plt.clf()
    confs[0].roc_xlim = [0, 0.4]
    confs[0].pr_rc_xlim = [0.6, 1.]
    alpha = 0.3

    lw = 1
    # PM curves
    aucs = [auc(np.asarray(l_fpr_pm[i]).ravel(), np.asarray(l_tpr_pm[i]).ravel(),reorder=True) for i in range(len(l_fpr_pm))]
    auc_roc = np.mean(aucs)
    met_auc_pm.append((np.mean(aucs),np.std(aucs)))
    auc_pr_rc = np.mean(
        [auc(np.asarray(l_pr_pm[i]).ravel(), np.asarray(l_rc_pm[i]).ravel(),reorder=True) for i in range(len(l_pr_pm))])
    max_f1s = [np.max(l_f1_pm[i].ravel()) for i in range(len(l_f1_pm))]
    max_f1 = np.mean(max_f1s)
    met_f1_pm.append((np.mean(max_f1s),np.std(max_f1s)))


    fpr, tpr = utls.concat_interp(l_fpr_pm, l_tpr_pm, 2000)
    rc, pr = utls.concat_interp(l_rc_pm, l_pr_pm, 2000)

    plt.subplot(121)
    plt.plot(
        fpr,
        tpr.mean(axis=0),
        'r-',
        lw=lw,
        label='KSP/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_roc, max_f1))
    plt.fill_between(fpr,
                     tpr.mean(axis=0)+tpr.std(axis=0),
                     tpr.mean(axis=0)-tpr.std(axis=0),
                     facecolor='r',
                     alpha=alpha)

    plt.subplot(122)
    plt.plot(
        rc,
        pr.mean(axis=0),
        'r-',
        lw=lw,
        label='KSP/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_pr_rc,
                                                            max_f1))
    plt.fill_between(rc,
                     pr.mean(axis=0)+pr.std(axis=0),
                     pr.mean(axis=0)-pr.std(axis=0),
                     facecolor='r',
                     alpha=alpha)

    # Plot KSP+SS PM
    aucs = [auc(np.asarray(l_fpr_pm_ss[i]).ravel(), np.asarray(l_tpr_pm_ss[i]).ravel(),reorder=True) for i in range(len(l_fpr_pm_ss))]
    auc_roc = np.mean(aucs)
    met_auc_pm_ss.append((np.mean(aucs),np.std(aucs)))
    auc_pr_rc = np.mean(
        [auc(np.asarray(l_pr_pm_ss[i]).ravel(), np.asarray(l_rc_pm_ss[i]).ravel(),reorder=True) for i in range(len(l_pr_pm_ss))])
    max_f1s = [np.max(l_f1_pm_ss[i].ravel()) for i in range(len(l_f1_pm_ss))]
    max_f1 = np.mean(max_f1s)
    met_f1_pm_ss.append((np.mean(max_f1s),np.std(max_f1s)))


    fpr, tpr = utls.concat_interp(l_fpr_pm_ss, l_tpr_pm_ss, 2000)
    rc, pr = utls.concat_interp(l_rc_pm_ss, l_pr_pm_ss, 2000)
    plt.subplot(121)
    plt.plot(
        fpr,
        tpr.mean(axis=0),
        'g-',
        lw=lw,
        label='KSP+SS/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_roc,
                                                                max_f1))
    plt.fill_between(fpr,
                     tpr.mean(axis=0)+tpr.std(axis=0),
                     tpr.mean(axis=0)-tpr.std(axis=0),
                     facecolor='g',
                     alpha=alpha)

    plt.subplot(122)
    plt.plot(
        rc,
        pr.mean(axis=0),
        'g-',
        lw=lw,
        label='KSP+SS/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_pr_rc,
                                                                max_f1))
    plt.fill_between(rc,
                     pr.mean(axis=0)+pr.std(axis=0),
                     pr.mean(axis=0)-pr.std(axis=0),
                     facecolor='g',
                     alpha=alpha)

    # Plot KSP+SS PM thresholded
    aucs = [auc(np.asarray(l_fpr_pm_thr[i]).ravel(), np.asarray(l_tpr_pm_thr[i]).ravel(),reorder=True) for i in range(len(l_fpr_pm_thr))]
    auc_roc = np.mean(aucs)
    met_auc_pm_thr.append((np.mean(aucs),np.std(aucs)))
    auc_pr_rc = np.mean(
        [auc(np.asarray(l_pr_pm_thr[i]).ravel(), np.asarray(l_rc_pm_thr[i]).ravel(),reorder=True) for i in range(len(l_pr_pm_thr))])
    max_f1s = [np.max(l_f1_pm_thr[i].ravel()) for i in range(len(l_f1_pm_thr))]
    max_f1 = np.mean(max_f1s)
    met_f1_pm_thr.append((np.mean(max_f1s),np.std(max_f1s)))

    fpr, tpr = utls.concat_interp(l_fpr_pm_thr, l_tpr_pm_thr, 2000)
    rc, pr = utls.concat_interp(l_rc_pm_thr, l_pr_pm_thr, 2000)
    plt.subplot(121)
    plt.plot(
        fpr,
        tpr.mean(axis=0),
        'b-',
        lw=lw,
        label='KSP+SS/PM (thr: %0.1f) (area = %0.4f, max(F1) = %0.4f)' %
        (confs[0].pm_thr, auc_roc, max_f1))
    plt.fill_between(fpr,
                     tpr.mean(axis=0)+tpr.std(axis=0),
                     tpr.mean(axis=0)-tpr.std(axis=0),
                     facecolor='b',
                     alpha=alpha)

    plt.subplot(122)
    plt.plot(
        rc,
        pr.mean(axis=0),
        'b-',
        lw=lw,
        label='KSP+SS/PM (thr: %0.1f) (area = %0.4f, max(F1) = %0.4f)' %
        (confs[0].pm_thr, auc_pr_rc, max_f1))
    plt.fill_between(rc,
                     pr.mean(axis=0)+pr.std(axis=0),
                     pr.mean(axis=0)-pr.std(axis=0),
                     facecolor='b',
                     alpha=alpha)

    plt.subplot(121)
    plt.legend()
    plt.xlim(confs[0].roc_xlim)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.subplot(122)
    plt.legend()
    plt.xlim(confs[0].pr_rc_xlim)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.suptitle(confs[0].seq_type + '(' + confs[0].ds_dir + ')' + ' Num. gaze sets: ' + str(len(confs)))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(out_dir, plot_fname+'_all.pdf'), dpi=200)


    rc, pr = utls.concat_interp(l_rc_pm, l_pr_pm, 2000)

    plt.plot(
        rc,
        pr.mean(axis=0),
        'r-',
        lw=lw,
        label='KSP/PM (area = %0.4f, max(F1) = %0.4f)' % (auc_pr_rc,
                                                            max_f1))
    plt.fill_between(rc,
                     pr.mean(axis=0)+pr.std(axis=0),
                     pr.mean(axis=0)-pr.std(axis=0),
                     facecolor='r',
                     alpha=alpha)

    plt.savefig(os.path.join(out_dir, plot_fname+'_pr.pdf'), dpi=200)



    logger.info('Done generating curves')

    file_out = os.path.join(out_dir,metrics_fname)
    logger.info('Writing mean and std of metrics to: ' + file_out)

    I = pd.Index(["mean AUC", "mean F1", "std AUC", "std F1"], name="rows")
    C = pd.Index(["ksp", "ksp+ss", "ksp+ss+thr"], name="columns")
    data = np.asarray([np.asarray([np.asarray(met_auc_pm), np.asarray(met_f1_pm)]),
            np.asarray([np.asarray(met_auc_pm_ss), np.asarray(met_f1_pm_ss)]),
            np.asarray([np.asarray(met_auc_pm_thr), np.asarray(met_f1_pm_thr)])]).T
    data = data.reshape(4,3)
    df = pd.DataFrame(data=data, index=I, columns=C)
    df.to_csv(path_or_buf=file_out)

    return True

def main(out_dir,
         confs,
         plot_fname='metrics',
         metrics_fname='metrics.csv',
         logger=None):

    logger = logging.getLogger('plot_results_ksp')

    out_dirs = [c.dataOutDir for c in confs]
    logger.info('--------')
    logger.info('Self-learning on: ')
    logger.info(out_dirs)
    logger.info('out_dir: ')
    logger.info(out_dir)
    logger.info('--------')

    l_dataset = learning_dataset.LearningDataset(confs[0], pos_thr=0.5)

    plot_curves(out_dir, confs, plot_fname, metrics_fname, logger)

    l_ksp_scores = list()
    l_ksp_ss_scores = list()
    l_ksp_ss_thr_scores = list()

    for i in range(len(confs)):

        file_ = os.path.join(confs[i].dataOutDir, 'metrics.npz')
        logger.info('Loading ' + file_)
        npzfile = np.load(file_)

        l_ksp_scores.append(npzfile['ksp_scores'])
        l_ksp_ss_scores.append(npzfile['ksp_ss_scores'])
        l_ksp_ss_thr_scores.append(npzfile['ksp_ss_thr_scores'])


    # Make plots
    mean_ksp_scores = np.mean(np.asarray(l_ksp_scores), axis=0)
    mean_ksp_ss_scores = np.mean(np.asarray(l_ksp_ss_scores), axis=0)
    mean_ksp_ss_thr_scores = np.mean(np.asarray(l_ksp_ss_thr_scores), axis=0)

    std_ksp_scores = np.std(np.asarray(l_ksp_scores), axis=0)
    std_ksp_ss_scores = np.std(np.asarray(l_ksp_ss_scores), axis=0)
    std_ksp_ss_thr_scores = np.std(np.asarray(l_ksp_ss_thr_scores), axis=0)

    path_ = os.path.join(out_dir, 'dataset.npz')
    data = dict()
    data['mean_ksp_scores'] = mean_ksp_scores
    data['mean_ksp_ss_scores'] = mean_ksp_ss_scores
    data['mean_ksp_ss_thr_scores'] = mean_ksp_ss_thr_scores
    data['std_ksp_scores'] = std_ksp_scores
    data['std_ksp_ss_scores'] = std_ksp_ss_scores
    data['std_ksp_ss_thr_scores'] = std_ksp_ss_thr_scores

    np.savez(path_, ** data)

    logger.info('Saving KSP, PM and SS merged frames...')
    gt = l_dataset.gt
    frame_dir = 'ksp_pm_frames'
    frame_path = os.path.join(out_dir, frame_dir)
    if (os.path.exists(frame_path)):
        logger.info('[!!!] Frame dir: ' + frame_path +
                    ' exists. Delete to rerun.')
    else:
        os.mkdir(frame_path)
        c0 = confs[0]
        with progressbar.ProgressBar(maxval=len(c0.frameFileNames)) as bar:
            for f in range(len(c0.frameFileNames)):
                cont_gt = segmentation.find_boundaries(
                    gt[..., f], mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(c0.frameFileNames[f])
                im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
                for c in confs:
                    im = gaze.drawGazePoint(c.myGaze_fg, f, im, radius=7)

                bar.update(f)
                plt.subplot(241)
                plt.imshow(mean_ksp_scores[..., f])
                plt.title('mean KSP')
                plt.subplot(242)
                plt.imshow(std_ksp_scores[..., f])
                plt.title('std KSP')
                plt.subplot(243)
                plt.imshow(mean_ksp_ss_scores[..., f])
                plt.title('mean KSP+SS')
                plt.subplot(244)
                plt.imshow(std_ksp_ss_scores[..., f])
                plt.title('std KSP+SS')
                plt.subplot(245)
                plt.imshow(mean_ksp_ss_thr_scores[..., f])
                plt.title('mean KSP+SS -> PM -> (thr = %0.2f)' % (c.pm_thr))
                plt.subplot(246)
                plt.imshow(std_ksp_ss_thr_scores[..., f])
                plt.title('std KSP+SS -> PM -> (thr = %0.2f)' % (c.pm_thr))
                plt.subplot(247)
                plt.imshow(im)
                plt.title('image')
                plt.suptitle('frame: ' + str(f))
                plt.savefig(
                    os.path.join(frame_path, 'f_' + str(f) + '.png'), dpi=200)
