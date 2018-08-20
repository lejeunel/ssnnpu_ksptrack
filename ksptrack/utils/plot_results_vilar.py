from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import progressbar
import sys
import os
import numpy as np
import gazeCsv as gaze
import matplotlib.pyplot as plt
from skimage import (color, segmentation)
import my_utils as utls
import dataset_vilar as ds
import selective_search as ss
import learning_dataset
import logging
import pandas as pd

def main(conf, plot_fname='metrics.pdf', csv_fname='score.csv', logger=None):

    logger = logging.getLogger('plot_results_vilar')

    logger.info('--------')
    logger.info('Self-learning on: ' + conf.dataOutDir)
    logger.info('--------')

    if (not os.path.exists(os.path.join(conf.dataOutDir, 'metrics.npz'))):

        my_dataset = ds.DatasetVilar(conf)
        my_dataset.load_labels_if_not_exist()
        #l_dataset = learning_dataset.Dataset(conf)
        l_dataset = learning_dataset.LearningDataset(conf)

        logger.info('[1/4] Loading predicted frames... ')
        pred_frames = np.asarray([my_dataset.get_pred_frame(f) for f in range(len(conf.frameFileNames))]).transpose(1,2,0)

        logger.info('[2/4] Extracting seeds... ')
        seeds = utls.make_y_array_true(pred_frames,my_dataset.labels)

        l_dataset.make_y_array_true(l_dataset.gt)
        seeds_true = l_dataset.y_true

        logger.info('[3/4] Calculating metrics... ')
        f1 = f1_score(seeds_true[:,2], seeds[:,2])
        logger.info('f1 score: ' + str(f1))

        logger.info('[4/4] Calculating maps... ')

        # Saving metrics
        data = dict()
        data['f1'] = f1
        data['seeds'] = seeds
        data['seeds_true'] = seeds_true

        np.savez(os.path.join(conf.dataOutDir, 'metrics.npz'), **data)

    else:
        logger.info('Loading metrics.npz...')
        metrics = np.load(os.path.join(conf.dataOutDir, 'metrics.npz'))
        f1 = metrics['f1']
        seeds = metrics['seeds']
        seeds_true = metrics['seeds_true']

        my_dataset = ds.DatasetVilar(conf)
        my_dataset.load_labels_if_not_exist()
        l_dataset = learning_dataset.LearningDataset(conf, pos_thr=0.5)

    csv_out = os.path.join(conf.dataOutDir, csv_fname)
    logger.info('Saving f1 scores to: ' + csv_fname)
    C = pd.Index(["F1"], name="columns")
    data = np.asarray(f1).reshape(1,1)
    df = pd.DataFrame(data=data, columns=C)
    df.to_csv(path_or_buf=csv_out)

    # Plot all iterations of PM

    # Make plots
    logger.info('Saving frames...')
    gt = l_dataset.gt
    frame_dir = 'vilar_frames'
    frame_path = os.path.join(conf.dataOutDir, frame_dir)
    if (os.path.exists(frame_path)):
        logger.info('[!!!] Frame dir: ' + frame_path +
                    ' exists. Delete to rerun.')
    else:
        os.mkdir(frame_path)
        seeds_true = seeds_true[np.where(seeds_true[:, 2])[0], 0:2]
        seeds = seeds[np.where(seeds[:, 2])[0], 0:2]
        scores_true = utls.seeds_to_scores(my_dataset.labels,
                                            seeds_true)
        scores = utls.seeds_to_scores(my_dataset.labels,
                                            seeds)
        with progressbar.ProgressBar(maxval=len(conf.frameFileNames)) as bar:
            for f in range(len(conf.frameFileNames)):
                cont_gt = segmentation.find_boundaries(
                    gt[..., f], mode='thick')
                idx_cont_gt = np.where(cont_gt)
                im = utls.imread(conf.frameFileNames[f])
                im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)
                im =  gaze.drawGazePoint(conf.myGaze_fg,f,im,radius=7)
                pred_frame = my_dataset.get_pred_frame(f)

                bar.update(f)
                plt.subplot(321)
                plt.imshow(scores_true[..., f])
                plt.title('True')
                plt.subplot(322)
                plt.imshow(scores[..., f])
                plt.title('Vilarino')
                plt.subplot(323)
                plt.imshow(im)
                plt.title('image')
                plt.subplot(324)
                plt.imshow(pred_frame)
                plt.title('pixel prediction')
                plt.suptitle('frame: ' + str(f))
                plt.savefig(
                    os.path.join(frame_path, 'f_' + str(f) + '.png'), dpi=200)


if __name__ == "__main__":
    main(sys.argv)
