from sklearn.metrics import (f1_score,roc_curve,auc,precision_recall_curve)
import warnings, itertools, _pickle, progressbar, sys, os, datetime, yaml, hashlib
from labeling.utils import my_utils as utls
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labeling.utils import learning_dataset as ld
from skimage import (color, segmentation, util, transform, io)
from labeling.utils import csv_utils as csv
from labeling.exps import results_dirs as rd
import shutil as sh

n_decs = 2
dsets_per_type = 1

out_result_dir = os.path.join(rd.root_dir, 'plots_results')

# Steps

n_sets_per_type = 2

cmap = plt.get_cmap('viridis')

ind = 0

for key in rd.types:

    dsets_to_plot = np.asarray(rd.best_dict_ksp[key][0:n_sets_per_type])
    for dset, gset in zip(dsets_to_plot[:,0], dsets_to_plot[:,1]):

        path_out = os.path.join(rd.root_dir, 'plots_results',
                                (key + '_{}_{}').format(ind, dset))

        ims = []

        # Make images/gts/gaze-point
        confs = rd.confs_dict_ksp[key][dset]
        conf = confs[0]

        dataset = ld.LearningDataset(conf)
        gt = dataset.gt

        ksp_mat = np.load(os.path.join(rd.root_dir,
                                rd.res_dirs_dict_ksp[key][dset][gset],
                                'results.npz'))['ksp_scores_mat']

        for f in range(len(conf.frameFileNames)):
            cont_gt = segmentation.find_boundaries(
                gt[..., f], mode='thick')
            idx_cont_gt = np.where(cont_gt)
            im = utls.imread(conf.frameFileNames[f])
            im[idx_cont_gt[0], idx_cont_gt[1], :] = (255, 0, 0)

            myGaze_fg = utls.readCsv(os.path.join(conf.dataInRoot,
                                                    conf.dataSetDir,
                                                    conf.gazeDir,
                                                    conf.csvFileName_fg))
            im = csv.draw2DPoint(myGaze_fg,
                                    f,
                                    im,
                                    radius=7)

            ksp_ = (cmap(ksp_mat[..., f]*255)[..., 0:3]*255).astype(np.uint8)
            im = np.concatenate((im, ksp_),
                                axis=0)

            ims.append(im)

        # Save frames
        if(not os.path.exists(path_out)):
            os.mkdir(path_out)
            for f, im in enumerate(ims):
                io.imsave(os.path.join(path_out, 'f{}.png'.format(f)),
                          im)
        else:
            print(path_out + ' exists!')

    ind += 1
