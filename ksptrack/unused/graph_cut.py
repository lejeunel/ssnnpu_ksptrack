from pygco import cut_from_graph
import numpy as np
from sklearn import metrics
from skimage import (color, io, segmentation)
import my_utils as utls
import matplotlib.pyplot as plt
import progressbar


class graph_cut:
    def __init__(self,
                 conf,
                 fg_pm,
                 bg_pm,
                 scores_ksp,
                 labels,
                 gt,
                 gamma=None,
                 lambda_=None):
        self.conf = conf
        self.fg_pm = fg_pm
        self.labels = labels
        self.scores_ksp = scores_ksp
        self.gc_maps = None
        self.gc_scores = None
        self.lambda_ = None
        self.gamma = None

        #If no bg model,
        if (bg_pm is None):
            self.bg_pm = self.get_comp_scores_ksp(
                self.labels)  #Will call calc_bagging
        else:
            self.bg_pm = bg_pm

        if (lambda_ is None): self.lambda_ = 50
        else: self.lambda_ = lambda_
        if (gamma is None): self.gamma = 0.8
        else: self.gamma = gamma

        if (self.fg_pm.shape[0] != len(self.conf.frameFileNames)):
            self.fg_pm = self.fg_pm.transpose((2, 0, 1))
        if (self.bg_pm.shape[0] != len(self.conf.frameFileNames)):
            self.bg_pm = self.bg_pm.transpose((2, 0, 1))
        if (self.scores_ksp.shape[0] != len(self.conf.frameFileNames)):
            self.scores_ksp = self.scores_ksp.transpose((2, 0, 1))
        if (gt.shape[0] != len(self.conf.frameFileNames)):
            self.gt = gt.transpose((2, 0, 1))

        self.gc_maps = np.zeros(self.scores_ksp.shape)

    def get_scores(self):

        print('Calculating confusion matrix...')
        conf_mat = metrics.confusion_matrix(self.gt.ravel(),
                                            self.gc_maps.ravel())
        precision = float(
            conf_mat[1, 1]) / float(conf_mat[1, 1] + conf_mat[0, 1])
        recall = float(conf_mat[1, 1]) / float(conf_mat[1, 1] + conf_mat[1, 0])
        f1 = float(2 * conf_mat[1, 1]) / float(2 * conf_mat[1, 1] +
                                               conf_mat[0, 1] + conf_mat[1, 0])
        print('...done')

        return (conf_mat, f1)

    def run(self):

        small = np.finfo(np.float).eps

        norm_scores = self.scores_ksp.astype(float) / np.max(self.scores_ksp)
        heat_maps = np.zeros(self.scores_ksp.shape)
        self.gc_scores = None
        print("(gamma,lambda_) = (" + str(self.gamma) + ',' +
              str(self.lambda_) + ")")
        with progressbar.ProgressBar(
                maxval=len(self.conf.frameFileNames)) as bar:
            for j in range(len(self.conf.frameFileNames)):
                #for j in np.array([10,20]):
                #for j in np.asarray([14]):
                bar.update(j)
                min_p = 0.01
                n = 1000

                this_pm_fg = self.fg_pm[j, :, :].copy()
                this_pm_bg = self.bg_pm[j, :, :].copy()

                this_fg = 0.5 * np.ones(this_pm_fg.shape)
                this_fg[self.scores_ksp[j, :, :] > 0] = 1

                this_bg = 0.5 * np.ones(this_pm_fg.shape)
                this_bg[this_pm_fg == 0] = 1

                this_pm_fg = np.clip(this_pm_fg, a_max=1.0, a_min=min_p)
                this_pm_bg = np.clip(this_pm_bg, a_max=1.0, a_min=min_p)

                this_fg_costs = -np.log(this_fg + small) / -np.log(
                    small) - self.gamma * np.log(this_pm_fg +
                                                 small) / -np.log(small)
                this_bg_costs = -np.log(this_bg + small) / -np.log(
                    small) - self.gamma * np.log(this_pm_bg +
                                                 small) / -np.log(small)

                #Digitize
                bin_min = np.min([this_fg_costs, this_bg_costs])
                bin_max = np.max([this_fg_costs, this_bg_costs])
                bins = np.linspace(bin_min, bin_max, n)

                this_fg_costs = np.digitize(this_fg_costs, bins)
                this_bg_costs = np.digitize(this_bg_costs, bins)

                unaries = ((np.dstack([this_fg_costs, this_bg_costs
                                       ]).copy("C"))).astype(np.int32)

                # use the gerneral graph algorithm
                # first, we construct the grid graph
                inds = np.arange(self.scores_ksp[j, :, :].size).reshape(
                    self.scores_ksp[j, :, :].shape)
                horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
                vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]

                orig_image = color.rgb2gray(
                    utls.imread(self.conf.frameFileNames[j]))
                dx_im = orig_image[:, 0:-1] - orig_image[:, 1::]
                dy_im = orig_image[0:-1, :] - orig_image[1::, :]
                mean_squared = np.mean(
                    np.hstack([(dx_im**2).ravel(), (dy_im**2).ravel()]))
                beta = 1. / (2 * mean_squared)
                horz_weights = np.exp(-dx_im**2 * beta)
                vert_weights = np.exp(-dy_im**2 * beta)

                horz_weights = (n * horz_weights).astype(np.int32)
                vert_weights = (n * vert_weights).astype(np.int32)

                #pairwise = -100* np.eye(2, dtype=np.int32)
                pairwise = 1 - np.eye(2, dtype=np.int32)

                weights = np.vstack([
                    self.lambda_ * horz_weights.reshape(-1, 1),
                    self.lambda_ * vert_weights.reshape(-1, 1)
                ])
                edges_idx = np.vstack([horz, vert]).astype(np.int32)
                edges = np.hstack([edges_idx, weights]).astype(np.int32)

                this_gc = np.logical_not(
                    cut_from_graph(edges, unaries.reshape(-1, 2),
                                   pairwise).reshape(
                                       self.scores_ksp[j, :, :].shape))
                self.gc_maps[j, :, :] = this_gc
            #self.gc_scores = (self.lambda_, self.gamma, f1, precision, recall)
        #print(self.gc_scores)
