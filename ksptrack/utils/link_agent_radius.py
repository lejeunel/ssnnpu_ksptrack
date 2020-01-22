import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from itertools import count
from ksptrack.utils import csv_utils as csv
import glob
from skimage.draw import circle
from skimage import io
from ksptrack.utils.lfda import myLFDA
from sklearn.decomposition import PCA
import warnings
import torch
from ksptrack.utils.link_agent import LinkAgent


class LinkAgentRadius(LinkAgent):
    def __init__(self,
                 csv_path,
                 data_path,
                 thr_entrance=0.5,
                 sigma=0.07,
                 entrance_radius=None):

        super().__init__(csv_path, data_path, thr_entrance)

        self.entrance_radius = entrance_radius
        self.thr_entrance = thr_entrance
        self.sigma = sigma

    def get_all_entrance_sps(self, sp_desc_df):

        sps = []

        for f in range(self.labels.shape[-1]):
            if(f in self.locs['frame'].to_numpy()):
                for i, loc in self.locs[self.locs['frame'] == f].iterrows():
                    i, j = self.get_i_j(loc)
                    label = self.labels[i, j, f]
                    sps += [(f, label)]

        return sps

    def make_entrance_mask(self, frame):
        mask = np.zeros(self.shape, dtype=bool)
        all_locs = [
            self.get_i_j(loc)
            for _, loc in self.locs[self.locs['frame'] == frame].iterrows()
        ]
        for loc in all_locs:
            rr, cc = circle(loc[0],
                            loc[1],
                            self.shape[0] * self.entrance_radius,
                            shape=self.shape)
            mask[rr, cc] = True
        return mask

    def get_proba_entrance(self, tl, tl_loc, sp_desc):

        label_user = self.get_closest_label(tl, tl_loc)
        label_tl = tl.get_in_label()
        frame_tl = tl.get_in_frame()
        frame_user = tl.get_in_frame()

        return self.get_proba(sp_desc, frame_user, label_user, frame_tl,
                              label_tl)

    def get_proba_inter_frame(self, tracklet1, tracklet2, sp_desc):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(sp_desc, frame_1, label_1, frame_2, label_2)

        return proba

    def get_distance(self, sp_desc, f1, l1, f2, l2, p=2):
        d1 = sp_desc.loc[(sp_desc['frame'] == f1) &
                         (sp_desc['label'] == l1), 'desc'].values[0][None, ...]
        d2 = sp_desc.loc[(sp_desc['frame'] == f2) &
                         (sp_desc['label'] == l2), 'desc'].values[0][None, ...]
        d1 = self.trans_transform.transform(d1)
        d2 = self.trans_transform.transform(d2)

        dist = np.linalg.norm(d1 - d2, ord=p)
        return dist

    def get_proba(self, sp_desc, f1, l1, f2, l2):

        dist = self.get_distance(sp_desc, f1, l1, f2, l2)
        proba = np.exp((-dist**2) * self.sigma)
        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)

        return proba

    def update_trans_transform(self,
                               sp_desc,
                               pm,
                               threshs,
                               n_samps,
                               n_dims,
                               k,
                               embedding_type='weighted'):

        # descs_cat = utls.concat_arr(sp_desc['desc'])
        descs_cat = np.vstack(sp_desc['desc'].values)
        if (descs_cat.shape[1] == sp_desc.shape[0]):
            descs_cat = descs_cat.T

        self.trans_transform = myLFDA(n_components=n_dims,
                                      k=k,
                                      embedding_type=embedding_type)
        probas = pm['proba'].values
        if ((probas < threshs[0]).sum() < n_samps):
            sorted_probas = np.sort(probas)
            threshs[0] = sorted_probas[n_samps]
            warnings.warn('Not enough negatives. Setting thr to {}'.format(
                threshs[0]))
        if ((probas > threshs[1]).sum() < n_samps):
            sorted_probas = np.sort(probas)[::-1]
            threshs[1] = sorted_probas[n_samps]
            warnings.warn('Not enough positives. Setting thr to {}'.format(
                threshs[1]))
        self.trans_transform.fit(descs_cat, pm['proba'].values, threshs,
                                 n_samps)
