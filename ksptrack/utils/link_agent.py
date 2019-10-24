import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from itertools import count
from ksptrack.utils import csv_utils as csv
import glob
from skimage.draw import circle
from ksptrack.utils.lfda import myLFDA
from sklearn.decomposition import PCA


class LinkAgent:
    def __init__(self,
                 shape,
                 logger,
                 csv_path,
                 thr_entrance=0.5,
                 mask_path=None,
                 entrance_radius=None):

        if ((entrance_radius is None) and (mask_path is None)):
            raise Exception(
                'entrance_radius and mask_path cannot be both None')

        if ((entrance_radius is not None) and (mask_path is not None)):
            raise Exception(
                'entrance_radius and mask_path cannot be both specified')

        if ((entrance_radius is not None) and (shape is None)):
            raise Exception(
                'when using entrance_radius you must specify input shape')

        self.entrance_radius = entrance_radius
        self.shape = shape
        self.mask_path = mask_path
        self.logger = logger
        self.trans_transform = None
        self.thr_clip = 0.05
        self.thr_entrance = thr_entrance

        self.locs = csv.readCsv(csv_path, as_pandas=True)

        if (mask_path is not None):
            self.mask_paths = sorted(glob.glob(pjoin(mask_path, '*.png')))
            self.masks = {
                f: plt.imread(p)
                for f, p in zip(self.locs['frame'], self.mask_paths)
            }


    def get_i_j(self, loc):
        i, j = csv.coord2Pixel(loc['x'], loc['y'], self.shape[1],
                               self.shape[0])
        return i, j

    def make_radius_mask(self, idx):
        mask = np.zeros(self.shape, dtype=bool)
        all_locs = [
            self.get_i_j(loc)
            for _, loc in self.locs[self.locs['frame'] == idx].iterrows()
        ]
        for loc in all_locs:
            rr, cc = circle(loc[0],
                            loc[1],
                            self.shape[0] * self.entrance_radius,
                            shape=self.shape)
            mask[rr, cc] = True
        return mask

    def is_entrance(self, loc):

        i, j = self.get_i_j(loc)
        if (self.mask_path is not None):
            frame = int(loc['frame'])
            if(frame in self.masks.keys()):
                map_value = self.masks[frame][i, j]
                return map_value > self.thr_entrance
            return False

        mask = self.make_radius_mask(int(loc['frame']))
        return mask[i, j]

    def get_closest_label(self, tl, tl_loc, labels):
        loc_compare = self.locs.loc[self.locs['frame'] == tl.get_in_frame()]
        dists = [
            np.linalg.norm(
                np.array((tl_loc['x'], tl_loc['y'])) - np.array(r_compare))
            for r_compare in [(
                x, y) for x, y in zip(loc_compare['x'], loc_compare['y'])]
        ]
        loc_min = self.locs.loc[np.argmin(dists)]
        i_min, j_min = csv.coord2Pixel(loc_min['x'], loc_min['y'],
                                       self.shape[1], self.shape[0])
        return labels[i_min, j_min, tl.get_in_frame()]

    def get_proba_entrance(self, tl, tl_loc, sp_desc, labels):

        if (self.mask_path is None):
            label_user = self.get_closest_label(tl, tl_loc, labels)
            label_tl = tl.get_in_label()
            frame_tl = tl.get_in_frame()
            frame_user = tl.get_in_frame()

            return self.get_proba(sp_desc, frame_user, label_user, frame_tl,
                                  label_tl)
        else:
            # compute average probability on mask on label occupied by tl
            if(tl.get_in_frame() in self.masks.keys()):
                label_mask = labels[..., tl.get_in_frame()] == tl.get_in_label()

                proba = np.mean(self.masks[tl.get_in_frame()][label_mask])

                proba = np.clip(proba,
                                a_min=self.thr_clip,
                                a_max=1 - self.thr_clip)
                return proba
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2, sp_desc, mode):

        if (mode == 'tail'):  # Invert order
            t1 = tracklet2
            t2 = tracklet1
        else:
            t1 = tracklet1
            t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(sp_desc, frame_1, label_1, frame_2, label_2)

        return proba

    def get_proba(self, sp_desc, f1, l1, f2, l2):
        d1 = sp_desc.loc[(sp_desc['frame'] == f1) &
                         (sp_desc['label'] == l1), 'desc'].values[0][None, ...]
        d2 = sp_desc.loc[(sp_desc['frame'] == f2) &
                         (sp_desc['label'] == l2), 'desc'].values[0][None, ...]
        d1 = self.trans_transform.transform(d1)
        d2 = self.trans_transform.transform(d2)

        dist = np.linalg.norm(d1 - d2)

        proba = np.exp(-dist**2)
        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)

        return proba

    def make_trans_transform(self,
                             sp_desc,
                             pm,
                             thresh,
                             n_samps,
                             n_dims,
                             k,
                             pca=False,
                             n_comps_pca=3):

        # descs_cat = utls.concat_arr(sp_desc['desc'])
        descs_cat = np.vstack(sp_desc['desc'].values)
        if (descs_cat.shape[1] == sp_desc.shape[0]):
            descs_cat = descs_cat.T

        if (not pca):

            self.trans_transform = myLFDA(n_components=n_dims, k=k)
            self.trans_transform.fit(descs_cat,
                                     pm['proba'].values,
                                     thresh,
                                     n_samps,
                                     clean_zeros=True)
            self.logger.info(
                'Fitting LFDA (dims,k,n_samps): ({}, {}, {})'.format(
                    n_dims, k, n_samps))

        else:
            self.logger.info(
                'Fitting PCA with {} components'.format(n_comps_pca))
            self.trans_transform = PCA(n_components=n_comps_pca, whiten=True)
            self.trans_transform.fit(descs_cat)
