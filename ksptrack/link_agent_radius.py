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
from siamese_sp.siamese import Siamese
import torch


class LinkAgentRadius:
    def __init__(self,
                csv_path,
                labels,
                sp_desc,
                thr_entrance=0.5,
                mask_path=None,
                model_path=None,
                sigma=0.07,
                cuda=False,
                entrance_radius=None):

        # assert((entrance_radius is not None) or (mask_path is not None) and (model_path is None)), 'entrance_radius, mask_path, and model_path cannot be all None'

        self.entrance_radius = entrance_radius
        self.labels = labels
        self.shape = self.labels[..., 0].shape
        self.mask_path = mask_path
        self.model_path = model_path
        self.cuda = cuda
        self.sigma = sigma
        self.trans_transform = None
        self.thr_clip = 0.05
        self.thr_entrance = thr_entrance
        self.labels = labels

        self.locs = csv.readCsv(csv_path, as_pandas=True)
        self.mode = 'radius'

        if (mask_path is not None):
            self.mask_paths = sorted(glob.glob(pjoin(mask_path, '*.png')))
            self.mask_paths += sorted(glob.glob(pjoin(mask_path, '*.jpg')))
            self.masks = {
                f: io.imread(p, as_gray=True) / 255
                for f, p in zip(self.locs['frame'], self.mask_paths)
            }
            self.mode = 'mask'

        if (model_path is not None):
            assert(sp_desc is not None), 'when using model, provide sp_desc'
            assert(mask_path is not None), 'when using model, provide mask_path'
            print('Loading model {}'.format(model_path))
            self.model = Siamese(balanced=False)
            cp = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(cp)
            self.mode = 'model'


    def get_i_j(self, loc):
        i, j = csv.coord2Pixel(loc['x'], loc['y'], self.shape[1],
                                self.shape[0])
        return i, j

    def make_radius_mask(self, frame):
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

    def get_all_entrance_sps(self, sp_desc_df):

        sps = []

        for f in range(self.labels.shape[-1]):
            mask = self.make_entrance_mask(f)
            labels_ = np.unique(self.labels[mask, f])
            descs_ = sp_desc_df[sp_desc_df['frame'] == f]
            descs_ = descs_.loc[descs_['label'].isin(labels_)]
            sps += [(f, row['label']) for _, row in descs_.iterrows()
                    if (mask[self.get_i_j(row)])]

        return sps

    def make_entrance_mask(self, frame):
        mask = np.zeros(self.labels[..., 0].shape).astype(bool)
        if ((self.mode == 'mask') or (self.mode == 'model')):
            if (frame in self.masks.keys()):
                mask = self.masks[frame] > self.thr_entrance
        else:
            mask = self.make_radius_mask(frame)

        return mask

    def is_entrance(self, frame, label, mask=None):
        """
        """

        if (mask is None):
            mask = self.make_entrance_mask(frame)

        mask = mask[self.labels[..., frame] == label]
        return np.mean(mask) > self.thr_entrance

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
            if (tl.get_in_frame() in self.masks.keys()):
                label_mask = labels[..., tl.get_in_frame()] == tl.get_in_label(
                )

                proba = np.mean(self.masks[tl.get_in_frame()][label_mask])

                proba = np.clip(proba,
                                a_min=self.thr_clip,
                                a_max=1 - self.thr_clip)
                return proba
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2, sp_desc):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        if(self.mode == 'model'):
            d1 = sp_desc.loc[(sp_desc['frame'] == frame_1) &
                            (sp_desc['label'] == label_1), 'desc'].values[0][None, ...]
            d2 = sp_desc.loc[(sp_desc['frame'] == frame_2) &
                            (sp_desc['label'] == label_2), 'desc'].values[0][None, ...]
            x = torch.tensor([d1, d2]).to(self.model)
            proba = self.model.calc_probas(x)
        else:
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
        proba = np.exp((-dist**2) / (2 * self.sigma**2))
        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)

        return proba


    def make_trans_transform(self,
                                sp_desc,
                                pm,
                                threshs,
                                n_samps,
                                n_dims,
                                k,
                                embedding_type='weighted',
                                pca=False,
                                n_comps_pca=3):

        # descs_cat = utls.concat_arr(sp_desc['desc'])
        descs_cat = np.vstack(sp_desc['desc'].values)
        if (descs_cat.shape[1] == sp_desc.shape[0]):
            descs_cat = descs_cat.T

        if (not pca):

            self.trans_transform = myLFDA(n_components=n_dims,
                                            k=k,
                                            embedding_type=embedding_type)
            probas = pm['proba'].values
            if((probas < threshs[0]).sum() < n_samps):
                sorted_probas = np.sort(probas)
                threshs[0] = sorted_probas[n_samps]
                warnings.warn('Not enough negatives. Setting thr to {}'.format(threshs[0]))
            if((probas > threshs[1]).sum() < n_samps):
                sorted_probas = np.sort(probas)[::-1]
                threshs[1] = sorted_probas[n_samps]
                warnings.warn('Not enough positives. Setting thr to {}'.format(threshs[1]))
            self.trans_transform.fit(descs_cat, pm['proba'].values, threshs,
                                        n_samps)

        else:
            self.trans_transform = PCA(n_components=n_comps_pca, whiten=False)
            self.trans_transform.fit(descs_cat)
