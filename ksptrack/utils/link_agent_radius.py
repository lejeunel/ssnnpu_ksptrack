import numpy as np
from skimage.draw import disk
from ksptrack.utils.lfda import myLFDA
from ksptrack.utils.link_agent import LinkAgent
from ksptrack.utils import my_utils as utls


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
            if (f in self.locs['frame'].to_numpy()):
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
            rr, cc = disk((loc[0], loc[1]),
                          self.shape[0] * self.entrance_radius,
                          shape=self.shape)
            mask[rr, cc] = True
        return mask

    def get_proba_entrance(self, sp, sp_desc):

        label_user = self.get_closest_label(sp)

        if (label_user is not None):

            return self.get_proba(sp['frame'], label_user, sp['frame'],
                                  sp['label'], sp_desc)
        else:
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2, sp_desc):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(frame_1, label_1, frame_2, label_2, sp_desc)

        return proba

    def get_distance(self, sp_desc, f1, l1, f2, l2, p=2):
        d1 = sp_desc.loc[(sp_desc['frame'] == f1) & (sp_desc['label'] == l1),
                         'desc'].values[0][None, ...]
        d2 = sp_desc.loc[(sp_desc['frame'] == f2) & (sp_desc['label'] == l2),
                         'desc'].values[0][None, ...]
        d1 = self.trans_transform.transform(d1)
        d2 = self.trans_transform.transform(d2)

        dist = np.linalg.norm(d1 - d2, ord=p)
        return dist

    def get_proba(self, f1, l1, f2, l2, sp_desc):

        dist = self.get_distance(sp_desc, f1, l1, f2, l2)
        proba = np.exp((-dist**2) * self.sigma)
        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)

        return proba

    def update_trans_transform(self,
                               features,
                               probas,
                               threshs,
                               n_samps,
                               n_dims,
                               k,
                               embedding_type='orthonormalized'):

        # threshs = utls.check_thrs(threshs, probas, n_samps)

        # X, y = utls.sample_features(features, probas, threshs, n_samps)

        self.trans_transform = myLFDA(n_components=n_dims,
                                      n_components_prestage=n_dims,
                                      k=None,
                                      embedding_type=embedding_type)
        self.trans_transform.fit(features, probas, threshs, n_samps)
