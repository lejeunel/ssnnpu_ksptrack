import numpy as np
from skimage.draw import disk
from ksptrack.utils.lfda import myLFDA
from ksptrack.utils.link_agent import LinkAgent
from ksptrack.utils import my_utils as utls
from ksptrack.pu.modeling.unet import UNet
import torch
from ksptrack.pu.loader import Loader
from torch.utils.data import DataLoader
from ksptrack.pu.im_utils import get_features


class LinkAgentRadius(LinkAgent):
    def __init__(self,
                 csv_path,
                 data_path,
                 model_pred_path,
                 model_trans_path='',
                 loc_prior=False,
                 use_coordconv=False,
                 thr_entrance=0.5,
                 sigma=0.07,
                 sp_labels_fname='sp_labels.npy',
                 in_shape=512,
                 entrance_radius=0.05,
                 cuda=True):

        super().__init__(csv_path,
                         data_path,
                         thr_entrance=thr_entrance,
                         sp_labels_fname=sp_labels_fname)

        self.entrance_radius = entrance_radius
        self.thr_entrance = thr_entrance
        self.sigma = sigma

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.loc_prior = loc_prior
        self.model_pred = UNet(in_channels=3 + (1 if loc_prior else 0),
                               out_channels=1,
                               use_coordconv=use_coordconv)
        self.model_trans = UNet(out_channels=3, skip_mode='none')

        print('loading checkpoint {}'.format(model_pred_path))
        state_dict = torch.load(model_pred_path,
                                map_location=lambda storage, loc: storage)

        self.model_pred.load_state_dict(state_dict)
        self.model_pred.to(self.device)
        self.model_pred.eval()

        # if not model_trans_path:
        #     model_trans_path = model_pred_path

        # print('loading checkpoint {}'.format(model_trans_path))
        # state_dict = torch.load(model_trans_path,
        #                         map_location=lambda storage, loc: storage)

        # self.model_trans.load_state_dict(state_dict, strict=False)
        # self.model_trans.to(self.device)
        # self.model_trans.eval()

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = Loader(data_path,
                           normalization='rescale',
                           resize_shape=in_shape,
                           sp_labels_fname=sp_labels_fname)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.prepare_feats()

    def prepare_feats(self):
        print('preparing features for linkAgent')

        res = get_features(self.model_pred,
                           self.dl,
                           self.device,
                           loc_prior=self.loc_prior)

        self.feats = res['feats']
        self.labels_pos = res['labels_pos_mask']
        self.obj_preds = res['outs']
        self.pos = res['pos']

        # res = get_features(self.model_trans, self.dl, self.device)
        # self.feats_trans = res['feats']
        # self.feats_bag = res['feats_bag']

    def get_all_entrance_sps(self, *args):

        return np.concatenate(self.labels_pos)

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

    def get_proba_entrance(self, sp):

        label_user = self.get_closest_label(sp)

        if (label_user is not None):

            return self.get_proba(sp['frame'], label_user, sp['frame'],
                                  sp['label'])
        else:
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(frame_1, label_1, frame_2, label_2)

        return proba

    def get_distance_(self, f1, l1, f2, l2, p=2):
        d1 = self.feats_trans[f1][l1][None, ...]
        d2 = self.feats_trans[f2][l2][None, ...]
        # d1 = sp_desc.loc[(sp_desc['frame'] == f1) & (sp_desc['label'] == l1),
        #                  'desc_trans'].values[0][None, ...]
        # d2 = sp_desc.loc[(sp_desc['frame'] == f2) & (sp_desc['label'] == l2),
        #                  'desc_trans'].values[0][None, ...]
        d1 = self.trans_transform.transform(d1)
        d2 = self.trans_transform.transform(d2)

        dist = np.linalg.norm(d1 - d2, ord=p)
        return dist

    def get_proba(self, f1, l1, f2, l2):

        dist = self.get_distance(f1, l1, f2, l2)
        proba = np.exp((-dist**2) * self.sigma)
        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)

        return proba

    def get_distance(self, *args, **kwargs):
        return 0

    def update_trans_transform(self,
                               threshs=[0.3, 0.7],
                               n_samps=500,
                               n_dims=15,
                               k=7,
                               embedding_type='orthonormalized'):

        pass

        # X = np.concatenate(self.feats_trans)
        # y = np.concatenate(self.obj_preds)
        # threshs = utls.check_thrs(threshs, y, n_samps)

        # try:
        #     X, y = utls.sample_features(X, y, threshs, n_samps)

        #     self.trans_transform = myLFDA(n_components=n_dims,
        #                                   n_components_prestage=n_dims,
        #                                   k=k,
        #                                   embedding_type=embedding_type)
        #     self.trans_transform.fit(X, y, threshs, n_samps)
        # except ValueError as e:
        #     print('setting all distances to 0')
        #     self.get_distance = self.dummy_get_distance
