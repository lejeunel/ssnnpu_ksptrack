from glob import glob
from os.path import join as pjoin

import numpy as np
import torch
from ksptrack.pu.im_utils import get_features
from ksptrack.modeling.unet import UNet
from ksptrack.utils.link_agent import LinkAgent
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from skimage.draw import disk
from torch.utils.data import DataLoader


class LinkAgentRadius(LinkAgent):
    def __init__(self,
                 csv_path,
                 data_path,
                 model_pred_path,
                 loc_prior=False,
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
        self.model_pred = UNet(in_channels=3, out_channels=1)

        if not model_pred_path.endswith('.tar'):
            model_pred_path = sorted(
                glob(pjoin(model_pred_path, 'cp_*.pth.tar')))[-1]
        print('loading checkpoint {}'.format(model_pred_path))
        state_dict = torch.load(model_pred_path,
                                map_location=lambda storage, loc: storage)

        self.model_pred.load_state_dict(state_dict)
        self.model_pred.to(self.device)
        self.model_pred.eval()

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = LocPriorDataset(data_path,
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

        self.labels_pos = res['labels_pos_mask']
        self.obj_preds = res['outs']
        self.pos = res['pos']

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
