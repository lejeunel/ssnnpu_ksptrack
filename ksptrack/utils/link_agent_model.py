import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from ksptrack.utils import csv_utils as csv
from siamese_sp.modeling.dec import DEC
from siamese_sp.loader import Loader
import torch
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.my_augmenters import rescale_augmenter, Normalize
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import tqdm
import pandas as pd


class LinkAgentModel(LinkAgentRadius):
    def __init__(self,
                 csv_path,
                 data_path,
                 model_path,
                 thr_entrance=0.5,
                 entrance_radius=None,
                 cuda=False):

        super().__init__(csv_path, data_path,
                         thr_entrance=thr_entrance,
                         entrance_radius=entrance_radius)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path


        print('Loading model {}'.format(model_path))
        self.model = DEC(balanced=False)
        print('loading checkpoint {}'.format(model_path))
        state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def prepare_feats(self):
        print('preparing features for linkAgentModel')

        batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        transf = iaa.Sequential([
            rescale_augmenter,
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        dl = Loader(
            self.data_path,
            normalization=transf)
        dl = DataLoader(dl,
                        collate_fn=dl.collate_fn)

        self.feats = []

        import pdb; pdb.set_trace() ## DEBUG ##
        pbar = tqdm.tqdm(total=len(dl))
        for i, data in enumerate(dl):
            data = batch_to_device(data)
            with torch.no_grad():
                res = self.model.autoencoder
                feats_ = res['feats'].detach().cpu().numpy()
                labels_ = self.labels[..., i]
                self.feats += [(data['frame_idx'],
                                l,
                                f)
                               for l, f in zip(np.unique(labels_),
                                               feats_)]
            pbar.update(1)
        pbar.close()

        self.feats = pd.DataFrame(
            self.feats,
            columns=['frame', 'label', 'desc'])

    def make_siam_tensor(self, tl, tl_loc):

        f1 = tl.get_out_frame()
        l1 = tl.get_out_label()
        f2 = tl_loc.get_in_frame()
        l2 = tl_loc.get_in_label()

        feat1 = self.sp_desc.loc[(self.sp_desc['frame'] == f1) &
                         (self.sp_desc['label'] == l1), 'desc'].values[0][None, ...]
        feat1 = torch.tensor(feat1).to(self.device)
        feat2 = self.sp_desc.loc[(self.sp_desc['frame'] == f2) &
                         (self.sp_desc['label'] == l2), 'desc'].values[0][None, ...]
        feat2 = torch.tensor(feat2).to(self.device)

        X = torch.stack((feat1, feat2))

        return X

    def get_proba_inter_frame(self, t1, t2, *args):

        X = self.make_siam_tensor(t1, t2)

        proba = self.model.calc_probas(X).detach().cpu().numpy()

        proba = np.clip(proba,
                        a_min=self.thr_clip,
                        a_max=1 - self.thr_clip)
        return proba

    def get_proba_entrance(self, tl, tl_loc, *args):

        X = self.make_siam_tensor(tl, tl_loc)

        proba = self.model.calc_probas(X).detach().cpu().numpy()

        proba = np.clip(proba,
                        a_min=self.thr_clip,
                        a_max=1 - self.thr_clip)
        return proba

