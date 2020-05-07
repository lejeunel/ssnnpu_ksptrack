import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from ksptrack.siamese import utils as utls
from ksptrack.siamese.clustering import get_features
from ksptrack.siamese.loader import Loader
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_gmm import make_cluster_maps


class LinkAgentSiam(LinkAgentRadius):
    def __init__(self,
                 csv_path,
                 data_path,
                 model,
                 entrance_radius=0.1,
                 cuda=False):

        super().__init__(csv_path,
                         data_path,
                         model,
                         entrance_radius=entrance_radius)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.model = model
        self.model.to(self.device)

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = Loader(data_path,
                           normalization='rescale',
                           resize_shape=512)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.sigmoid = torch.nn.Sigmoid()
        self.prepare_all()

    def prepare_all(self, all_edges_nn=None, feat_field='pooled_feats'):
        print('preparing features for linkAgent')
        # form initial cluster centres
        self.obj_preds = []
        self.feats_csml = []
        self.feats = []
        self.labels_pos = []
        self.assignments = []

        edges_list = utls.make_edges_ccl(self.model, self.dl, self.device)

        print('getting features')
        pbar = tqdm.tqdm(total=len(self.dl))
        for index, data in enumerate(self.dl):
            data = utls.batch_to_device(data, self.device)
            edges_ = torch.cat(
                [edges_list[f].edge_index for f in data['frame_idx']], dim=1)
            with torch.no_grad():

                res = self.model(data, edges_nn=edges_.to(self.device))

            clicked_labels = [
                item for sublist in data['labels_clicked'] for item in sublist
            ]

            to_add = np.zeros(
                np.unique(data['labels'].cpu().numpy()).shape[0]).astype(bool)
            to_add[clicked_labels] = True
            self.labels_pos.append(to_add)

            self.obj_preds.append(self.sigmoid(res['rho_hat']).cpu().numpy())
            self.feats_csml.append(res['siam_feats'])
            self.feats.append(
                res['pooled_feats'].detach().cpu().numpy().squeeze())
            self.assignments.append(
                res['clusters'].argmax(dim=1).cpu().numpy())

            pbar.update(1)
        pbar.close()

        self.model.train()

    def make_cluster_maps(self):
        return make_cluster_maps(self.model, self.dl, self.device)

    def get_all_entrance_sps(self, *args):

        return np.concatenate(self.labels_pos)

    def get_proba(self, f0, l0, f1, l1, *args):

        f0 = self.feats_csml[f0][l0].detach().cpu().numpy()
        f1 = self.feats_csml[f1][l1].detach().cpu().numpy()

        p = np.dot(f0, f1)

        return p
