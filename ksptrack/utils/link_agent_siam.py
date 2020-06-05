import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from ksptrack.siamese import utils as utls
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.clustering import get_features
from ksptrack.siamese.loader import Loader, StackLoader
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_gmm import make_clusters


def make_cluster_maps(model, dl, device):

    batch_to_device = lambda batch: {
        k: v.to(device) if (isinstance(v, torch.Tensor)) else v
        for k, v in batch.items()
    }

    maps = dict()

    pbar = tqdm.tqdm(total=len(dl))
    for i, data in enumerate(dl):
        data = batch_to_device(data)
        with torch.no_grad():
            res = model(data)

        start = 0
        for i, f in enumerate(data['frame_idx']):
            end = start + torch.unique(data['labels'][i]).numel()
            preds = res['clusters'][start:end].detach().cpu().numpy()
            labels = data['labels'][i].squeeze().cpu().numpy()
            maps[f] = make_clusters(labels, preds)
        pbar.update(1)
    pbar.close()

    maps = [maps[k] for k in sorted(maps.keys())]

    return np.stack(maps)


class LinkAgentSiam(LinkAgentRadius):
    def __init__(self,
                 csv_path,
                 data_path,
                 model_path,
                 model_clst_path,
                 embedded_dims,
                 n_clusters,
                 entrance_radius=0.1,
                 cuda=False):

        super().__init__(csv_path, data_path, entrance_radius=entrance_radius)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.model = Siamese(embedded_dims=embedded_dims,
                             cluster_number=n_clusters,
                             backbone='unet')
        print('loading checkpoint {}'.format(model_path))
        state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

        self.model_clst = Siamese(embedded_dims=embedded_dims,
                                  cluster_number=n_clusters,
                                  backbone='unet')
        print('loading checkpoint {}'.format(model_clst_path))
        state_dict = torch.load(model_clst_path,
                                map_location=lambda storage, loc: storage)

        self.model_clst.load_state_dict(state_dict, strict=False)
        self.model_clst.to(self.device)

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = StackLoader(data_path,
                                normalization='rescale',
                                depth=2,
                                resize_shape=512)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.sigmoid = torch.nn.Sigmoid()
        self.prepare_all()
        self.cs = torch.nn.CosineSimilarity(dim=0)

    def prepare_all(self, all_edges_nn=None, feat_field='pooled_feats'):
        print('preparing features for linkAgent')
        # form initial cluster centres
        self.obj_preds = dict()
        self.feats_ml = dict()
        self.feats = dict()
        self.assignments = dict()

        edges_list = utls.make_edges_ccl(self.model_clst,
                                         self.dl,
                                         self.device,
                                         fully_connected=True,
                                         return_signed=True)

        self.model.eval()

        print('getting features')
        pbar = tqdm.tqdm(total=len(self.dl))
        for index, data in enumerate(self.dl):
            data = utls.batch_to_device(data, self.device)
            edges_ = edges_list[data['frame_idx'][0]].edge_index

            with torch.no_grad():
                res = self.model(data, edges_nn=edges_.to(self.device))

            start = 0
            for i, f in enumerate(data['frame_idx']):
                end = start + torch.unique(data['labels'][i]).numel()
                self.obj_preds[f] = self.sigmoid(
                    res['rho_hat_pooled'][start:end]).detach().cpu().numpy()
                self.feats_ml[f] = res['siam_feats'][start:end]
                self.assignments[f] = res['clusters'][start:end].argmax(
                    dim=1).detach().cpu().numpy()
                self.feats[f] = res['proj_pooled_feats'][start:end].detach(
                ).cpu().numpy().squeeze()
                start += end

            pbar.update(1)
        pbar.close()

        self.obj_preds = [
            self.obj_preds[k] for k in sorted(self.obj_preds.keys())
        ]
        self.feats_ml = [
            self.feats_ml[k] for k in sorted(self.feats_ml.keys())
        ]
        self.feats = [self.feats[k] for k in sorted(self.feats.keys())]
        self.assignments = [
            self.assignments[k] for k in sorted(self.assignments.keys())
        ]

        self.model.train()

    def make_cluster_maps(self):
        return make_cluster_maps(self.model, self.dl, self.device)

    def get_all_entrance_sps(self, *args):

        labels_pos = dict()
        n_labels = dict()
        for s in self.dl:
            for i, f in enumerate(s['frame_idx']):
                labels_pos[f] = s['labels_clicked'][i]
                n_labels[f] = torch.unique(s['labels'][i]).numel()

        labels_pos_bool = []
        for f in sorted(n_labels.keys()):
            labels_pos_ = np.zeros(n_labels[f]).astype(bool)
            labels_pos_[labels_pos[f]] = True
            labels_pos_bool.append(labels_pos_)

        return np.concatenate(labels_pos_bool)

    def get_proba(self, f0, l0, f1, l1, *args):

        f0 = self.feats_ml[f0][l0]
        f1 = self.feats_ml[f1][l1]

        p = self.cs(f0, f1)
        p = p.detach().cpu().numpy()

        return p
