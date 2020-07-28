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
                 embedded_dims,
                 n_clusters,
                 sigma_max=0.1,
                 siamese='gcn',
                 sp_labels_fname='sp_labels.npy',
                 cuda=False):

        super().__init__(csv_path, data_path)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path
        self.sigma_max = sigma_max

        self.model = Siamese(embedded_dims=embedded_dims,
                             cluster_number=n_clusters,
                             backbone='unet',
                             siamese=siamese)
        print('loading checkpoint {}'.format(model_path))
        state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

        self.cs_sigma = self.model.cs_sigma.sigmoid().item()

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = StackLoader(data_path,
                                normalization='rescale',
                                sp_labels_fname=sp_labels_fname,
                                depth=2,
                                resize_shape=512)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.sigmoid = torch.nn.Sigmoid()
        self.cs = torch.nn.CosineSimilarity(dim=0)
        self.prepare_all()

    def prepare_all(self, all_edges_nn=None, feat_field='pooled_feats'):
        print('preparing features for linkAgentSiam')

        self.feats, self.labels_pos, self.assignments, self.obj_preds = get_features(
            self.model,
            self.dl,
            self.device,
            return_distribs=True,
            return_obj_preds=True,
            feat_fields=[
                'pooled_feats', 'proj_pooled_feats', 'siam_feats', 'pos'
            ])

        self.model.train()

    def make_cluster_maps(self):
        return make_cluster_maps(self.model, self.dl, self.device)

    def get_all_entrance_sps(self, *args):

        labels_pos = dict()
        n_labels = dict()
        for s in self.dl:
            offsets = [
                s['labels'][i].max().int().item()
                for i in range(s['labels'].shape[0])
            ]
            for i, f in enumerate(s['frame_idx']):
                if (i > 0):
                    labels_pos[f] = s['clicked'][i] - (offsets[i - 1] + 1)
                else:
                    labels_pos[f] = s['clicked'][i]
                n_labels[f] = torch.unique(s['labels'][i]).numel()

        labels_pos_bool = []
        for f in sorted(n_labels.keys()):
            labels_pos_ = np.zeros(n_labels[f]).astype(bool)
            labels_pos_[labels_pos[f]] = True
            labels_pos_bool.append(labels_pos_)

        return np.concatenate(labels_pos_bool)

    def get_proba(self, f0, l0, f1, l1, *args):

        feat0 = torch.tensor(self.feats['siam_feats'][f0][l0])
        feat1 = torch.tensor(self.feats['siam_feats'][f1][l1])

        pos0 = torch.tensor(self.feats['pos'][f0][l0])
        pos1 = torch.tensor(self.feats['pos'][f1][l1])

        p = torch.exp(-(pos0 - pos1).norm()**2 / (
            (self.sigma_max * self.cs_sigma)**2)) * self.cs(feat0, feat1)
        p = p.detach().cpu().numpy()

        return p
