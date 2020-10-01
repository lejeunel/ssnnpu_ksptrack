import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import os

from ksptrack.siamese import utils as utls
from ksptrack.siamese.modeling.siamese import Siamese
from ksptrack.siamese.clustering import get_features
from ksptrack.siamese.loader import Loader, StackLoader
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from ksptrack.utils.link_agent_gmm import make_clusters
from ksptrack.siamese.losses import cosine_distance_torch, pairwise_distances
import math


def gauss(x, mu, sig):
    x = -(1 / 2) * ((x - mu)**2 / sig**2)
    x = np.exp(x)
    x = x / np.sqrt(2 * math.pi * sig**2)

    return x


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
                 alpha=3,
                 sp_labels_fname='sp_labels.npy',
                 entrance_radius=0.1,
                 cuda=True):

        super().__init__(csv_path, data_path, model_path, embedded_dims,
                         n_clusters)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path
        self.alpha = alpha

        self.model = Siamese(embedded_dims=embedded_dims,
                             cluster_number=n_clusters,
                             backbone='unet')
        print('loading checkpoint {}'.format(model_path))
        state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = StackLoader(data_path,
                                depth=1,
                                normalization='rescale',
                                sp_labels_fname=sp_labels_fname,
                                resize_shape=512)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.sigmoid = torch.nn.Sigmoid()
        self.cs = torch.nn.CosineSimilarity(dim=0)

        path_ = model_path.split(os.path.sep)
        path_ = os.path.sep.join(path_[0:-2])
        clst_path = os.path.join(path_, 'init_clusters.npz')
        self.clusters = np.load(clst_path, allow_pickle=True)['preds']
        self.max_norm_samples = 5000
        # self.get_clst_stats()

    def get_clst_stats(self):

        self.sigp = 0
        self.sign = 0
        self.mup = 0
        self.mun = 0
        all_feats = np.concatenate(self.siam_feats)
        all_clst = np.concatenate(self.clusters)
        all_clst_ = all_clst.argmax(1)
        num_clst = np.unique(all_clst).size
        print('computing stats...')

        for c in np.unique(all_clst.argmax(1)):

            clst_centroids = torch.from_numpy(
                np.array([
                    np.mean(all_clst[all_clst_ == c], axis=0)
                    for c in np.unique(all_clst.argmax(1))
                ]))
            pw_cc_centroids = pairwise_distances(clst_centroids)
            # torch.randint(0, a.size(0), (1,))
            n_pos = (all_clst_ == c).sum()
            idx_pos = torch.randint(0, n_pos,
                                    (min(self.max_norm_samples, n_pos), ))
            c_nn = torch.argsort(pw_cc_centroids[c])[1].item()
            n_neg = (all_clst_ == c_nn).sum()
            idx_neg = torch.randint(0, n_neg,
                                    (min(self.max_norm_samples, n_neg), ))

            pos_feats = torch.from_numpy(all_feats[all_clst_ == c][idx_pos])
            neg_feats = torch.from_numpy(all_feats[all_clst_ == c_nn][idx_neg])

            # within cluster stats
            pw = cosine_distance_torch(pos_feats)
            pw = pw[(1 - torch.eye(idx_pos.numel())).bool()]

            self.sigp += pw.std().item() / num_clst
            self.mup += pw.mean().item() / num_clst

            # cluster to nearest neighbor cluster stats
            # print('neg. samples: {}'.format((all_clst == c_nn).sum()))
            pw = cosine_distance_torch(pos_feats, neg_feats)
            self.sign += pw.std().item() / num_clst
            self.mun += pw.mean().item() / num_clst

        print('done.')

    def get_clst_stats_cc(self):

        print('computing connected components...')
        _, nodes_list = utls.make_edges_ccl(self.clusters,
                                            self.dl,
                                            return_signed=True,
                                            return_nodes=True,
                                            fully_connected=True)
        self.sigp = {f: 0 for f in range(len(self.dl))}
        self.sign = self.sigp.copy()
        self.mup = self.sigp.copy()
        self.mun = self.sigp.copy()
        print('computing stats...')

        pbar = tqdm.tqdm(total=len(self.dl))
        for i in range(len(self.dl)):

            all_cc = nodes_list[i][1]
            num_cc = np.unique(all_cc).size
            cc_centroids = torch.from_numpy(
                np.array([
                    np.mean(self.siam_feats[i][all_cc == c], axis=0)
                    for c in torch.unique(all_cc)
                ]))
            pw_cc_centroids = pairwise_distances(cc_centroids)
            for c in np.unique(all_cc):
                # torch.randint(0, a.size(0), (1,))
                n_pos = (all_cc == c).sum()
                # idx_pos = torch.randint(0, n_pos,
                #                         (min(self.max_norm_samples, n_pos), ))
                idx_pos = torch.arange(n_pos)
                c_nn = torch.argsort(pw_cc_centroids[c])[1].item()
                n_neg = (all_cc == c_nn).sum()
                # idx_neg = torch.randint(0, n_neg,
                #                         (min(self.max_norm_samples, n_neg), ))
                idx_neg = torch.arange(n_neg)

                pos_feats = torch.from_numpy(
                    self.siam_feats[i][all_cc == c][idx_pos])
                if idx_pos.numel() == 1:
                    continue
                neg_feats = torch.from_numpy(
                    self.siam_feats[i][all_cc == c_nn][idx_neg])
                if idx_neg.numel() == 1:
                    continue

                # within cluster stats
                pw = cosine_distance_torch(pos_feats)
                pw = pw[(1 - torch.eye(idx_pos.numel())).bool()]

                self.sigp[i] += pw.std().item() / num_cc
                self.mup[i] += pw.mean().item() / num_cc

                # cluster to nearest neighbor cluster stats
                # print('neg. samples: {}'.format((all_clst == c_nn).sum()))
                pw = cosine_distance_torch(pos_feats, neg_feats)
                self.sign[i] += pw.std().item() / num_cc
                self.mun[i] += pw.mean().item() / num_cc
            pbar.update(1)
        pbar.close()

        print('done.')

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

        feat0 = torch.tensor(self.siam_feats[f0][l0])
        feat1 = torch.tensor(self.siam_feats[f1][l1])

        cos = self.cs(feat0, feat1)
        cos = cos.detach().cpu().numpy()

        theta = np.arccos(cos)

        p = 1 - theta / np.pi

        p = p**self.alpha

        return p

        # return (p + 1) / 2
        # nom = gauss(p, self.mup, self.sigp)
        # denom = gauss(p, self.mup, self.sigp)
        # denom += gauss(p, self.mun, self.sign)
        # lik = nom / (denom + 1e-8)

        # nom = gauss(p, np.mean([self.mup[f] for f in (f0, f1)]),
        #             np.mean([self.sigp[f] for f in (f0, f1)]))
        # denom = gauss(p, np.mean([self.mup[f] for f in (f0, f1)]),
        #               np.mean([self.sigp[f] for f in (f0, f1)]))
        # denom += gauss(p, np.mean([self.mun[f] for f in (f0, f1)]),
        #                np.mean([self.sign[f] for f in (f0, f1)]))
        # lik = nom / (denom + 1e-8)

        # return lik
        # return (p + 1) / 2


if __name__ == "__main__":

    data_path = '/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset00'
    csv_path = os.path.join(data_path, 'gaze-measurements', 'video1.csv')
    model_path = '/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset00/checkpoints/cp_test.pth.tar'
    embedded_dims = 15
    n_clusters = 15

    agent = LinkAgentSiam(csv_path, data_path, model_path, embedded_dims,
                          n_clusters)
