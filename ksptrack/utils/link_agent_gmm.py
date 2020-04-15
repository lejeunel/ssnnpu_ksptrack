import numpy as np
from ksptrack.siamese.clustering import get_features
import torch
from ksptrack.utils.link_agent_radius import LinkAgentRadius
from torch.utils.data import DataLoader
import tqdm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from ksptrack.siamese.loader import Loader


class LinkAgentGMM(LinkAgentRadius):
    def __init__(self,
                 csv_path,
                 data_path,
                 model,
                 entrance_radius=0.1,
                 cuda=False):

        super().__init__(csv_path, data_path, entrance_radius=entrance_radius)

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        self.dset = Loader(data_path,
                           normalization='rescale',
                           resize_shape=512)

        self.dl = DataLoader(self.dset, collate_fn=self.dset.collate_fn)

        self.prepare_feats()

        self.fit_gmm()

    def prepare_feats(self, all_edges_nn=None, feat_field='pooled_feats'):
        print('preparing features for linkAgentGMM')

        self.feats, self.labels_pos, self.assignments, self.obj_preds = get_features(
            self.model,
            self.dl,
            self.device,
            return_assign=True,
            feat_field=feat_field,
            return_obj_preds=True)

    def get_all_entrance_sps(self, *args):

        return np.concatenate(self.labels_pos)

    def fit_gmm(self):
        print('fitting GMM...')
        centroids = self.model.dec.assignment.cluster_centers.detach().cpu(
        ).numpy()
        L = self.model.dec.transform.weight.detach().cpu().numpy()
        feats_ = np.dot(np.concatenate(self.feats, axis=0), L.T)
        assign_ = self.assignments

        n_clusters = self.model.dec.cluster_number
        weights = np.array([
            np.sum(assign_ == c) / assign_.shape[0]
            for c in np.arange(n_clusters)
        ])

        # correct for collapsed clusters
        nz = weights > 0
        weights = weights[nz]
        n_clusters = nz.sum()
        centroids = centroids[nz]
        try:
            covs = [
                np.cov(feats_[assign_ == c, :].T)
                for c in np.arange(n_clusters)
            ]
            precs = np.stack([np.linalg.inv(s) for s in covs])
            self.gmm = GaussianMixture(n_components=n_clusters,
                                       means_init=centroids,
                                       weights_init=weights,
                                       reg_covar=1e-3,
                                       covariance_type='diag')
            # precisions_init=precs)
            self.gmm.fit(feats_)
        except:
            print('something wrong happened with covariance initialization...')
            self.gmm = GaussianMixture(n_components=n_clusters,
                                       means_init=centroids,
                                       weights_init=weights,
                                       reg_covar=1e-3,
                                       covariance_type='diag')
            self.gmm.fit(feats_)
        self.probas = [
            self.gmm.predict_proba(np.dot(f, L.T)) for f in self.feats
        ]

    def make_clusters(self, labels, predictions):
        cmap = plt.get_cmap('viridis')
        shape = labels.shape
        n_clusters = predictions.shape[1]
        mapping = np.array([
            (np.array(cmap(c / n_clusters)[:3]) * 255).astype(np.uint8)
            for c in np.argmax(predictions, axis=1)
        ])
        mapping = np.concatenate((np.unique(labels)[..., None], mapping),
                                 axis=1)

        _, ind = np.unique(labels, return_inverse=True)
        clusters_colorized = mapping[ind, 1:].reshape((shape[0], shape[1], 3))
        clusters_colorized = clusters_colorized.astype(np.uint8)

        return clusters_colorized

    def make_cluster_maps(self):
        batch_to_device = lambda batch: {
            k: v.to(self.device) if (isinstance(v, torch.Tensor)) else v
            for k, v in batch.items()
        }

        maps = []

        pbar = tqdm.tqdm(total=len(self.dl))
        for i, data in enumerate(self.dl):
            data = batch_to_device(data)
            with torch.no_grad():
                res = self.model(data)
                preds = res['clusters'].detach().cpu().numpy()
                labels = data['labels'].squeeze().cpu().numpy()
            maps.append(self.make_clusters(labels, preds))
            pbar.update(1)
        pbar.close()

        return np.stack(maps)

    def get_proba(self, f0, l0, f1, l1, *args):

        proba0 = self.probas[f0][l0]
        proba1 = self.probas[f1][l1]

        proba = np.sqrt(proba0 * proba1).sum()

        proba = np.clip(proba, a_min=self.thr_clip, a_max=1 - self.thr_clip)
        return proba

    def get_proba_entrance(self, sp, *args):

        label_user = self.get_closest_label(sp)
        if (label_user is not None):
            return self.get_proba(sp['frame'], label_user, sp['frame'],
                                  sp['label'])
        else:
            return self.thr_clip

    def get_proba_inter_frame(self, tracklet1, tracklet2, *args):

        t1 = tracklet1
        t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.get_proba(frame_1, label_1, frame_2, label_2)

        return proba
