from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
import numpy as np
import itertools
from scipy.cluster.vq import whiten
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.cluster import KMeans


class MyKMeans(PCKMeans):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def fit(self, X, **kwargs):
        X = whiten(X)
        super().fit(X, **kwargs)

    def predict(self, X, **kwargs):
        X = whiten(X)
        return super().predict(X, **kwargs)


class MyPCKMeans(PCKMeans):
    def __init__(self, n_clusters, max_samples=4000):

        super().__init__(n_clusters=n_clusters)

        self.n_clusters = n_clusters
        self.max_samples = max_samples

    def fit(self, feats, clicked_mask):

        to_decimate = [
            i for i in range(feats.shape[0]) if (not clicked_mask[i])
        ]
        n_to_draw = self.max_samples - np.sum(clicked_mask)
        idx_to_draw = np.random.choice(np.array(to_decimate),
                                       size=n_to_draw,
                                       replace=True)

        feats = np.concatenate((feats[clicked_mask, :], feats[idx_to_draw, :]),
                               axis=0)
        ml = list(itertools.combinations(np.arange(0, np.sum(clicked_mask)),
                                         2))

        feats = whiten(feats)
        super().fit(feats, ml=ml)

    def predict(self, x):
        x = whiten(x)
        x_squared_norms = row_norms(x, squared=True)
        return _labels_inertia(x,
                               sample_weight=None,
                               x_squared_norms=x_squared_norms,
                               centers=self.cluster_centers_)[0]
