from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
import numpy as np
import itertools
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class MyKMeans:
    def __init__(self, n_clusters, use_locs, max_samples=4000):

        self.n_clusters = n_clusters
        self.max_samples = max_samples
        self.use_locs = use_locs

        self.clf = PCKMeans(n_clusters) if use_locs else KMeans(n_clusters)

    def fit_predict(self, feats, clicked_mask):

        if(self.use_locs):
            to_decimate = [
                i for i in range(feats.shape[0]) if (not clicked_mask[i])
            ]
            n_to_draw = self.max_samples - np.sum(clicked_mask)
            idx_to_draw = np.random.choice(np.array(to_decimate),
                                        size=n_to_draw,
                                        replace=True)

            feats_for_train = np.concatenate((feats[clicked_mask, :], feats[idx_to_draw, :]),
                                             axis=0)
            ml = list(itertools.combinations(np.arange(0, np.sum(clicked_mask)),
                                            2))
            self.clf.fit(whiten(feats_for_train), ml=ml)
            distances = euclidean_distances(whiten(feats),
                                            self.clf.cluster_centers_)
            self.cluster_centers = self.clf.cluster_centers_
            labels = np.argmax(distances, axis=1)

        else:
            # feats = whiten(feats)
            labels = self.clf.fit_predict(feats)
            self.cluster_centers = self.clf.cluster_centers_

        return labels, self.cluster_centers

            
