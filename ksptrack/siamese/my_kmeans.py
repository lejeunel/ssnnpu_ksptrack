import numpy as np
import itertools
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class MyKMeans:
    def __init__(self, n_clusters):

        self.n_clusters = n_clusters

        self.clf = KMeans(n_clusters,
                          n_init=20,
                          max_iter=500)

    def fit_predict(self, feats, weights=None):

        # feats = whiten(feats)
        labels = self.clf.fit_predict(feats)
        self.cluster_centers = self.clf.cluster_centers_

        return labels, self.cluster_centers

            
