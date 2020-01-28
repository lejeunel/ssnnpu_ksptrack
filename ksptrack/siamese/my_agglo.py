import numpy as np
import itertools
from scipy.cluster.vq import whiten
from sklearn.cluster import AgglomerativeClustering
from loader import Loader
from os.path import join as pjoin
from tqdm import tqdm
import networkx as nx


def prepare_full_rag(graphs, labels, do_inter_frame=False):

    # we relabel nodes so that labels are unique accross sequence
    max_label = 0
    relabeled = []

    full_rag = nx.Graph()
    print('making frame-by-frame RAGs')
    pbar = tqdm(total=len(graphs))
    for labs, g in zip(labels, graphs):
        # keep only adjacent nodes
        to_remove = [e for e in g.edges() if not g.edges[e]['adjacent']]
        g.remove_edges_from(to_remove)
        # relabel nodes
        mapping = {n: n+max_label for n in g.nodes()}
        max_label += max(g.nodes()) + 1
        g = nx.relabel_nodes(g, mapping)
        relabeled.append(np.vectorize(mapping.get)(labs))
        full_rag.add_edges_from(g.edges())
        pbar.update(1)
    pbar.close()

    if(do_inter_frame):
        print('making full RAG')
        pbar = tqdm(total=len(relabeled) - 1)
        for i in range(len(relabeled) - 1):
            # find overlaps between consecutive label maps
            labels0 = relabeled[i]
            labels1 = relabeled[i+1]
            concat_ = np.concatenate((labels0, labels1), axis=-1)
            concat_ = concat_.reshape((-1, 2))
            ovl = np.asarray(list(set(list(map(tuple, concat_)))))
            edges = [(n[0], n[1]) for n in ovl]
            full_rag.add_edges_from(edges)
            pbar.update(1)
        pbar.close()

    return full_rag

class MyAggloClustering:
    def __init__(self, n_clusters, linkage):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X, connectivity_matrix):

        clf = AgglomerativeClustering(n_clusters=self.n_clusters,
                                      linkage=self.linkage,
                                      connectivity=connectivity_matrix)
        # X = whiten(X)
        labels = clf.fit_predict(X)

        # compute centers for each label
        cluster_centers = []
        for l in np.unique(labels):
            X_ = np.mean(X[labels == l, :], axis=0)
            cluster_centers.append(X_)

        self.cluster_centers = np.array(cluster_centers)

        return labels, self.cluster_centers


if __name__ == "__main__":

    dl = Loader(
        pjoin('/home/ubelix/lejeune/data/medical-labeling', 'Dataset00'))
    graphs = [s['graph'] for s in dl]
    labels = [s['labels'] for s in dl]
    full_rag = prepare_full_rag(graphs, labels)
