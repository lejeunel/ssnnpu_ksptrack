from torch import nn
import numpy as np
import torch
import networkx as nx
import pandas as pd

def graph_to_pd(g, adjacent=True):
    df = nx.to_pandas_edgelist(g, source='n0', target='n1')

    if(adjacent):
        df = df.loc[df['adjacent']]

    return df

def sample_edges(g, n_edges, adjacent=True):

    df = graph_to_pd(g, adjacent)

    df_pos = df.loc[df['clust_sim'] == 1]
    df_neg = df.loc[df['clust_sim'] == 0]

    pos_dists = df_pos['feat_dist']
    probas_pos = 1 / (pos_dists + 1e-6)
    probas_pos = probas_pos / probas_pos.sum()
                  
    pos_idx = np.random.choice(df_pos.index,
                               size=n_edges,
                               p=probas_pos,
                               replace=True)
                # p=df_pos['avg_conf'] / df_pos['avg_conf'].sum())
    neg_dists = df_neg['feat_dist']
    probas_neg = neg_dists
    probas_neg = probas_neg / probas_neg.sum()
    neg_idx = np.random.choice(df_neg.index,
                               size=n_edges,
                               p=probas_neg,
                               replace=True)
                # p=df_neg['avg_conf'] / df_neg['avg_conf'].sum())

    edges_pw = pd.concat((df_pos.ix[pos_idx], df_neg.ix[neg_idx]))
    edges_pw = edges_pw[['n0', 'n1', 'clust_sim']]

    return edges_pw

def set_cluster_assignments_to_graph(graph, clusters):
    node_list = [(n,
                    dict(cluster=torch.argmax(c).cpu().item()))
                    for n, c in zip(graph.nodes(), clusters)]
    graph.add_nodes_from(node_list)
    edge_list = [
        (n0, n1,
            dict(clust_sim=0 if
                 (graph.nodes[n0]['cluster'] != graph.nodes[n1]['cluster']) else 1))
        for n0, n1 in graph.edges()
    ]
    graph.add_edges_from(edge_list)

    return graph


def get_featured_edges(edges_nn, clusters, feats, sample_n_edges=None):

    X = []
    Y = []

    edge_distribs = torch.stack((clusters[edges_nn[:, 0]],
                                 clusters[edges_nn[:, 1]]))
    bat_coeff = torch.prod(edge_distribs, dim=0).sqrt().sum(dim=1)

    clst_assign_nodes = torch.argmax(clusters, dim=1)

    pos_edges = clst_assign_nodes[edges_nn[:, 0]] == clst_assign_nodes[edges_nn[:, 1]]
    pos_edges_idx = pos_edges.nonzero().squeeze()
    pos_bat_coeff = bat_coeff[pos_edges_idx].squeeze()
    neg_edges = clst_assign_nodes[edges_nn[:, 0]] != clst_assign_nodes[edges_nn[:, 1]]
    neg_edges_idx = neg_edges.nonzero().squeeze()
    neg_bat_coeff = 1 - bat_coeff[neg_edges_idx].squeeze()

    if(sample_n_edges is not None):
        smpld_pos = torch.multinomial(pos_bat_coeff, sample_n_edges).squeeze()
        smpld_neg = torch.multinomial(neg_bat_coeff, sample_n_edges).squeeze()
        Y = torch.cat((torch.ones(sample_n_edges), torch.zeros(sample_n_edges)), dim=0)
    else:
        smpld_pos = pos_edges_idx
        smpld_neg = neg_edges_idx
        Y = torch.cat((torch.ones(pos_edges_idx.numel()),
                       torch.zeros(neg_edges_idx.numel())), dim=0)

    X_pos = torch.stack((feats[edges_nn[pos_edges_idx[smpld_pos], 0]],
                         feats[edges_nn[pos_edges_idx[smpld_pos], 1]]))
    X_neg = torch.stack((feats[edges_nn[neg_edges_idx[smpld_neg], 0]],
                         feats[edges_nn[neg_edges_idx[smpld_neg], 1]]))

    X = torch.cat((X_pos, X_neg), dim=1)

    Y = Y.to(X)

    return X, Y


class PairwiseConstrainedClustering(nn.Module):
    def __init__(self, lambda_, n_edges):
        super(PairwiseConstrainedClustering, self).__init__()
        self.lambda_ = lambda_
        self.n_edges = n_edges

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')

    def forward(self, edges_nn, feats, clusters, targets):

        X, Y = get_featured_edges(edges_nn, clusters,
                                  feats,
                                  self.n_edges)
        loss_clust = self.criterion_clust(clusters.log(),
                                          targets) / clusters.shape[0]
        Y = Y[0, :]

        loss_pw_ml = torch.norm(X[:, 0, Y == 1, ...] - X[:, 1, Y == 1, ...], p=2, dim=-1)**2
        loss_pw_cl = torch.norm(X[:, 0, Y == 0, ...] - X[:, 1, Y == 0, ...], p=2, dim=-1)**2
        loss_pw = torch.mean(loss_pw_ml - loss_pw_cl)

        loss = loss_clust + self.lambda_ * loss_pw

        return loss

