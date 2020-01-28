from torch import nn
import numpy as np
import torch
import networkx as nx
import pandas as pd


def sample_edges(g, n_edges, adjacent=True):

    df = nx.to_pandas_edgelist(g, source='n0', target='n1')

    if(adjacent):
        df = df.loc[df['adjacent']]

    df_pos = df.loc[df['clust_sim'] == 1]
    df_neg = df.loc[df['clust_sim'] == 0]

    pos_dists = df_pos['feat_dist']
    probas_pos = 1 / (pos_dists + 1e-6)
    probas_pos = probas_pos / probas_pos.sum()
                  
    pos_idx = np.random.choice(df_pos.index,
                               size=n_edges,
                               p=probas_pos,
                               replace=False)
                # p=df_pos['avg_conf'] / df_pos['avg_conf'].sum())
    neg_dists = df_neg['feat_dist']
    probas_neg = neg_dists
    probas_neg = probas_neg / probas_neg.sum()
    neg_idx = np.random.choice(df_neg.index,
                               size=n_edges,
                               p=probas_neg,
                               replace=False)
                # p=df_neg['avg_conf'] / df_neg['avg_conf'].sum())

    edges_pw = pd.concat((df_pos.ix[pos_idx], df_neg.ix[neg_idx]))
    edges_pw = edges_pw[['n0', 'n1', 'clust_sim']]

    return edges_pw


def sample_batch(graphs, clusters, targets, feats, n_edges):

    sampled_edges = []
    X = []
    Y = []

    n_labels = [g.number_of_nodes() for g in graphs]
    if (isinstance(clusters, torch.Tensor)):
        clusters = torch.split(clusters, n_labels, dim=0)
    if (isinstance(targets, torch.Tensor)):
        targets = torch.split(targets, n_labels, dim=0)

    if (isinstance(feats, torch.Tensor)):
        feats = torch.split(feats, n_labels, dim=0)

    # set cluster indices and cluster similarities to graphs
    for g, tgt, clst, feat in zip(graphs, targets, clusters, feats):
        feat_np = feat.clone().detach().cpu().numpy()
        node_list = [(n,
                      dict(cluster=torch.argmax(c).cpu().item(),
                           feat=feat_np[n],
                           confidence=torch.max(c).cpu().item()))
                     for n, c in zip(g.nodes(), clst)]
        g.add_nodes_from(node_list)
        edge_list = [
            (n0, n1,
             dict(clust_sim=0 if
                  (g.nodes[n0]['cluster'] != g.nodes[n1]['cluster']) else 1,
                  feat_dist=np.linalg.norm(g.nodes[n0]['feat'] - g.nodes[n1]['feat']),
                  avg_conf=np.mean(
                      (g.nodes[n0]['confidence'], g.nodes[n1]['confidence']))))
            for n0, n1 in g.edges()
        ]
        g.add_edges_from(edge_list)

        edges_pw = sample_edges(g, n_edges)

        X_ = torch.stack(
            [torch.stack((feat[e['n0']], feat[e['n1']]), dim=0)
             for _, e in edges_pw[['n0', 'n1']].iterrows()],
            dim=1)
        Y_ = torch.tensor(edges_pw['clust_sim'].to_numpy())

        X.append(X_)
        Y.append(Y_)
        sampled_edges.append(edges_pw)

    X = torch.stack(X)
    Y = torch.stack(Y).to(X)

    return X, Y, sampled_edges


class PairwiseConstrainedClustering(nn.Module):
    def __init__(self, lambda_, n_edges):
        super(PairwiseConstrainedClustering, self).__init__()
        self.lambda_ = lambda_
        self.n_edges = n_edges

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')

    def forward(self, graphs, feats, clusters, targets):

        X, Y, edges_pw = sample_batch(graphs, clusters, targets,
                                      feats, self.n_edges)
        loss_clust = self.criterion_clust(clusters.log(),
                                          targets) / clusters.shape[0]
        Y = Y[0, :]

        loss_pw_ml = torch.norm(X[:, 0, Y == 1, ...] - X[:, 1, Y == 1, ...], p=2, dim=-1)**2
        loss_pw_cl = torch.norm(X[:, 0, Y == 0, ...] - X[:, 1, Y == 0, ...], p=2, dim=-1)**2
        loss_pw = torch.mean(loss_pw_ml - loss_pw_cl)

        loss = loss_clust + self.lambda_ * loss_pw

        return loss, edges_pw


class SiameseLoss(nn.Module):
    def __init__(self, lambda_, n_edges, with_flow):
        super(SiameseLoss, self).__init__()
        self.lambda_ = lambda_
        self.n_edges = n_edges
        self.with_flow = with_flow

        self.criterion_clust = nn.BCEWithLogitsLoss()
        self.criterion_flow = nn.BCEWithLogitsLoss()

    def forward(self, graphs, clusters, targets, feats, fn_probas):

        X, Y, sampled_edges = sample_batch(graphs, clusters, targets,
                                           feats, self.n_edges)
        pred_simil = fn_probas(X)
        loss = self.criterion_clust(pred_simil, Y)

        if(self.with_flow):
            tgt_flow = torch.stack([[g[(n0, n1)]['hoof_inter'] for n0, n1 in g.edges()]
                                    for g in graphs])
            loss *= self.lambda_
            loss += (1 - self.lambda_) * self.criterion_flow(pred_simil, tgt_flow)
        return {'loss': loss,
                'feats': X,
                'Y': Y}
