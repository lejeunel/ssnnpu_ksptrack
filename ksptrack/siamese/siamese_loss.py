from torch import nn
import numpy as np
import torch


def sample_edges(g, n_edges):

    pos_edges = [(n0, n1) for n0, n1 in g.edges()
                 if (g.edges[(n0, n1)]['clust_sim'] == 1)]
    pos_edges_conf = np.array(
        [g.edges[(n0, n1)]['avg_conf'] for n0, n1 in pos_edges])
    pos_edges_conf = pos_edges_conf / pos_edges_conf.sum()

    neg_edges = [(n0, n1) for n0, n1 in g.edges()
                 if (g.edges[(n0, n1)]['clust_sim'] == 0)]
    neg_edges_conf = np.array(
        [g.edges[(n0, n1)]['avg_conf'] for n0, n1 in neg_edges])
    neg_edges_conf = neg_edges_conf / neg_edges_conf.sum()

    pos_idx = np.random.choice(len(pos_edges), size=n_edges, p=pos_edges_conf)
    pos_edges = [pos_edges[i] for i in pos_idx]

    neg_idx = np.random.choice(len(neg_edges), size=n_edges, p=neg_edges_conf)
    neg_edges = [neg_edges[i] for i in neg_idx]

    idxs = np.concatenate((pos_edges, neg_edges))
    y = np.concatenate((np.ones(n_edges), np.zeros(n_edges)))

    return idxs, y


def sample_batch(graphs, clusters, feats, n_edges):

    sampled_edges = []
    X = []

    n_labels = [g.number_of_nodes() for g in graphs]
    if (isinstance(clusters, torch.Tensor)):
        clusters = torch.split(clusters, n_labels, dim=0)

    if (isinstance(feats, torch.Tensor)):
        feats = torch.split(feats, n_labels, dim=0)

    # set cluster indices and cluster similarities to graphs
    for g, clusters, feats in zip(graphs, clusters, feats):
        node_list = [(n,
                      dict(cluster=torch.argmax(c).cpu().item(),
                           confidence=torch.max(c).cpu().item()))
                     for n, c in zip(g.nodes(), clusters)]
        g.add_nodes_from(node_list)
        edge_list = [
            (n0, n1,
             dict(clust_sim=0 if
                  (g.nodes[n0]['cluster'] != g.nodes[n1]['cluster']) else 1,
                  avg_conf=np.mean(
                      (g.nodes[n0]['confidence'], g.nodes[n1]['confidence']))))
            for n0, n1 in g.edges()
        ]
        g.add_edges_from(edge_list)

        edges = sample_edges(g, n_edges)
        sampled_edges.append(edges)

        X_ = torch.stack(
            [torch.stack((feats[n0], feats[n1]), dim=0) for n0, n1 in edges],
            dim=1)
        X.append(X_)

    X = torch.stack(X)

    return X, sampled_edges


class SiameseLoss(nn.Module):
    def __init__(self, lambda_, n_edges, with_oflow):
        super(SiameseLoss, self).__init__()
        self.lambda_ = lambda_
        self.n_edges = n_edges
        self.with_oflow = with_oflow

        self.criterion_clust = nn.BCEWithLogitsLoss()
        self.criterion_flow = nn.BCEWithLogitsLoss()

    def forward(self, graphs, clusters, feats, fn_probas):

        X, sampled_edges = sample_batch(graphs, clusters, feats, self.n_edges)
        tgt_simil = torch.stack([
            torch.cat((torch.ones(self.n_edges), torch.zeros(self.n_edges)))
            for _ in len(graphs)
        ])
        pred_simil = fn_probas(X)
        loss = self.criterion_clust(pred_simil, tgt_simil)

        if(self.with_flow):
            tgt_flow = torch.stack([[g[(n0, n1)]['hoof_inter'] for n0, n1 in g.edges()]
                                    for g in graphs])
            loss *= self.lambda_
            loss += (1 - self.lambda_) * self.criterion_flow(pred_simil, tgt_flow)
        return loss
