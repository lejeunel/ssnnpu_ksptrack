from torch import nn
import numpy as np
import torch
import networkx as nx


def get_featured_edges(edges_nn,
                       clusters,
                       feats,
                       sample_edges_ratio=None,
                       normalize_bc=False):

    edge_distribs = torch.stack(
        (clusters[edges_nn[:, 0]], clusters[edges_nn[:, 1]]))
    bat_coeff = torch.prod(edge_distribs, dim=0).sqrt().sum(dim=1)

    clst_assign_nodes = torch.argmax(clusters, dim=1)

    pos_edges = clst_assign_nodes[edges_nn[:, 0]] == clst_assign_nodes[
        edges_nn[:, 1]]
    pos_edges_idx = pos_edges.nonzero().squeeze()
    pos_bat_coeff = bat_coeff[pos_edges_idx].squeeze()

    neg_edges = clst_assign_nodes[edges_nn[:, 0]] != clst_assign_nodes[
        edges_nn[:, 1]]
    neg_edges_idx = neg_edges.nonzero().squeeze()
    neg_bat_coeff = 1 - bat_coeff[neg_edges_idx].squeeze()

    # normalize edge sampling proba by cluster frequency of nodes
    clst_freqs = torch.tensor([
        (clst_assign_nodes == c).sum() / float(clst_assign_nodes.numel())
        for c in range(clusters.shape[1])
    ])

    neg_sampling_probs = neg_bat_coeff
    pos_sampling_probs = pos_bat_coeff

    if (normalize_bc):

        pos_norm_factor = clst_freqs[clst_assign_nodes[
            edges_nn[pos_edges_idx, 0]]]
        pos_norm_factor[pos_norm_factor == 0] = 1

        neg_norm_factor = 0.5 * (clst_freqs[clst_assign_nodes[edges_nn[neg_edges_idx, 0]]] + \
                                clst_freqs[clst_assign_nodes[edges_nn[neg_edges_idx, 1]]])
        neg_norm_factor[neg_norm_factor == 0] = 1
        neg_sampling_probs = neg_bat_coeff / neg_norm_factor.to(feats)
        pos_sampling_probs = pos_bat_coeff / pos_norm_factor.to(feats)

    if (sample_edges_ratio is not None):
        sample_n_edges = int(edges_nn.shape[0] * sample_edges_ratio)
        smpld_pos = torch.multinomial(pos_sampling_probs,
                                      sample_n_edges).squeeze()
        smpld_neg = torch.multinomial(neg_sampling_probs,
                                      sample_n_edges).squeeze()
        Y = torch.cat(
            (torch.ones(sample_n_edges), torch.zeros(sample_n_edges)), dim=0)
    else:
        smpld_pos = pos_edges_idx
        smpld_neg = neg_edges_idx
        Y = torch.cat((torch.ones(
            pos_edges_idx.numel()), torch.zeros(neg_edges_idx.numel())),
                      dim=0)

    X_pos = torch.stack((feats[edges_nn[pos_edges_idx[smpld_pos], 0]],
                         feats[edges_nn[pos_edges_idx[smpld_pos], 1]]))
    X_neg = torch.stack((feats[edges_nn[neg_edges_idx[smpld_neg], 0]],
                         feats[edges_nn[neg_edges_idx[smpld_neg], 1]]))

    X = torch.cat((X_pos, X_neg), dim=1)

    Y = Y.to(X)

    return X, Y


def get_edges_pseudo_clique(data, edges_nn, clusters):

    must_link_edges = []

    all_assigned_clst = torch.argmax(clusters, dim=1)
    max_node = 0
    n_nodes = [g.number_of_nodes() for g in data['graph']]
    for labels, n_nodes_ in zip(data['label_keypoints'],
                                n_nodes):
        for l in labels:
            l += max_node
            assigned_clst = torch.argmax(clusters[l]).item()
            # keep edges such that both nodes are assigned to assigned_clst
            edges_mask = all_assigned_clst[edges_nn[:, 0]] == assigned_clst
            edges_mask *= all_assigned_clst[edges_nn[:, 1]] == assigned_clst
            edges_ = edges_nn[edges_mask, :]
            must_link_edges.append(edges_)
        max_node += n_nodes_ + 1

    if(len(must_link_edges) > 0):
        return torch.cat(must_link_edges, dim=0)
    else:
        return None


class LocationPairwiseLoss(nn.Module):
    def __init__(self):
        super(LocationPairwiseLoss, self).__init__()

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')

    def forward(self, data, edges_nn, feats, clusters, targets):

        loss_clust = self.criterion_clust(clusters.log(),
                                          targets) / clusters.shape[0]

        must_link_edges = get_edges_pseudo_clique(data, edges_nn, clusters)
        if(must_link_edges is not None):
            X_ml = torch.stack((feats[must_link_edges[:, 0]],
                                feats[must_link_edges[:, 1]]))

            loss_pw_ml = torch.norm(X_ml[0, ...] - X_ml[1, ...],
                                    p=2,
                                    dim=-1)**2
            loss_pw = torch.mean(loss_pw_ml)
        else:
            loss_pw = torch.tensor(0.).to(feats)

        return {'loss_clust': loss_clust, 'loss_pw': loss_pw}


class PairwiseConstrainedClustering(nn.Module):
    def __init__(self, edges_ratio):
        super(PairwiseConstrainedClustering, self).__init__()
        self.edges_ratio = edges_ratio

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')

    def forward(self, edges_nn, feats, clusters, targets, hard_pw_pos=None):

        X, Y = get_featured_edges(edges_nn, clusters, feats, self.edges_ratio)

        if (hard_pw_pos is not None):
            X_hard = torch.stack(
                (feats[hard_pw_pos[:, 0]], feats[hard_pw_pos[:, 1]])).to(X)
            Y_hard = torch.ones(hard_pw_pos.shape[0]).to(Y)

            X = torch.cat((X_hard, X), dim=1)
            Y = torch.cat((Y_hard, Y))

        loss_clust = self.criterion_clust(clusters.log(),
                                          targets) / clusters.shape[0]

        loss_pw_ml = torch.norm(X[0, Y == 1, ...] - X[1, Y == 1, ...],
                                p=2,
                                dim=-1)**2
        loss_pw_cl = torch.norm(X[0, Y == 0, ...] - X[1, Y == 0, ...],
                                p=2,
                                dim=-1)**2
        loss_pw = torch.mean(loss_pw_ml)
        # loss_pw -= - torch.mean(loss_pw_cl)

        return {'loss_clust': loss_clust, 'loss_pw': loss_pw}
