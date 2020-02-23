from torch import nn
import numpy as np
import torch


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
    for labels, n_nodes_ in zip(data['label_keypoints'], n_nodes):
        for l in labels:
            l += max_node
            # get cluster of pointed ROI
            assigned_clst = torch.argmax(clusters[l]).item()

            # get edges with both nodes on assigned cluster
            edges_mask = (all_assigned_clst[edges_nn[:, 0]] == assigned_clst)
            edges_mask *= (all_assigned_clst[edges_nn[:, 1]] == assigned_clst)

            # get edges that touch pointed ROI
            edges_mask *= (edges_nn[:, 0] == l) + (edges_nn[:, 1] == l)

            edges_ = edges_nn[edges_mask, :]
            must_link_edges.append(edges_)
        max_node += n_nodes_ + 1

    if (len(must_link_edges) > 0):
        return torch.cat(must_link_edges, dim=0)
    else:
        return None


def get_edges_pseudo_clique_probas(probas, edges_nn, thr=0.8):

    # get edges with both nodes on assigned cluster
    edges_mask = probas[edges_nn[:, 0]] >= thr
    edges_mask *= probas[edges_nn[:, 1]] >= thr
    edges = edges_nn[edges_mask, :]

    if (len(edges) == 0):
        return None

    return edges


def get_edges_on_location(data, edges_nn, clusters):

    edges = []

    max_node = 0
    n_nodes = [g.number_of_nodes() for g in data['graph']]
    for labels, n_nodes_ in zip(data['label_keypoints'], n_nodes):
        for l in labels:
            l += max_node
            # get cluster of pointed ROI
            assigned_clst = clusters[l].item()

            # get edges with one node on assigned cluster
            edges_mask = (clusters[edges_nn[:, 0]] == assigned_clst)
            edges_mask += (clusters[edges_nn[:, 1]] == assigned_clst)

            edges_ = edges_nn[edges_mask, :]
            edges.append(edges_)
        max_node += n_nodes_ + 1

    return torch.cat(edges, dim=0)


class LocationPairwiseLoss(nn.Module):
    def __init__(self):
        super(LocationPairwiseLoss, self).__init__()

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')

    def forward(self, data, edges_nn, feats, clusters, targets):

        ml_edges = get_edges_pseudo_clique(data, edges_nn, clusters)
        if (ml_edges is not None):
            feats_edges = torch.stack((feats[ml_edges[:, 0]],
                                       feats[ml_edges[:, 1]]))
            norm = torch.norm(feats_edges, dim=0)
            loss_pw = -norm**2
        else:
            loss_pw = None

        loss_clust = self.criterion_clust(
            clusters.log(), targets) / clusters.shape[0]

        return {'loss_clust': loss_clust, 'loss_pw': loss_pw}


class TripletLoss(nn.Module):
    def __init__(self, max_samples=400):
        super(TripletLoss, self).__init__()
        self.max_samples = max_samples

    def forward(self, probas, data, clusters, edges_nn):
        # probas are predicted similarity probabilities
        # Y = 0 if nodes of edges are of different cluster
        # edges_nn are corresponding edges

        # TODO: pick random point when no location!
        edges = get_edges_on_location(data, edges_nn,
                                      clusters.argmax(dim=1))

        # compute similarities
        sims = clusters[edges[:, 0]].argmax(dim=1) == clusters[edges[:, 1]].argmax(dim=1)
        edges = edges
        pos_idx = torch.nonzero(sims == 1).view(-1)
        neg_idx = torch.nonzero(sims == 0).view(-1)

        i, j = torch.meshgrid(pos_idx, neg_idx)
        i = i.flatten()
        j = j.flatten()

        # candidates are such that "left" nodes are identical (should shuffle stacks en amont)
        i_ = i[edges[i, 0] == edges[j, 0]]
        j_ = j[edges[i, 0] == edges[j, 0]]

        bc_pos = (clusters[edges[i_, 0]] * clusters[edges[i_, 1]]).sqrt().sum()
        bc_neg = (clusters[edges[j_, 0]] * clusters[edges[j_, 1]]).sqrt().sum()
        sampling_prob = 1 - (bc_pos + bc_neg) * .5

        inds = torch.multinomial(sampling_prob, self.max_samples)

        tplts_p = torch.stack((probas[i_[inds]], probas[j_[inds]]))
        loss = -torch.log(tplts_p[0, :] / (tplts_p[0, :] + tplts_p[1, :])).mean()

        return loss



if __name__ == "__main__":
    from ksptrack.siamese.utils import make_couple_graphs
    from ksptrack.models.my_augmenters import rescale_augmenter, Normalize
    from torch.utils.data import DataLoader
    from ksptrack.siamese.modeling.siamese import Siamese
    from loader import StackLoader
    from os.path import join as pjoin

    transf_normal = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    dl_stack = StackLoader(2,
                           pjoin('/home/ubelix/lejeune/data/medical-labeling',
                                 'Dataset00'),
                           normalization=transf_normal)
    dl = DataLoader(dl_stack, collate_fn=dl_stack.collate_fn, shuffle=True)

    model = Siamese(10, 10, roi_size=1, roi_scale=1, alpha=1)

    device = torch.device('cpu')
    run_path = '/home/ubelix/lejeune/runs/siamese_dec/Dataset00'
    cp_path = pjoin(run_path, 'checkpoints', 'best_dec.pth.tar')
    state_dict = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model.dec.load_state_dict(state_dict)
    L = np.load(pjoin(run_path, 'init_clusters.npz'), allow_pickle=True)['L']
    L = torch.tensor(L).float().to(device)
    criterion = TripletLoss()

    for data in dl:
        edges_nn, _, _ = make_couple_graphs(model, device, data, 0.1, L, True)
        res = model(data, edges_nn, L=L, do_assign=True)
        loss = criterion(res['probas_preds'], data, res['clusters'], edges_nn)
