from torch import nn
import numpy as np
import torch
from ksptrack.siamese import utils as utls
import networkx as nx
import itertools


def get_edges_probas(probas, edges_nn, thrs=[0.6, 0.8], clusters=None):

    edges = {'sim': [], 'disim': []}

    # get edges with both nodes in positive or negative segment
    edges_mask_sim = (probas[edges_nn[:, 0]] >=
                      thrs[1]) * (probas[edges_nn[:, 1]] >= thrs[1])
    edges_mask_sim += (probas[edges_nn[:, 0]] <
                       thrs[0]) * (probas[edges_nn[:, 1]] < thrs[0])
    if (clusters is not None):
        edges_mask_sim *= (clusters[edges_nn[:, 1]].argmax(
            dim=1) == clusters[edges_nn[:, 0]].argmax(dim=1))
    edges_sim = edges_nn[edges_mask_sim, :]

    sim_probas = torch.ones(edges_sim.shape[0]).to(probas).float()
    edges_sim = [edges_sim, sim_probas]
    edges['sim'] = edges_sim

    # get edges with one node > thr[1] and other < thr[0]
    edges_mask_disim_0 = probas[edges_nn[:, 0]] >= thrs[1]
    edges_mask_disim_0 *= probas[edges_nn[:, 1]] < thrs[0]
    edges_mask_disim_1 = probas[edges_nn[:, 1]] >= thrs[1]
    edges_mask_disim_1 *= probas[edges_nn[:, 0]] < thrs[0]
    edges_mask_disim = edges_mask_disim_0 + edges_mask_disim_1
    if (clusters is not None):
        edges_mask_disim *= (clusters[edges_nn[:, 1]].argmax(dim=1) !=
                             clusters[edges_nn[:, 0]].argmax(dim=1))
    edges_disim = edges_nn[edges_mask_disim, :]

    disim_probas = torch.zeros(edges_disim.shape[0]).to(probas).float()

    edges_disim = [edges_disim, disim_probas]
    edges['disim'] = edges_disim

    for k in edges.keys():
        if (len(edges[k]) == 0):
            edges[k] = torch.tensor([]).to(probas)

    return edges


def get_edges_keypoint_probas(probas, data, edges_nn, thrs=[0.6, 0.8]):

    edges = {'sim': [], 'disim': []}

    max_node = 0
    n_nodes = [g.number_of_nodes() for g in data['graph']]
    for labels, n_nodes_ in zip(data['label_keypoints'], n_nodes):
        for l in labels:
            l += max_node

            # get edges with one node on keypoint
            edges_mask = edges_nn[:, 0] == l
            edges_mask += edges_nn[:, 1] == l

            if (edges_mask.sum() > 0):
                # get edges with nodes with proba > thr
                edges_mask_sim = edges_mask * probas[edges_nn[:, 0]] >= thrs[1]
                edges_mask_sim *= probas[edges_nn[:, 1]] >= thrs[1]
                edges_sim = edges_nn[edges_mask_sim, :]
                edges['sim'].append(edges_sim)

                # get edges with nodes with proba < thr
                edges_mask_disim = edges_mask * probas[
                    edges_nn[:, 0]] <= thrs[0]
                edges_mask_disim *= probas[edges_nn[:, 1]] <= thrs[0]
                edges_disim = edges_nn[edges_mask_disim, :]
                edges['disim'].append(edges_disim)

        max_node += n_nodes_ + 1

    for k in edges.keys():
        if (len(edges[k]) == 0):
            edges[k] = torch.tensor([]).to(probas)

    return edges


class LabelPairwiseLoss(nn.Module):
    def __init__(self, thrs=[0.6, 0.8]):
        super(LabelPairwiseLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')
        self.thrs = thrs

    def forward(self, edges_nn, probas, feats, clusters=None):

        edges = get_edges_probas(probas,
                                 edges_nn,
                                 thrs=self.thrs,
                                 clusters=clusters)
        constrained_feats = dict()
        probas = dict()

        constrained_feats['sim'] = torch.stack(
            (feats[edges['sim'][0][:, 0]], feats[edges['sim'][0][:, 1]]))
        constrained_feats['disim'] = torch.stack(
            (feats[edges['disim'][0][:, 0]], feats[edges['disim'][0][:, 1]]))

        probas['sim'] = torch.exp(-torch.norm(
            constrained_feats['sim'][0] - constrained_feats['sim'][1], dim=1))
        probas['disim'] = torch.exp(-torch.norm(constrained_feats['disim'][0] -
                                                constrained_feats['disim'][1],
                                                dim=1))

        n_pos = probas['sim'].numel()
        n_neg = probas['disim'].numel()
        pos_weight = float(max((n_pos, n_neg))) / n_pos
        neg_weight = float(max((n_pos, n_neg)) / n_neg)

        weights = torch.cat((torch.ones_like(probas['sim']) * pos_weight,
                             torch.ones_like(probas['disim']) * neg_weight))
        loss = self.criterion(
            torch.cat((probas['sim'], probas['disim'])),
            torch.cat((edges['sim'][1], edges['disim'][1])).float())
        loss *= weights

        return loss.mean()


class LabelKLPairwiseLoss(nn.Module):
    def __init__(self, thr=0.5):
        super(LabelKLPairwiseLoss, self).__init__()
        self.criterion_clst = torch.nn.KLDivLoss(reduction='mean')
        self.criterion_pw = torch.nn.KLDivLoss(reduction='mean')
        self.thr = thr

    def forward(self, edges_nn, probas, clusters, targets, keypoints=None):

        edges = get_edges_probas(probas,
                                 edges_nn,
                                 thrs=[self.thr, self.thr],
                                 clusters=targets)
        # clusters=targets)

        if (keypoints is not None):
            # filter out similar edges
            mask = torch.zeros(edges['sim'][0].shape[0]).to(clusters).bool()
            for kp in keypoints:
                mask += edges['sim'][0][:, 0] == kp
                mask += edges['sim'][0][:, 1] == kp

            edges['sim'][0] = edges['sim'][0][mask, :]

        loss_pw = torch.tensor((0.)).to(clusters)

        clst_sim = torch.stack(
            (clusters[edges['sim'][0][:, 0]], clusters[edges['sim'][0][:, 1]]))
        tgt_sim = torch.stack(
            (targets[edges['sim'][0][:, 0]], targets[edges['sim'][0][:, 1]]))
        clst_disim = torch.stack((clusters[edges['disim'][0][:, 0]],
                                  clusters[edges['disim'][0][:, 1]]))
        tgt_disim = torch.stack((targets[edges['disim'][0][:, 0]],
                                 targets[edges['disim'][0][:, 1]]))

        n_pos = edges['sim'][0].shape[0]
        n_neg = edges['disim'][0].shape[0]

        if (n_pos > 0):
            loss_sim = self.criterion_pw(
                (clst_sim[0] + 1e-7).log(), tgt_sim[1]) / clst_sim[0].shape[0]
            loss_sim += self.criterion_pw(
                (clst_sim[1] + 1e-7).log(), tgt_sim[0]) / clst_sim[0].shape[0]
            loss_sim = loss_sim / 2
            pos_weight = float(max((n_pos, n_neg))) / n_pos
            # loss_pw += loss_sim * pos_weight
            loss_pw += loss_sim

        # if (n_neg > 0):
        #     loss_disim = self.criterion_pw(
        #         (clst_disim[0] + 1e-7).log(),
        #         tgt_disim[1]) / clst_disim[0].shape[0]
        #     loss_disim += self.criterion_pw(
        #         (clst_disim[1] + 1e-7).log(),
        #         tgt_disim[0]) / clst_disim[0].shape[0]
        #     neg_weight = float(max((n_pos, n_neg)) / n_neg)
        #     loss_pw -= loss_disim * neg_weight

        return loss_pw


class PairwiseContrastive(nn.Module):
    def __init__(self, margin=0.2):
        super(PairwiseContrastive, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.margin = 0.

    def forward(self, input, target):

        loss = self.bce(input, target)

        loss_pos = loss[target == 1]
        loss_pos = loss_pos[loss_pos <= 1 - self.margin].mean()

        loss_neg = loss[target == 0]
        loss_neg = loss_neg[loss_neg >= self.margin].mean()

        return (loss_neg + loss_pos) / 2


class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def forward(self, z, pos_edges, neg_edges):

        loss_1 = self.pos_embedding_loss(z, pos_edges)
        loss_2 = self.neg_embedding_loss(z, neg_edges)
        return loss_1 + loss_2


def structured_negative_sampling(edge_index):
    r"""Samples a negative sample :obj:`(k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = edge_index.max().item() + 1

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j

    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)


def complete_graph_from_list(L, create_using=None):
    G = nx.empty_graph(len(L), create_using)
    if len(L) > 1:
        if G.is_directed():
            edges = itertools.permutations(L, 2)
        else:
            edges = itertools.combinations(L, 2)
        G.add_edges_from(edges)
    return G


class CosineSoftMax(nn.Module):
    def __init__(self, kappa=5.):
        super(CosineSoftMax, self).__init__()
        self.kappa = kappa
        self.loss = nn.CrossEntropyLoss()

    def forward(self, z, targets):

        targets_ = targets.argmax(dim=1).to(z.device)
        inputs = self.kappa * z

        return self.loss(inputs, targets_)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.cosine_sim = nn.CosineSimilarity()
        self.margin = margin
        # self.K = 0.5
        # self.T = 0.2

    def forward(self, z, edges_list):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """

        # relabel edges
        relabeled_edges_list = []
        max_node = 0
        for edges in edges_list:
            e_idx = edges.edge_index.clone()
            e_idx += max_node
            max_node = edges.n_nodes
            relabeled_edges_list.append(e_idx)

        relabeled_edges_list = torch.cat(relabeled_edges_list, dim=1)
        relabeled_edges_list.to(z.device)

        # do sampling
        i, j, k = structured_negative_sampling(relabeled_edges_list)

        d_ap = 1 - self.cosine_sim(z[i], z[j])
        d_an = 1 - self.cosine_sim(z[i], z[k])

        loss = torch.log1p(d_ap - d_an).mean()
        return loss


if __name__ == "__main__":
    from ksptrack.models.my_augmenters import Normalize
    from torch.utils.data import DataLoader
    from ksptrack.siamese.modeling.siamese import Siamese
    from loader import Loader
    from os.path import join as pjoin
    from ksptrack.siamese import utils as utls
    from ksptrack.siamese.distrib_buffer import DistribBuffer

    device = torch.device('cuda')
    transf_normal = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    dl = Loader(pjoin('/home/ubelix/artorg/lejeune/data/medical-labeling',
                      'Dataset20'),
                normalization='rescale',
                resize_shape=512)
    dl = DataLoader(dl, collate_fn=dl.collate_fn, batch_size=2, shuffle=True)

    model = Siamese(15, 15, backbone='unet')

    run_path = '/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset20'
    cp_path = pjoin(run_path, 'checkpoints', 'init_dec.pth.tar')
    state_dict = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    criterion = TripletLoss()

    distrib_buff = DistribBuffer(10, thr_assign=0.0001)
    distrib_buff.maybe_update(model, dl, device)

    for data in dl:
        data = utls.batch_to_device(data, device)

        _, targets = distrib_buff[data['frame_idx']]
        res = model(data)
        loss = criterion(res['proj_pooled_feats'], targets, data['graph'])
