from torch import nn
import numpy as np
import torch
from ksptrack.siamese import utils as utls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
import networkx as nx
import itertools
import random
import torch.nn.functional as F
from skimage import io


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
    def __init__(self, kappa=20.):
        super(CosineSoftMax, self).__init__()
        self.kappa = kappa
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, z, targets):

        targets_ = targets.argmax(dim=1).to(z.device)

        bc = torch.bincount(targets_)
        freq_weights = bc.max() / bc.float()
        freq_smp_weights = freq_weights[targets.argmax(dim=1)]
        inputs = self.kappa * z

        loss = (freq_smp_weights * self.loss(inputs, targets_)).mean()
        return loss


def make_coord_map(batch_size, w, h):
    xx_ones = torch.ones([1, 1, 1, w], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, 1, h], dtype=torch.int32)

    xx_range = torch.arange(w, dtype=torch.int32)
    yy_range = torch.arange(h, dtype=torch.int32)
    xx_range = xx_range[None, None, :, None]
    yy_range = yy_range[None, None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 1, 3, 2)

    xx_channel = xx_channel.float() / (w - 1)
    yy_channel = yy_channel.float() / (h - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

    out = torch.cat([xx_channel, yy_channel], dim=1)

    return out


def sp_pool(f, labels):
    roi_pool = SupPixPool()
    upsamp = nn.UpsamplingBilinear2d(labels.size()[2:])
    pooled = [
        roi_pool(upsamp(f[b].unsqueeze(0)), labels[b].unsqueeze(0)).squeeze().T
        for b in range(labels.shape[0])
    ]
    pooled = torch.cat(pooled)
    return pooled


class PWClusteringLoss(nn.Module):
    def __init__(self, sigma=1e-3):
        super(PWClusteringLoss, self).__init__()
        self.sigma = sigma
        self.kl = nn.KLDivLoss(reduction='none')

    def forward(self, inputs, targets, edges=None, labels=None):

        if (edges is not None):
            b, _, w, h = labels.shape
            coords = make_coord_map(b, w, h).to(labels.device)
            locations = sp_pool(coords, labels)

            weights_dist = torch.exp(
                -(locations[edges[0]] - locations[edges[1]]).norm(dim=1)**2 /
                self.sigma)

            loss = self.kl((inputs[edges[0]] + 1e-8).log(),
                           targets[edges[1]]).sum(dim=1)
            loss += self.kl((inputs[edges[1]] + 1e-8).log(),
                            targets[edges[0]]).sum(dim=1)
            loss = loss / 2
            bc = torch.bincount(targets[edges[0]].argmax(dim=1))
            freq_weights = bc.max() / bc.float()
            freq_smp_weights = freq_weights[targets[edges[0]].argmax(dim=1)]
            loss = (freq_smp_weights * weights_dist * loss).mean()
        else:
            loss = self.kl(inputs, targets).mean()

        return loss


def sample_triplets(edges):

    # for each clique, generate a mask
    tplts = []
    for c in torch.unique(edges[-1, :]):
        cands = torch.unique(edges[:2, edges[-1, :] != c].flatten())
        idx = torch.randint(0,
                            cands.numel(),
                            size=((edges[-1, :] == c).sum(), ))
        tplts.append(
            torch.cat((edges[:2, edges[-1, :] == c], cands[idx][None, ...]),
                      dim=0))

    tplts = torch.cat(tplts, dim=1)

    return tplts


class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, labels, labels_clicked):

        labels_clicked_batch = []
        max_l = 0
        for i, l in enumerate(labels_clicked):
            labels_clicked_batch.append([l_ + max_l for l_ in l])
            max_l += torch.unique(labels[i]).numel() + 1

        idx = torch.tensor(labels_clicked_batch).to(input.device).flatten()
        loss = -(torch.log(self.sigmoid(input[idx]) + 1e-8)).mean()

        return loss


class LSMLoss(nn.Module):
    def __init__(self, K=0.5, T=0.1):
        super(LSMLoss, self).__init__()
        self.cs = nn.CosineSimilarity(dim=1)
        self.K = K
        self.T = T

    def forward(self, feats, edges):
        """
        Computes the triplet loss between positive node pairs and sampled
        non-node pairs.
        """

        # remove negative edges
        edges = edges[:, edges[-1, :] != -1]
        tplts = sample_triplets(edges)

        cs_ap = self.cs(feats[tplts[0]], feats[tplts[1]])
        cs_an = self.cs(feats[tplts[0]], feats[tplts[2]])

        loss_ap = torch.log1p(torch.exp(-(cs_ap - self.K) / self.T))
        loss_an = torch.log1p(torch.exp((cs_an - self.K) / self.T))

        return loss


def num_nan_inf(t):
    return torch.isnan(t).sum() + torch.isinf(t).sum()


def do_previews(labels, pos_nodes, neg_nodes):
    max_node = 0
    pos_maps = []
    neg_maps = []
    for lab in labels:
        lab = lab.squeeze().detach().cpu().numpy()
        pos_map = np.zeros_like(lab)
        neg_map = np.zeros_like(lab)
        for n in pos_nodes:
            pos_map[lab == n.item() - max_node] = True
        for n in neg_nodes:
            neg_map[lab == n.item() - max_node] = True

        max_node += lab.max() + 1

        pos_maps.append(pos_map)
        neg_maps.append(neg_map)

    pos_maps = np.concatenate(pos_maps, axis=0)
    neg_maps = np.concatenate(neg_maps, axis=0)
    maps = np.concatenate((pos_maps, neg_maps), axis=1)
    return maps


class ClusterObj(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(ClusterObj, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, edges, data):
        """
        """

        keypoints = data['clicked']
        edges_ = edges[:, edges[-1] != -1]
        # get edges that corresponds to keypoints
        cluster_clicked = torch.cat([
            edges_[-1, (edges_[0, :] == l) + (edges_[1, :] == l)]
            for l in keypoints
        ])

        pos_nodes = torch.cat([
            torch.unique(edges_[:2, edges_[-1] == c])
            for c in torch.unique(cluster_clicked)
        ])
        all_nodes = torch.arange(0, input.numel()).to(input.device)

        # select random negative cluster
        all_clusters = torch.unique(edges_[-1])
        compareview = cluster_clicked.repeat(all_clusters.shape[0], 1).T
        neg_clusters = all_clusters[(
            compareview != all_clusters).T.prod(1) == 1]
        neg_cluster = neg_clusters[torch.randint(neg_clusters.numel(), (1, ))]
        neg_nodes = torch.unique(edges_[:2, edges_[-1] == neg_cluster])

        # take all in non-intersection
        # negative nodes are the non-intersection
        # compareview = pos_nodes.repeat(all_nodes.shape[0], 1).T
        # neg_nodes = all_nodes[(compareview != all_nodes).T.prod(1) == 1]

        # maps = do_previews(data['labels'], pos_nodes, neg_nodes)
        # io.imsave('/home/ubelix/artorg/lejeune/runs/maps.png', maps)

        pos_tgt = torch.ones(pos_nodes.numel()).float().to(edges.device)
        neg_tgt = torch.zeros(neg_nodes.numel()).float().to(edges.device)
        tgt = torch.cat((neg_tgt, pos_tgt))
        input = torch.cat((input[neg_nodes], input[pos_nodes]))

        # if (self.alpha is None):
        #     pos_weight = tgt.numel() / (2 * pos_nodes.numel())
        #     neg_weight = tgt.numel() / (2 * neg_nodes.numel())
        #     weights = torch.cat(
        #         (neg_weight * torch.ones(neg_nodes.numel()),
        #          pos_weight * torch.ones(pos_nodes.numel()))).to(edges.device)
        # else:
        #     weights = torch.cat(
        #         (self.alpha[0] * torch.ones(neg_nodes.numel()), self.alpha[1] *
        #          torch.ones(pos_nodes.numel()))).to(edges.device)

        p = input.sigmoid()
        pt = p * tgt + (1 - p) * (1 - tgt)  # pt = p if t > 0 else 1-p
        w = self.alpha * tgt + (1 - self.alpha) * (1 - tgt)
        w = w * (1 - pt).pow(self.gamma)
        w = w.detach()
        loss = F.binary_cross_entropy_with_logits(input, tgt, w)

        return loss


class RAGTripletLoss(nn.Module):
    def __init__(self):
        super(RAGTripletLoss, self).__init__()
        self.cs = nn.CosineSimilarity(dim=1)
        # self.cs = nn.CosineSimilarity(dim=1, eps=1e-3)
        self.margin = 0.5

    def forward(self, feats, edges):
        """
        Computes the triplet loss between positive node pairs and sampled
        non-node pairs.
        """

        # remove negative edges
        edges = edges[:, edges[-1, :] != -1]
        tplts = sample_triplets(edges)

        xa = feats[tplts[0]]
        xp = feats[tplts[1]]
        xn = feats[tplts[2]]
        # print('[RAGTripletLoss] num. NaN feats: {}'.format(
        #     num_nan_inf(xa) + num_nan_inf(xp) + num_nan_inf(xn)))
        cs_ap = self.cs(xa, xp)
        cs_an = self.cs(xa, xn)
        # print('[RAGTripletLoss] num. NaN cs: {}'.format(
        #     num_nan_inf(cs_ap) + num_nan_inf(cs_an)))
        dap = 1 - cs_ap
        dan = 1 - cs_an

        # weight by clique size
        # bc = torch.bincount(edges[-1])
        # freq_weights = bc.float() / edges.shape[-1]
        # freq_smp_weights = freq_weights[edges[-1]]

        # loss = torch.log1p(dap - dan)
        loss = torch.clamp(dap - dan + self.margin, min=0)
        loss = loss[loss > 0]
        loss = loss.mean()
        # print('[RAGTripletLoss] loss: {}'.format(loss))

        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.5

    def forward(self, sim_ap, sim_an):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.
        """

        dap = 1 - sim_ap
        dan = 1 - sim_an

        loss = torch.clamp(dap - dan + self.margin, min=0)
        # print('[TripletLoss] ratio loss items > 0: {}'.format(
        #     (loss > 0).sum().float().item() / loss.numel()))
        loss = loss[loss > 0]
        loss = loss.mean()
        # print('[RAGTripletLoss] loss: {}'.format(loss))

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

    dl = Loader(pjoin('/home/ubelix/artorg/lejeune/data/medical-labeling',
                      'Dataset30'),
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
