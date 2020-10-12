import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from torch import nn
import pandas as pd
from .utils import df_to_tgt


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


def num_nan_inf(t):
    return torch.isnan(t).sum() + torch.isinf(t).sum()


def do_previews(images, labels, pos_nodes, neg_nodes):
    import matplotlib.pyplot as plt

    max_node = 0
    ims = []
    for im in images:
        im = (255 *
              np.rollaxis(im.squeeze().detach().cpu().numpy(), 0, 3)).astype(
                  np.uint8)
        ims.append(im)

    labels = labels.squeeze().detach().cpu().numpy()
    pos_map = np.zeros_like(labels)
    neg_map = np.zeros_like(labels)

    for n in pos_nodes:
        pos_map[labels == n.item()] = True
    for n in neg_nodes:
        neg_map[labels == n.item()] = True

    pos_map = np.concatenate([a for a in pos_map], axis=0)
    neg_map = np.concatenate([a for a in neg_map], axis=0)
    maps = np.concatenate((pos_map, neg_map), axis=1)
    ims = np.concatenate(ims, axis=0)

    cmap = plt.get_cmap('viridis')
    maps = (cmap(
        (maps * 255).astype(np.uint8))[..., :3] * 255).astype(np.uint8)

    all_ = np.concatenate((ims, maps), axis=1)

    return all_


def get_pos_negs(keypoints, nodes, input, mode):
    all_clusters = torch.unique(nodes[-1])

    if (len(keypoints) > 0):
        if (mode == 'single'):
            all_nodes = torch.unique(nodes[0, :])
            pos_nodes = torch.tensor(keypoints).to(all_nodes.device).long()
            compareview = pos_nodes.repeat(all_nodes.numel(), 1).T
            neg_nodes = all_nodes[(compareview != all_nodes).T.prod(1) == 1]

        else:
            cluster_clicked = torch.cat(
                [nodes[-1, (nodes[0, :] == l)] for l in keypoints])

            pos_nodes = torch.cat([
                torch.unique(nodes[0, nodes[-1] == c])
                for c in torch.unique(cluster_clicked)
            ])
            compareview = cluster_clicked.repeat(all_clusters.shape[0], 1).T
            compareview = compareview.to(all_nodes.device)
            neg_clusters = all_clusters[(
                compareview != all_clusters).T.prod(1) == 1]

            neg_nodes = [
                torch.unique(nodes[0, nodes[-1] == c]) for c in neg_clusters
            ]

            if ('rand' in mode):
                if ('weighted' in mode):
                    weights = torch.tensor([len(n) for n in neg_nodes
                                            ]).to(all_clusters.device)
                    weights = weights.max().float() / weights.float()
                else:
                    weights = torch.ones(len(neg_nodes)).to(
                        all_clusters.device)

                neg_cluster_idx = torch.multinomial(weights,
                                                    1,
                                                    replacement=True)
                neg_cluster = neg_clusters[neg_cluster_idx]
                neg_nodes = torch.unique(nodes[:2, nodes[-1] == neg_cluster])
            else:
                neg_nodes = torch.cat(neg_nodes)

        pos_tgt = torch.ones(pos_nodes.numel()).float().to(nodes.device)
        neg_tgt = torch.zeros(neg_nodes.numel()).float().to(nodes.device)
        tgt = torch.cat((neg_tgt, pos_tgt))
        input = torch.cat((input[neg_nodes], input[pos_nodes]))
    else:
        tgt = torch.zeros(input.numel()).to(nodes.device)

    # path = '/home/ubelix/artorg/lejeune/runs/maps_{:04d}.png'.format(
    #     data['frame_idx'][0])
    # if (not os.path.exists(path)):
    #     maps = do_previews(data['image'], data['labels'], pos_nodes,
    #                        neg_nodes)
    #     io.imsave(path, maps)

    return input, tgt


def cross_entropy_logits(in_, tgt, reduction='none'):
    # in_ : tensor (without logit)
    # tgt : integer (0 or 1)
    tgt = torch.ones_like(in_) * tgt

    return F.binary_cross_entropy_with_logits(in_, tgt, reduction=reduction)


class BalancedBCELoss(nn.Module):
    """
    """
    def __init__(self, pi=0.25):
        super(BalancedBCELoss, self).__init__()
        self.pi = pi

    def forward(self, input, target, pi=None, pi_mul=1.):
        """
        """
        if pi is None:
            pi = self.pi

        if not isinstance(pi, list):
            pi = np.unique(target['frame']).shape[0] * [pi]

        loss = 0
        for in_, tgt, pi_ in zip(input, target, pi):
            target_pos, target_neg, target_aug = df_to_tgt(tgt)

            in_p_plus = in_[(target_pos + target_aug).bool()]
            Rp_plus = cross_entropy_logits(in_p_plus, 1)

            in_u_minus = in_[target_neg.bool()]
            Ru_minus = cross_entropy_logits(in_u_minus, 0)

            loss_p_plus = pi_mul * pi_ * Rp_plus.sum() / target_pos.sum()
            loss_u_minus = (1 -
                            pi_mul * pi_) * Ru_minus.sum() / target_neg.sum()

            loss += loss_p_plus + loss_u_minus

        return loss / len(input)


class PULoss(nn.Module):
    """
    https://arxiv.org/abs/2002.04672
    """
    def __init__(self, pi=0.25, do_ascent=False, beta=0, aug_in_neg=False):
        super(PULoss, self).__init__()
        self.pi = pi
        self.do_ascent = do_ascent
        self.beta = beta
        self.aug_in_neg = aug_in_neg

    def forward(self, input, target, pi=None, pi_mul=1.):
        """
        """
        if pi is None:
            pi = self.pi

        if not isinstance(pi, list):
            pi = np.unique(target['frame']).shape[0] * [pi]

        loss = 0
        for in_, tgt, pi_ in zip(input, target, pi):
            target_pos, target_neg, target_aug = df_to_tgt(tgt)

            in_p_plus = in_[(target_pos + target_aug).bool()]
            Rp_plus = cross_entropy_logits(in_p_plus, 1)

            in_u_minus = in_[target_neg.bool()]
            Ru_minus = cross_entropy_logits(in_u_minus, 0)

            if self.aug_in_neg:
                in_p_minus = in_[(target_pos + target_aug).bool()]
                Rp_minus = cross_entropy_logits(in_p_minus, 0)
            else:
                in_p_minus = in_[target_pos.bool()]
                Rp_minus = cross_entropy_logits(in_p_minus, 0)

            loss_p_plus = pi_mul * pi_ * Rp_plus
            loss_u_minus = Ru_minus
            loss_p_minus = pi_mul * pi_ * Rp_minus

            if target_pos.sum() == 0:
                loss_p_plus = torch.tensor([0.]).to(in_.device)
                loss_p_minus = torch.tensor([0.]).to(in_.device)

            if self.do_ascent:
                # check_neg_risk = loss_u_minus - loss_p_minus.mean()
                do_ascent = loss_u_minus.mean() - loss_p_minus.mean(
                ) < self.beta

                # loss += loss_p_plus.sum() / target_pos.sum()
                loss += loss_p_plus.mean()
                if do_ascent:
                    l_ = loss_u_minus.mean() - loss_p_minus.mean()
                    # l_ = torch.clamp(l_, min=self.beta)
                    loss -= l_
                    # loss += loss_u_minus[~idx_ascent].mean() - loss_p_minus.mean()
                else:
                    loss += loss_u_minus.mean() - loss_p_minus.mean()
            else:
                # loss += loss_p_plus.sum() / target_pos.sum()
                loss += loss_p_plus.mean()
                loss += F.relu(loss_u_minus.mean() - loss_p_minus.mean())

        return loss / len(input)


class ClusterPULoss(nn.Module):
    """
    https://arxiv.org/abs/2002.04672
    """
    def __init__(self, pi=0.25, mode='all'):
        super(ClusterPULoss, self).__init__()
        self.pi = pi
        self.mode = mode
        modes = ['all', 'single', 'rand_weighted', 'rand_uniform']

        assert mode in modes, print('mode should be either all or rand')

    def cross_entropy_logits(self, in_, tgt, reduction='none'):
        # in_ : tensor (without logit)
        # tgt : integer (0 or 1)
        tgt = torch.ones_like(in_) * tgt

        return F.binary_cross_entropy_with_logits(in_,
                                                  tgt,
                                                  reduction=reduction)

    def forward(self, input, nodes, data, augmented_set=None):
        """
        """

        input, tgt = get_pos_negs(data['pos_labels'], nodes, input, self.mode)

        in_pos = input[tgt == 1]
        in_unl = input[tgt == 0]

        Rp_plus = self.cross_entropy_logits(in_pos, 1)
        Rp_minus = self.cross_entropy_logits(in_pos, 0)
        Ru_minus = self.cross_entropy_logits(in_unl, 0)

        if (augmented_set is not None):
            # add a term to Rp_plus
            pass

        loss_p = self.pi * Rp_plus.mean()
        loss_u = Ru_minus.mean()
        loss_u -= self.pi * Rp_minus.mean()
        loss_u = loss_u.relu()

        return loss_p + loss_u


class ClusterFocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=0.25, mode='all'):
        super(ClusterFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-6
        self.mode = mode

    def forward(self, input, nodes, data, override_alpha=None):
        """
        """

        if (override_alpha is None):
            alpha = self.alpha
        else:
            alpha = override_alpha

        input, tgt = get_pos_negs(data['pos_labels'], nodes, input, self.mode)

        logit = input.sigmoid().clamp(self.eps, 1 - self.eps)
        pt0 = 1 - logit[tgt == 0]
        pt1 = logit[tgt == 1]
        loss = F.binary_cross_entropy_with_logits(input, tgt, reduction='none')
        loss[tgt == 1] = loss[tgt == 1] * alpha * (1 - pt1)**self.gamma
        loss[tgt == 0] = loss[tgt == 0] * (1 - alpha) * (1 - pt0)**self.gamma

        return loss.mean()


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def safe_random_choice(input_data, size, p=None):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, p=p, replace=replace)


# sample triplets, with a weighted distribution if weights is specified.
def get_random_triplet_indices(labels,
                               ref_labels=None,
                               t_per_anchor=None,
                               weights=None,
                               ap_pw_weights=None):

    a_idx, p_idx, n_idx = [], [], []
    labels = labels.cpu().numpy()
    ref_labels = labels if ref_labels is None else ref_labels.cpu().numpy()
    batch_size = ref_labels.shape[0]
    label_count = dict(zip(*np.unique(ref_labels, return_counts=True)))
    indices = np.arange(batch_size)

    for i, label in enumerate(labels):
        curr_label_count = label_count[label]
        if ref_labels is labels: curr_label_count -= 1
        if curr_label_count <= 1:
            continue
        k = curr_label_count if t_per_anchor is None else t_per_anchor

        if weights is not None and not np.any(np.isnan(weights[i])):
            n_idx += c_f.NUMPY_RANDOM.choice(batch_size, k,
                                             p=weights[i]).tolist()
        else:
            possible_n_idx = list(np.where(ref_labels != label)[0])
            n_idx += c_f.NUMPY_RANDOM.choice(possible_n_idx, k).tolist()

        a_idx.extend([i] * k)
        if (ap_pw_weights is None):
            curr_p_idx = safe_random_choice(
                np.where((ref_labels == label) & (indices != i))[0], k)
        else:
            possible_p_idx = np.where((ref_labels == label)
                                      & (indices != i))[0]
            p = ap_pw_weights[i, possible_p_idx]
            p = p + 1e-6
            p = p / p.sum()
            curr_p_idx = safe_random_choice(possible_p_idx, k, p=p)

        p_idx.extend(curr_p_idx.tolist())

    return (
        torch.LongTensor(a_idx),
        torch.LongTensor(p_idx),
        torch.LongTensor(n_idx),
    )


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


class DistanceWeightedMiner(BaseTupleMiner):
    def __init__(self,
                 cutoff=0.2,
                 nonzero_loss_cutoff=10,
                 n_triplets_per_anchor=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)
        self.n_triplets_per_anchor = n_triplets_per_anchor
        self.cutoff = cutoff

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
            labels = labels.to(embeddings.device)
            mining_output = self.mine(embeddings, labels)
        self.output_assertion(mining_output)
        return mining_output

    def mine(self, embeddings, labels):
        d = embeddings.size(1)
        dist_mat = pairwise_distances(embeddings)

        # Cut off to avoid high variance.
        dist_mat = torch.max(dist_mat,
                             torch.tensor(self.cutoff).to(dist_mat.device))

        # this is for the weights of picking negatives, i.e. it compute log(1/q(d))
        #
        # Subtract max(log(distance)) for stability.
        # See the first equation from Section 4 of the paper
        # log_weights = (2.0 - float(d)) * torch.log(dist_mat) - (
        #     float(d - 3) / 2) * torch.log(1.0 - 0.25 * (dist_mat**2.0))
        # weights = torch.exp(log_weights - torch.max(log_weights))
        weights = 1 / (dist_mat + 1e-8)

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones(weights.size()).to(embeddings.device)
        same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask[same_class] = 0

        weights = weights * mask * (
            (dist_mat < self.nonzero_loss_cutoff).float())
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        np_weights = weights.cpu().numpy()

        return get_random_triplet_indices(
            labels,
            weights=np_weights,
            t_per_anchor=self.n_triplets_per_anchor)


class TripletCosineMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        distance_norm: The norm used when calculating distance between embeddings
        power: Each pair's loss will be raised to this power.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """
    def __init__(self,
                 margin=0.3,
                 smooth_loss=False,
                 triplets_per_anchor="all",
                 **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.triplets_per_anchor = triplets_per_anchor
        self.cs = nn.CosineSimilarity(dim=1)
        self.smooth_loss = smooth_loss

    def forward(self, embeddings, labels, indices_tuple=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss (float)
        """
        self.reset_stats()
        c_f.assert_embeddings_and_labels_are_same_size(embeddings, labels)
        labels = labels.to(embeddings.device)

        loss_dict = self.compute_loss(embeddings, labels, indices_tuple)
        return self.reducer(loss_dict, embeddings, labels)

    def compute_loss(self, embeddings, labels, indices_tuple, sigma=1):
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[
            positive_idx], embeddings[negative_idx]
        a_p_dist = 1 - self.cs(anchors, positives)
        a_n_dist = 1 - self.cs(anchors, negatives)

        if self.smooth_loss:
            inside_exp = a_p_dist - a_n_dist
            inside_exp = self.maybe_modify_loss(inside_exp)
            loss = torch.log(1 + torch.exp(inside_exp))
        else:
            dist = a_p_dist - a_n_dist
            loss_modified = self.maybe_modify_loss(dist + self.margin)
            loss = torch.nn.functional.relu(loss_modified)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet"
            }
        }

    def maybe_modify_loss(self, x):
        return x

    def get_default_reducer(self):
        return AvgNonZeroReducer()


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = cosine_distance_torch(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis=1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis=1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg


class TripletLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sampler = BatchHardTripletSelector()
        self.cs = nn.CosineSimilarity(dim=1)

    def forward(self, feats, labels):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.
        """

        a, p, n = self.sampler(feats, labels)

        dap = 1 - self.cs(a, p)
        dan = 1 - self.cs(a, n)

        loss = torch.clamp(dap - dan + self.margin, min=0)
        loss = loss[loss > 0]
        # loss = torch.log1p((dap - dan).exp())
        loss = loss.mean()

        return loss


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
    	margin
    	type_of_triplets: options are "all", "hard", or "semihard".
    		"all" means all triplets that violate the margin
    		"hard" is a subset of "all", but the negative is closer to the anchor than the positive
    		"semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """
    def __init__(self, margin, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=[
            "avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"
        ],
                                          is_stat=True)
        self.type_of_triplets = type_of_triplets

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = get_random_triplet_indices(
            labels, ref_labels, t_per_anchor=1)
        anchors, positives, negatives = embeddings[anchor_idx], ref_emb[
            positive_idx], ref_emb[negative_idx]
        ap_dist = F.cosine_similarity(anchors, positives)
        an_dist = F.cosine_similarity(anchors, negatives)
        triplet_margin = an_dist - ap_dist
        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(triplet_margin).item()
        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= an_dist <= ap_dist
            elif self.type_of_triplets == "semihard":
                threshold_condition &= an_dist > ap_dist
        return anchor_idx[threshold_condition], positive_idx[
            threshold_condition], negative_idx[threshold_condition]


if __name__ == "__main__":
    criterion = TripletMarginMiner(margin=0.3, type_of_triplets="semihard")
    feats = torch.randn((11, 128))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

    criterion(feats, labels)
