from torch import nn
import numpy as np
import torch
from torch.nn import functional as F


def get_edges_probas(probas, edges_nn, thrs=[0.6, 0.8],
                     clusters=None):

    edges = {'sim': [], 'disim': []}

    # get edges with both nodes with proba > thr
    edges_mask_sim = probas[edges_nn[:, 0]] >= thrs[1]
    edges_mask_sim *= probas[edges_nn[:, 1]] >= thrs[1]
    if(clusters is not None):
        edges_mask_sim *= (clusters[edges_nn[:, 1]] == clusters[edges_nn[:, 0]])
    edges_sim = edges_nn[edges_mask_sim, :]

    # sim_probas = torch.stack((probas[edges_sim[:, 0]],
    #                           probas[edges_sim[:, 1]]), dim=1)
    # sim_probas = torch.max(sim_probas, dim=1)[0]
    sim_probas = torch.ones(edges_sim.shape[0]).to(probas).float()
    edges_sim = [edges_sim, sim_probas]
    edges['sim'] = edges_sim

    # get edges with one node > thr[1] and other < thr[0]
    edges_mask_disim_0 = probas[edges_nn[:, 0]] >= thrs[1]
    edges_mask_disim_0 *= probas[edges_nn[:, 1]] < thrs[0]
    edges_mask_disim_1 = probas[edges_nn[:, 1]] >= thrs[1]
    edges_mask_disim_1 *= probas[edges_nn[:, 0]] < thrs[0]
    edges_mask_disim = edges_mask_disim_0 + edges_mask_disim_1
    # if(clusters is not None):
    #     edges_mask_sim *= (clusters[edges_nn[:, 1]] != clusters[edges_nn[:, 0]])
    edges_disim = edges_nn[edges_mask_disim, :]

    # disim_probas = torch.stack((probas[edges_disim[:, 0]],
    #                             probas[edges_disim[:, 1]]), dim=1)
    # disim_probas = torch.min(disim_probas, dim=1)[0]
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

        cater_labls = torch.cat((torch.ones_like(probas['sim']),
                                 torch.zeros_like(probas['disim'])))
        pos_weight = (cater_labls == 0).sum().float() / cater_labls.numel()
        neg_weight = (cater_labls == 1).sum().float() / cater_labls.numel()

        weights = torch.cat((torch.ones_like(probas['sim']) * pos_weight,
                             torch.ones_like(probas['disim']) * neg_weight))
        loss = self.criterion(torch.cat((probas['sim'], probas['disim'])),
                              torch.cat((edges['sim'][1],
                                         edges['disim'][1])).float())
        loss *= weights

        return loss.mean()


class LocationPairwiseLoss(nn.Module):
    def __init__(self, thrs=[0.6, 0.8]):
        super(LocationPairwiseLoss, self).__init__()

        self.criterion_clust = torch.nn.KLDivLoss(reduction='sum')
        self.criterion_pw = torch.nn.KLDivLoss(reduction='sum')
        self.thrs = thrs

    def forward(self, probas, data, edges_nn, clusters, targets):

        edges = get_edges_keypoint_probas(probas,
                                          data,
                                          edges_nn,
                                          thrs=self.thrs)

        loss_pw = torch.tensor(0.).to(clusters)
        loss_pw_sim = None
        loss_pw_disim = None
        # do positives
        if (edges['sim'].numel() > 0):
            c0 = clusters[edges['sim'][:, 0]]
            c1 = clusters[edges['sim'][:, 1]]
            loss_pw_sim = self.criterion_pw(
                (c0 + 1e-7).log(), c1) / c0.shape[0]
            loss_pw_sim += self.criterion_pw(
                (c1 + 1e-7).log(), c0) / c0.shape[0]
            loss_pw += loss_pw_sim

        # do negatives
        if (edges['disim'].numel() > 0):
            c0 = clusters[edges['disim'][:, 0]]
            c1 = clusters[edges['disim'][:, 1]]
            loss_pw_disim = F.relu(1 -
                                   self.criterion_pw((c0 + 1e-7).log(), c1) /
                                   c0.shape[0])
            loss_pw_disim += F.relu(1 -
                                    self.criterion_pw((c1 + 1e-7).log(), c0) /
                                    c0.shape[0])
            loss_pw += loss_pw_disim

        loss_clust = self.criterion_clust(
            (clusters + 1e-7).log(), targets) / clusters.shape[0]

        return {'loss_clust': loss_clust, 'loss_pw': loss_pw}


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
