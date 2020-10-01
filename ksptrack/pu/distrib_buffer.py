from ksptrack.siamese import utils as utls
from ksptrack.siamese.clustering import get_features
import torch
import numpy as np


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch**2) / torch.sum(batch, 0)
    weight = (weight.t() / torch.sum(weight, 1)).t()

    return weight


class DistribBuffer:
    def __init__(self, period, thr_assign=0.001):
        self.epoch = 0
        self.period = period
        self.thr_assign = thr_assign

        self.tgt_distribs = []
        self.distribs = []
        self.assignments = []
        self.converged = False
        self.ratio_changed = 1

    def do_update(self, model, dataloader, device, clst_field='clusters'):

        print('updating targets...')
        feats, _, distrib = get_features(model,
                                         dataloader,
                                         device,
                                         return_distribs=True)

        tgt = target_distribution(torch.tensor(distrib))

        splits = [f.shape[0] for f in feats['pooled_feats']]
        self.tgt_distribs = torch.split(tgt.cpu().detach(), splits)
        self.distribs = torch.split(
            torch.tensor(distrib).cpu().detach(), splits)

        curr_assignments = [
            torch.argmax(f, dim=-1).cpu().detach().numpy()
            for f in self.distribs
        ]
        curr_assignments = np.concatenate(curr_assignments, axis=0)

        self.assignments.append(curr_assignments)

        if (len(self.assignments) > 1):

            n_changed = np.sum(self.assignments[-1] != self.assignments[-2])
            n = self.assignments[-1].size
            self.ratio_changed = n_changed / n
            print('ratio_changed: {}'.format(self.ratio_changed))
            if (self.ratio_changed < self.thr_assign):
                self.converged = True
        model.train()

    def maybe_update(self, model, dataloader, device, clst_field='clusters'):
        """
        Update target probabilities when we hit update period
        Increments epoch counter
        """

        if ((self.epoch == 0) or (self.epoch % self.period == 0)):
            self.do_update(model, dataloader, device, clst_field=clst_field)
            print('Next update in {} epochs'.format(self.period))

    def inc_epoch(self):
        self.epoch += 1

    def __getitem__(self, idx):

        if (isinstance(idx, int)):
            idx = [idx]

        targets = torch.cat([self.tgt_distribs[i] for i in idx])
        distribs = torch.cat([self.distribs[i] for i in idx])
        return distribs, targets
