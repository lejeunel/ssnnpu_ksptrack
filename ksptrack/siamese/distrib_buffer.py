import utils as utls
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
        self.batch = 0
        self.period = period
        self.thr_assign = thr_assign

        self.tgt_distribs = []
        self.distribs = []
        self.old_assignments = None
        self.converged = False

    def maybe_update(self, model, dataloader, device):
        """
        Update target probabilities when we hit update period
        Increments epoch counter
        """

        model.eval()

        if((self.batch == 0) or (self.batch % self.period == 0)):
            clusters = []
            print('updating distribution(s). Next update in {} batches'.format(self.period))
            
            for i, data in enumerate(dataloader):
                data = utls.batch_to_device(data, device)

                with torch.no_grad():
                    res = model(data)
                clusters.append(res['clusters'])

            distrib = torch.cat(clusters)
            tgt = target_distribution(torch.cat(clusters))

            splits = [np.unique(s['labels']).size
                      for s in dataloader.dataset]
            self.tgt_distribs = torch.split(tgt, splits)
            self.distribs = torch.split(distrib, splits)

        self.batch += 1

        # curr_assignments = [torch.argmax(f, dim=-1).cpu().detach().numpy()
        #                     for f in self.distribs]
        # curr_assignments = np.concatenate(curr_assignments, axis=0)
        # if(self.old_assignments is not None):

        #     n_changed = np.sum(curr_assignments != self.old_assignments)
        #     n = curr_assignments.size
        #     ratio_changed = n_changed / n < self.thr_assign
        #     print('ratio_changed: {}'.format(ratio_changed))
        #     if(ratio_changed):
        #         self.converged = True

        # self.old_assignments = curr_assignments.copy()

        model.train()

    def __getitem__(self, idx):

        if(isinstance(idx, int)):
            idx = [idx]

        targets = torch.cat([self.tgt_distribs[i] for i in idx])
        distribs = torch.cat([self.distribs[i] for i in idx])
        return distribs, targets
        
