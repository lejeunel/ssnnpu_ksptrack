import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from ksptrack.siamese import utils as utls
from ksptrack.siamese.clustering import get_features
from ksptrack.siamese.loader import Loader
from ksptrack.utils.link_agent_gmm import LinkAgentGMM


class LinkAgentSiam(LinkAgentGMM):
    def __init__(self,
                 csv_path,
                 data_path,
                 model,
                 entrance_radius=0.1,
                 cuda=False,
                 K=0.5,
                 T=0.2):

        super().__init__(csv_path,
                         data_path,
                         model,
                         entrance_radius=entrance_radius,
                         cuda=cuda)

        self.sigmoid = torch.nn.Sigmoid()
        self.prepare_all()
        self.K = K
        self.T = T

    def prepare_all(self, all_edges_nn=None, feat_field='pooled_feats'):
        print('preparing features for linkAgent')
        # form initial cluster centres
        self.obj_preds = []
        self.siam_feats = []

        self.model.eval()
        self.model.to(self.device)
        print('getting features')
        pbar = tqdm.tqdm(total=len(self.dl))
        for index, data in enumerate(self.dl):
            data = utls.batch_to_device(data, self.device)
            with torch.no_grad():
                res = self.model(data)

            self.obj_preds.append(self.sigmoid(res['obj_pred']).cpu().numpy())
            self.siam_feats.append(res['locmotionapp'])

            pbar.update(1)
        pbar.close()

    def get_proba(self, f0, l0, f1, l1, *args):

        f0 = self.siam_feats[f0][l0]
        f1 = self.siam_feats[f1][l1]

        dist = (f0 - f1).abs()
        pw = self.model.pw_clf(dist)
        prob = torch.exp(-pw).detach().cpu().numpy()

        return prob
