import numpy as np
from ksptrack.siamese.modeling.siamese import Siamese
import torch
from ksptrack.siamese.loader import StackLoader


class MySpectralClustering:
    def __init__(self,
                 n_clusters,
                 embedded_dims,
                 model_path,
                 data_path,
                 cuda=False):

        self.n_clusters = n_clusters
        self.embedded_dims = embedded_dims

        self.device = torch.device('cuda' if cuda else 'cpu')
        self.data_path = data_path

        self.dset = StackLoader(data_path,
                                normalization='rescale',
                                depth=2,
                                resize_shape=512)

        self.model = Siamese(embedded_dims=embedded_dims,
                             cluster_number=n_clusters,
                             backbone='unet')
        print('loading checkpoint {}'.format(model_path))
        state_dict = torch.load(model_path,
                                map_location=lambda storage, loc: storage)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

    def fit_predict(self, feats, weights=None):

        # feats = whiten(feats)
        labels = self.clf.fit_predict(feats)
        self.cluster_centers = self.clf.cluster_centers_

        return labels, self.cluster_centers
