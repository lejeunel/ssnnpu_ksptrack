import torch
import torch.nn as nn
import torch.nn.functional as F
from ksptrack.siamese.modeling.dec import DEC


class Siamese(nn.Module):
    """

    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def __init__(self,
                 embedded_dims=None,
                 cluster_number: int = 30,
                 roi_size=1,
                 roi_scale=1.0,
                 alpha: float = 1.0,
                 use_locations=False):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.dec = DEC(embedded_dims, cluster_number,
                       roi_size,
                       roi_scale,
                       alpha,
                       use_locations)

        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(embedded_dims,
                                 1,
                                 bias=False)

    def get_probas(self, X):

        X = torch.exp(
            -torch.norm(
                X[0] - X[1],
                dim=1))

        return X.squeeze()

    def forward(self, data, edges_nn=None):

        res = self.dec(data)

        if(edges_nn is not None):
            feats = res['proj_pooled_aspp_feats']
            X = torch.stack((feats[edges_nn[:, 0]],
                             feats[edges_nn[:, 1]]))

            probas = self.get_probas(X)

            res['probas_preds'] = probas

        return res
