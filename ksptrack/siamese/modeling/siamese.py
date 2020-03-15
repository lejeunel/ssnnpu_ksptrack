import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls


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
                 use_flow=False,
                 out_dim=128):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.dec = DEC(embedded_dims, cluster_number, roi_size, roi_scale,
                       alpha, use_flow)
        self.out_dim = out_dim

        self.roi_pool = RoIPooling((roi_size, roi_size), roi_scale)
        self.gcn0 = gnn.GCNConv(in_channels=3,
                                out_channels=3,
                                normalize=False)
        self.gcn1 = gnn.GCNConv(in_channels=512+3,
                                out_channels=256,
                                normalize=False)

    def get_probas(self, X):

        X = torch.exp(-torch.norm(X[0] - X[1], dim=1))

        return X.squeeze()

    def forward(self, data, edges_nn=None, nodes_color=None, probas=None, thrs=[0.5, 0.5]):

        res = self.dec(data)

        if(edges_nn is not None):
            # remove edges according to probas
            edges_mask_sim = (probas[edges_nn[:, 0]] >=
                              thrs[1]) * (probas[edges_nn[:, 1]] >= thrs[1])
            edges_mask_sim += (probas[edges_nn[:, 0]] <
                               thrs[0]) * (probas[edges_nn[:, 1]] < thrs[0])
            edges_nn = edges_nn[edges_mask_sim, :].T

            # add self loops
            edges_nn, _ = gutls.add_self_loops(edges_nn)

            # get features from image itself
            if(nodes_color is None):
                Z0 = self.roi_pool(data['image'], data['bboxes'])
                f0 = torch.stack((Z0[edges_nn[0, :]], Z0[edges_nn[1, :]]))
                S0 = torch.exp(-torch.norm(f0[0, ...] - f0[1, ...], p=2, dim=-1)**2)
            else:
                f0 = torch.stack((nodes_color[edges_nn[0, :]],
                                  nodes_color[edges_nn[1, :]]))
                S0 = torch.exp(-torch.norm(f0[0, ...] - f0[1, ...], p=2, dim=-1)**2/255.)
            g0 = Data(x=Z0, edge_index=edges_nn, edge_attr=S0.unsqueeze(1))

            # do GCNConv and combine result with encoder features
            Z1 = F.relu(self.gcn0(g0.x, g0.edge_index, g0.edge_attr))
            Z1 = torch.cat((Z1, self.roi_pool(res['feats'], data['bboxes'])),dim=1)
            g1 = Data(x=Z1, edge_index=edges_nn, edge_attr=S0)

            # do GCNConv and combine result with aspp features
            Z2 = self.gcn1(g1.x, g1.edge_index, g1.edge_attr)
            Z2 += res['pooled_aspp_feats']
            Z2 = 0.5 * Z2
            f2 = self.dec.transform(Z2)
            clusters = F.softmax(f2, dim=1)

            res.update({'clusters_gcn': clusters})

        return res
