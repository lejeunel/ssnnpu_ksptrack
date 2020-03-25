import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool


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
                 out_dim=128):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.dec = DEC(embedded_dims, cluster_number, roi_size, roi_scale,
                       alpha)
        self.out_dim = out_dim

        self.clf = nn.Linear(embedded_dims, 1)

        self.roi_pool = RoIPooling((roi_size, roi_size), roi_scale)

        self.in_dims = [3, 32, 64, 128, 256]
        self.gcns = []
        for i in range(len(self.in_dims) - 1):
            # gcn_ = gnn.GCNConv(in_channels=in_dims[i],
            #                    out_channels=in_dims[i+1],
            #                    normalize=False)
            # gcn_ = gnn.SAGEConv(in_channels=in_dims[i],
            #                     out_channels=in_dims[i+1],
            #                     normalize=False)
            gcn_ = gnn.SignedConv(in_channels=self.in_dims[i],
                                  out_channels=self.in_dims[i+1],
                                  first_aggr=True if i == 0 else False)

            self.gcns.append(gcn_)

        self.gcns = nn.ModuleList(self.gcns)

        self.roi_pool = SupPixPool()

    def sp_pool(self, feats, labels):

        upsamp = nn.UpsamplingBilinear2d(labels.size()[2:])
        pooled_feats = [self.roi_pool(upsamp(feats[b].unsqueeze(0)),
                                      labels[b].unsqueeze(0)).squeeze().T
                        for b in range(labels.shape[0])]
        pooled_feats = torch.cat(pooled_feats)
        return pooled_feats

    def forward(self, data,
                targets=None,
                edges_nn=None,
                nodes_color=None,
                probas=None, thrs=[0.5, 0.5]):

        res = self.dec(data)

        if(edges_nn is not None):
            # similar edges according to probas and clusters
            edges_mask_sim_p = (probas[edges_nn[:, 0]] >=
                                thrs[1]) * (probas[edges_nn[:, 1]] >= thrs[1])
            edges_mask_sim_p += (probas[edges_nn[:, 0]] <
                                 thrs[0]) * (probas[edges_nn[:, 1]] < thrs[0])
            edges_mask_sim_c = (targets[edges_nn[:, 0]].argmax(dim=1) ==
                                targets[edges_nn[:, 1]].argmax(dim=1))
            edges_nn_sim = edges_nn[edges_mask_sim_p * edges_mask_sim_c, :].T

            # dissimilar edges according to probas and clusters
            edges_mask_disim_p = (probas[edges_nn[:, 0]] >=
                                  thrs[1]) * (probas[edges_nn[:, 1]] < thrs[0])
            edges_mask_disim_p += (probas[edges_nn[:, 1]] >=
                                   thrs[1]) * (probas[edges_nn[:, 0]] < thrs[0])
            edges_mask_disim_c = (targets[edges_nn[:, 0]].argmax(dim=1) !=
                                  targets[edges_nn[:, 1]].argmax(dim=1))
            edges_nn_disim = edges_nn[edges_mask_disim_p * edges_mask_disim_c, :].T

            # add self loops
            edges_nn_sim, _ = gutls.add_self_loops(edges_nn_sim)

            Z = self.sp_pool(data['image'], data['labels'])

            H = [r for r in res['layers']][:-1]
            H.append(res['feats'])
            for i, gcn in enumerate(self.gcns):
                # do GCNConv and average result with encoder features
                H_pool = self.sp_pool(H[i], data['labels'])
                Z = F.relu(gcn(Z, edges_nn_sim, edges_nn_disim))
                Z_pos = Z[:, :self.in_dims[i+1]]
                Z_pos = torch.stack((Z_pos, H_pool),
                                    dim=0).mean(dim=0)
                Z_neg = Z[:, self.in_dims[i+1]:]
                Z_neg = torch.stack((Z_neg, H_pool),
                                    dim=0).mean(dim=0)
                Z = torch.cat((Z_pos, Z_neg), dim=1)

            Z_pos = self.dec.transform(Z[:, :self.in_dims[i+1]])
            Z_neg = self.dec.transform(Z[:, self.in_dims[i+1]:])

            res.update({'Z_pos': Z_pos})
            res.update({'Z_neg': Z_neg})

        return res
