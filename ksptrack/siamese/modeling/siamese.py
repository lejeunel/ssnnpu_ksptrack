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
                 alpha: float = 1.0,
                 backbone='drn'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.dec = DEC(embedded_dims, cluster_number,
                       alpha, backbone)

        self.clf = nn.Linear(self.dec.autoencoder.last_feats_dims, 1)
        self.sigmoid = nn.Sigmoid()

        if(backbone == 'drn'):
            self.in_dims = [3, 32, 64, 128, 256]
        else:
            self.in_dims = [3] + self.dec.autoencoder.filts_dims

        self.gcns = []
        for i in range(len(self.in_dims) - 1):
            # gcn_ = gnn.GCNConv(in_channels=in_dims[i],
            #                    out_channels=in_dims[i+1],
            #                    normalize=False)
            gcn_ = gnn.SAGEConv(in_channels=self.in_dims[i],
                                out_channels=self.in_dims[i+1],
                                normalize=False)
            # gcn_ = gnn.SignedConv(in_channels=self.in_dims[i],
            #                       out_channels=self.in_dims[i+1],
            #                       first_aggr=True if i == 0 else False)

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
                probas=None):

        res = self.dec(data)

        # obj_pred = self.clf(res['pooled_feats'])
        obj_pred = self.sp_pool(res['output'], data['labels'])
        res.update({'obj_pred': obj_pred})

        if(edges_nn is not None):
            # similar edges according clusters
            edges_mask_sim = (targets[edges_nn[:, 0]].argmax(dim=1) ==
                              targets[edges_nn[:, 1]].argmax(dim=1))
            edges_nn_sim = edges_nn[edges_mask_sim, :].T

            # dissimilar edges according to probas
            # dp = (probas[edges_nn[:, 0]] - probas[edges_nn[:, 1]]).abs()
            # edges_mask_disim_p = dp > 0.5

            # edges_mask_disim_c = (targets[edges_nn[:, 0]].argmax(dim=1) !=
            #                       targets[edges_nn[:, 1]].argmax(dim=1))
            # edges_nn_disim = edges_nn[edges_mask_disim_p, :].T

            # add self loops
            edges_nn_sim, _ = gutls.add_self_loops(edges_nn_sim)

            Z = self.sp_pool(data['image'], data['labels'])
            Z = F.relu(self.gcns[0](Z, edges_nn_sim))

            H = [r for r in res['layers']][:-1]
            H.append(res['feats'])
            for i, gcn in enumerate(self.gcns[1:]):
                # do GCNConv and average result with encoder features
                H_pool = self.sp_pool(H[i], data['labels'])
                Z = torch.stack((Z, H_pool),
                                dim=0).mean(dim=0)
                Z = F.relu(gcn(Z, edges_nn_sim))
                # Z_pos = Z[:, :self.in_dims[i+1]]
                # Z_pos = torch.stack((Z_pos, H_pool),
                #                     dim=0).mean(dim=0)
                # Z_neg = Z[:, self.in_dims[i+1]:]
                # Z_neg = torch.stack((Z_neg, H_pool),
                #                     dim=0).mean(dim=0)
                # Z = torch.cat((Z_pos, Z_neg), dim=1)

            Z = self.dec.transform(Z)
            # Z_pos = self.dec.transform(Z[:, :self.in_dims[i+1]])
            # Z_neg = self.dec.transform(Z[:, self.in_dims[i+1]:])
            # Z = torch.cat((Z_pos, Z_neg), dim=1)

            # res.update({'Z_pos': Z_pos})
            # res.update({'Z_neg': Z_neg})
            res.update({'Z': Z})
            # res.update({'edges_neg': edges_nn_disim})
            res.update({'edges_pos': edges_nn_sim})

        return res
