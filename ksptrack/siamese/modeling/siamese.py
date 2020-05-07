import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.losses import structured_negative_sampling
from ksptrack.siamese.modeling.dil_unet import ConvolutionalEncoder, ResidualBlock, DilatedConvolutions, init_weights
from ksptrack.siamese.modeling.coordconv import CoordConv2d


def sp_pool(feats, labels):
    roi_pool = SupPixPool()
    upsamp = nn.UpsamplingBilinear2d(labels.size()[2:])
    pooled_feats = [
        roi_pool(upsamp(feats[b].unsqueeze(0)),
                 labels[b].unsqueeze(0)).squeeze().T
        for b in range(labels.shape[0])
    ]
    pooled_feats = torch.cat(pooled_feats)
    return pooled_feats


class LocMotionAppearance(nn.Module):
    """

    """
    def __init__(self,
                 in_dims,
                 dropout_min=0.,
                 dropout_max=0.2,
                 mixing_coeff=0.,
                 sigma=0.05):
        """
        """
        super(LocMotionAppearance, self).__init__()
        self.gcns = []
        self.sigma = sigma
        self.mixing_coeff = mixing_coeff

        in_dims = in_dims + [in_dims[-1]]
        in_dims[0] += 4
        for i in range(len(in_dims) - 1):
            gcn_ = gnn.SAGEConv(in_channels=in_dims[i],
                                out_channels=in_dims[i + 1],
                                normalize=False)

            self.gcns.append(gcn_)

        self.gcns = nn.ModuleList(self.gcns)

        self.top_bns = nn.ModuleList([nn.BatchNorm1d(d) for d in in_dims])
        self.bns = nn.ModuleList([nn.BatchNorm1d(d) for d in in_dims[1:]])

    def make_coord_map(self, batch_size, w, h):
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

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        out = torch.cat([xx_channel, yy_channel], dim=1)

        return out

    def forward(self, data, autoenc_skips, edges_nn):

        # make location/motion input tensor
        batch_size, _, w, h = data['labels'].shape

        coord_maps = self.make_coord_map(batch_size, w,
                                         h).to(data['labels'].device)

        pooled_coords = sp_pool(coord_maps, data['labels'])
        pooled_fx = sp_pool(data['fx'], data['labels'])[..., None]
        pooled_fy = sp_pool(data['fy'], data['labels'])[..., None]

        n_samples = [torch.unique(l).numel() for l in data['labels']]

        data_x = sp_pool(autoenc_skips[0], data['labels'])
        data_x = torch.cat((pooled_coords, pooled_fx, pooled_fy, data_x),
                           dim=1)
        data_x = self.top_bns[0](data_x)
        x = Data(x=data_x, edge_index=edges_nn[:2])

        for i, g in enumerate(self.gcns):
            if (i > 0):
                skip = sp_pool(autoenc_skips[i], data['labels'])
                skip = self.top_bns[i](skip)
                x.x = self.bns[i - 1](x.x)
                x.x = (1 - self.mixing_coeff) * x.x + self.mixing_coeff * skip
            x.x = F.relu(g(x.x, x.edge_index))

        x = F.normalize(x.x, p=2, dim=1)

        return {'siam_feats': x}


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
                 backbone='drn',
                 out_channels=3):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.dec = DEC(embedding_dims=embedded_dims,
                       cluster_number=cluster_number,
                       alpha=alpha,
                       backbone=backbone,
                       out_channels=out_channels)

        self.sigmoid = nn.Sigmoid()

        self.roi_pool = SupPixPool()

        self.locmotionapp = LocMotionAppearance(
            in_dims=self.dec.autoencoder.filts_dims,
            dropout_min=0,
            dropout_max=0.2,
            mixing_coeff=0.5)

    def load_partial(self, state_dict):
        # filter out unnecessary keys
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict, strict=False)

    def forward(self, data, edges_nn=None):

        res = self.dec(data['image_noaug'], data['labels'])

        if (edges_nn is not None):
            res_siam = self.locmotionapp(data, res['skips'], edges_nn)
            res.update(res_siam)

        res['rho_hat'] = sp_pool(res['output'], data['labels'])

        return res
