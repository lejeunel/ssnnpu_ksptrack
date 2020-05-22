import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.losses import structured_negative_sampling
from ksptrack.siamese.modeling.dil_unet import ConvolutionalEncoder, ResidualBlock, DilatedConvolutions, init_weights, ConvolutionalDecoder
from ksptrack.siamese.modeling.coordconv import CoordConv2d


def num_nan_inf(t):
    return torch.isnan(t).sum() + torch.isinf(t).sum()


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


class SignedGCNBlock(nn.Module):
    """

    """
    def __init__(self, in_dims, out_dims, first_aggr=False):
        super(SignedGCNBlock, self).__init__()
        self.gcn = gnn.SignedConv(in_channels=in_dims,
                                  out_channels=out_dims,
                                  first_aggr=first_aggr)
        self.bn = nn.BatchNorm1d(2 * out_dims)
        self.relu = nn.ReLU()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x = self.gcn(x,
                     pos_edge_index=pos_edge_index,
                     neg_edge_index=neg_edge_index)
        x = self.bn(x)
        x = self.relu(x)
        return x


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
        self.mergers_pos = []
        self.mergers_neg = []
        self.sigma = sigma
        self.mixing_coeff = mixing_coeff

        in_dims[0] = 36
        for i in range(len(in_dims) - 1):
            block = SignedGCNBlock(in_dims[i],
                                   in_dims[i + 1],
                                   first_aggr=True if i == 0 else False)

            self.gcns.append(block)

            if (i > 0):
                merger_pos = nn.Sequential(*[
                    nn.Linear(2 * in_dims[i], in_dims[i]),
                    nn.BatchNorm1d(in_dims[i]),
                    nn.ReLU()
                ])
                merger_neg = nn.Sequential(*[
                    nn.Linear(2 * in_dims[i], in_dims[i]),
                    nn.BatchNorm1d(in_dims[i]),
                    nn.ReLU()
                ])
                self.mergers_pos.append(merger_pos)
                self.mergers_neg.append(merger_neg)

        self.bn_pre = nn.BatchNorm1d(4)
        self.gcns = nn.ModuleList(self.gcns)
        self.mergers_pos = nn.ModuleList(self.mergers_pos)
        self.mergers_neg = nn.ModuleList(self.mergers_neg)

        self.lin_pw = nn.Linear(512, 256)
        # self.bn_pw = nn.BatchNorm1d(1024)

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

        x = sp_pool(autoenc_skips[0], data['labels'])
        x = torch.cat((pooled_coords, pooled_fx, pooled_fy), dim=1)
        x = self.bn_pre(x)
        x = F.relu(x)

        # print('[LocMotionApp] lvl {} num. NaN feats: {}'.format(
        # 'pre', num_nan_inf(x)))
        pos_edge_index = edges_nn[:, edges_nn[-1] != -1][:2]
        neg_edge_index = edges_nn[:, edges_nn[-1] == -1][:2]

        # print('[LocMotionApp] lvl {} num. NaN feats: {}'.format(
        #     0, num_nan_inf(x)))
        # x = F.normalize(x, dim=1)

        for i, g in enumerate(self.gcns):
            skip = sp_pool(autoenc_skips[i], data['labels'])
            # print('[LocMotionApp] lvl {} num. NaN feats: {}'.format(
            #     i + 1, num_nan_inf(x)))
            if (i > 0):
                x_pos = x[:, x.shape[1] // 2:]
                x_neg = x[:, :x.shape[1] // 2]
                x_pos = torch.cat((x_pos, skip), dim=1)
                x_pos = self.mergers_pos[i - 1](x_pos)
                x_neg = torch.cat((x_neg, skip), dim=1)
                x_neg = self.mergers_neg[i - 1](x_neg)
                x = torch.cat((x_pos, x_neg), dim=1)
            else:
                x = torch.cat((x, skip), dim=1)

            x = g(x,
                  pos_edge_index=pos_edge_index,
                  neg_edge_index=neg_edge_index)

        # x = torch.cat((x_pos, x_neg), dim=1)
        x = self.lin_pw(x)
        # print('[LocMotionApp] lvl {} num. NaN feats: {}'.format(
        #     'last', num_nan_inf(x)))
        x = F.relu(x)

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

        self.rho_dec = ConvolutionalDecoder(out_channels=1,
                                            dropout_max=0.1,
                                            start_filts=32,
                                            skip_mode='conv',
                                            depth=4)

        in_dims = self.dec.autoencoder.filts_dims
        if (self.dec.autoencoder.dilatedConvs is not None):
            in_dims += [in_dims[-1]]
        self.locmotionapp = LocMotionAppearance(in_dims=in_dims,
                                                dropout_min=0,
                                                dropout_max=0.2,
                                                mixing_coeff=0.5)

        self.rho_dec.apply(init_weights)
        # self.locmotionapp.apply(init_weights)

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

        res['rho_hat'] = self.rho_dec(res['feats'], res['skips'][:-2][::-1])
        res['rho_hat_pooled'] = sp_pool(res['rho_hat'], data['labels'])

        return res
