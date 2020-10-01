import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling, sp_pool
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.modeling.dil_unet import ConvolutionalEncoder, ConvolutionalDecoder, auto_reshape
from ksptrack.siamese.modeling.coordconv import CoordConv2d, AddCoords
from ksptrack.siamese.loader import linearize_labels
from ksptrack.models.aspp import ASPP
import torch.nn.modules.conv as conv
from ksptrack.siamese.cosface import MarginCosineProduct


@torch.no_grad()
def init_weights_xavier_normal(m):
    # if type(m) == nn.Linear:
    #     m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.xavier_normal_(m.weight)


@torch.no_grad()
def init_kaiming_normal(m):
    # if type(m) == nn.Linear:
    #     m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.kaiming_normal_(m.weight)


@torch.no_grad()
def init_weights_xavier_uniform(m):
    # if type(m) == nn.Linear:
    #     m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.xavier_uniform_(m.weight)


def make_coord_map(batch_size, w, h, return_tuple=False):
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

    if return_tuple:
        return xx_channel, yy_channel
    else:
        return torch.cat([xx_channel, yy_channel], dim=1)


def num_nan_inf(t):
    return torch.isnan(t).sum() + torch.isinf(t).sum()


class GCNBlock(nn.Module):
    """

    """
    def __init__(self, in_dims, out_dims, first_aggr=False):
        super(GCNBlock, self).__init__()
        self.gcn = gnn.GCNConv(in_channels=in_dims, out_channels=out_dims)
        self.bn = nn.BatchNorm1d(out_dims)
        self.relu = nn.ReLU()

    def forward(self, x, pos_edge_index, neg_edge_index):
        weights_pos = torch.ones(pos_edge_index.shape[1]).to(x.device).float()
        weights_neg = torch.zeros(neg_edge_index.shape[1]).to(x.device).float()
        edge_index = torch.cat((pos_edge_index, neg_edge_index), dim=1)
        weights = torch.cat((weights_pos, weights_neg))
        x = self.gcn(x, edge_index=edge_index, edge_weight=weights)
        x = self.bn(x)
        x = self.relu(x)
        return x


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


def downsample_edges(pos_edge_index, neg_edge_index, max_edges):

    if (pos_edge_index.shape[1] > max_edges):
        bc = torch.bincount(pos_edge_index[-1])
        weights = bc[pos_edge_index[-1]]
        weights = weights.max().float() / weights.float()
        idx_pos = torch.multinomial(weights, max_edges, replacement=True)
        pos_edge_index = pos_edge_index[:, idx_pos]
    if (neg_edge_index.shape[1] > max_edges):
        idx = torch.randint(neg_edge_index.shape[-1], (max_edges, ))
        neg_edge_index = neg_edge_index[:, idx]

    return pos_edge_index, neg_edge_index


class Merger(nn.Module):
    """

    """
    def __init__(self, in_dims, out_dims):
        """

        """
        super(Merger, self).__init__()
        self.lin_ = nn.Linear(in_dims, out_dims, bias=False)

    def forward(self, x):

        x = self.lin_(x)
        x = F.relu(x)

        return x


class LocMotionAppearance(nn.Module):
    """

    """
    def __init__(self, pos_components=4, skip_dims=256, out_features=15):
        """
        """
        super(LocMotionAppearance, self).__init__()

        self.mergers = []

        self.bn_coords = nn.BatchNorm1d(pos_components)
        self.coords1 = nn.Sequential(*[
            nn.Conv1d(pos_components, 32, kernel_size=1),
            nn.Dropout(p=0.1),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU()
        ])
        self.bn_reduc = nn.BatchNorm1d(skip_dims)
        self.skip_reduc1 = nn.Sequential(*[
            nn.Conv1d(skip_dims, 512, kernel_size=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ])

        self.merger2 = nn.Sequential(nn.Dropout(p=0.1),
                                     nn.Linear(512 + 64, 256))
        self.margin_cos = MarginCosineProduct(256, out_features, s=10., m=0.2)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, data, autoenc_feats):

        labels = linearize_labels(data['labels'])
        device = data['labels'].device

        xx, yy = make_coord_map(data['image'].shape[0],
                                data['image'].shape[2],
                                data['image'].shape[3],
                                return_tuple=True)
        xx = sp_pool(xx.to(device), labels)
        yy = sp_pool(yy.to(device), labels)
        fx = sp_pool(data['fx'], labels)
        fy = sp_pool(data['fy'], labels)

        coords_flows = torch.stack((xx, yy, fx, fy), dim=1)
        coords_flows = self.bn_coords(coords_flows)
        coords_flows = self.coords1(coords_flows.T.unsqueeze(0))

        pooled_skips = sp_pool(autoenc_feats, labels)
        pooled_skip = self.bn_reduc(pooled_skips.T.unsqueeze(0))
        red_skips = self.skip_reduc1(pooled_skip)
        siam_feats = torch.cat(
            (coords_flows.squeeze().T, red_skips.squeeze().T), dim=1)
        siam_feats = self.merger2(siam_feats)
        # siam_feats = siam_feats.squeeze().T

        res = {'siam_feats': siam_feats, 'pos': torch.stack((xx, yy)).T}

        # if 'clusters' in data.keys():
        #     res['cosprod'] = self.margin_cos(siam_feats, data['clusters'][-1])

        return res


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
                 backbone='unet',
                 dropout_fg_min=0,
                 dropout_fg_max=0,
                 dropout_adj=0.1,
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
                       out_channels=out_channels,
                       dropout_min=dropout_fg_min,
                       dropout_max=dropout_fg_max)

        self.sigmoid = nn.Sigmoid()
        self.roi_pool = SupPixPool()

        self.rho_dec = ConvolutionalDecoder(out_channels=1,
                                            dropout_min=dropout_fg_min,
                                            dropout_max=dropout_fg_max,
                                            start_filts=16,
                                            skip_mode='aspp',
                                            depth=5)

        self.locmotionapp = LocMotionAppearance(out_features=cluster_number)

        # initialize weights
        self.rho_dec.apply(init_kaiming_normal)
        self.locmotionapp.apply(init_kaiming_normal)

    def forward(self, data, edges_nn=None):
        w, h = data['image'].shape[:2]

        data['image'] = auto_reshape(data['image'], self.dec.autoencoder.depth)

        x, skips = self.dec.autoencoder.encoder(data['image'])
        feats = x
        labels = linearize_labels(data['labels'])
        pooled_feats = sp_pool(feats, labels)

        res = dict()
        res['pooled_feats'] = pooled_feats
        res['feats'] = feats
        res['skips'] = skips + [feats]

        proj_pooled_feats = self.dec.transform(pooled_feats)
        res['proj_pooled_feats'] = proj_pooled_feats

        clusters = self.dec.assignment(proj_pooled_feats)
        res['clusters'] = clusters

        if (not isinstance(self.locmotionapp, nn.Identity)):
            res_siam = self.locmotionapp(data, res['skips'][-1])
            res.update(res_siam)

        rho_hat = self.rho_dec(res['feats'], res['skips'][:-2][::-1])
        res['rho_hat_pooled'] = sp_pool(rho_hat, labels)
        res['rho_hat'] = rho_hat

        coord_xx, coord_yy = make_coord_map(data['image'].shape[0],
                                            data['image'].shape[2],
                                            data['image'].shape[3],
                                            return_tuple=True)

        coord_xx = coord_xx.to(data['labels'].device)
        coord_yy = coord_yy.to(data['labels'].device)

        res['pos_x'] = sp_pool(coord_xx, labels)
        res['pos_y'] = sp_pool(coord_yy, labels)

        return res
