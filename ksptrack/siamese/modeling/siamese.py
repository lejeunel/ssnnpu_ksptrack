import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling, sp_pool
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.modeling.dil_unet import ConvolutionalEncoder, ResidualBlock, DilatedConvolutions, ConvolutionalDecoder
from ksptrack.siamese.modeling.coordconv import CoordConv2d, AddCoords
from ksptrack.models.aspp import ASPP
import torch.nn.modules.conv as conv


@torch.no_grad()
def init_weights_xavier_normal(m):
    # if type(m) == nn.Linear:
    #     m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.xavier_normal_(m.weight)


@torch.no_grad()
def init_weights_xavier_uniform(m):
    # if type(m) == nn.Linear:
    #     m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.xavier_uniform_(m.weight)


def make_coord_map(batch_size, w, h):
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
        self.conv1 = nn.Conv1d(in_dims, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, out_dims, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_dims)
        self.lin = nn.Linear(out_dims, out_dims)

    def forward(self, coords_flows, x, x_pred):
        x = torch.cat((coords_flows, x, x_pred), dim=1).T[None, ...]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.squeeze().T
        x = F.relu(self.lin(x))

        return x


class LocMotionAppearanceUNet(nn.Module):
    """

    """
    def __init__(self, dropout_reduc=0, pos_components=4):
        """
        """
        super(LocMotionAppearanceUNet, self).__init__()

        self.mergers = []

        batchNormObject = lambda n_features: nn.GroupNorm(16, n_features)

        self.encoder = ConvolutionalEncoder(depth=5,
                                            in_channels=3,
                                            start_filts=16,
                                            dropout_min=0,
                                            dropout_max=0,
                                            use_aspp=True,
                                            convObject=nn.Conv2d,
                                            batchNormObject=batchNormObject)

        self.bn_coords = nn.Sequential(
            *[nn.BatchNorm2d(pos_components),
              nn.ReLU()])

        self.merger = Merger(512 + pos_components, 128)

    def forward(self, data, autoenc_feats):

        x = data['image']

        for stage in self.encoder.stages:
            x = stage(x)

        coord_maps = make_coord_map(
            data['image'].shape[0], data['image'].shape[2],
            data['image'].shape[3]).to(data['labels'].device)
        coords_flows = torch.cat((coord_maps, data['fx'], data['fy']), dim=1)
        coords_flows = self.bn_coords(coords_flows)
        siam_feats = self.merger(sp_pool(coords_flows, data['labels']),
                                 sp_pool(x, data['labels']),
                                 sp_pool(autoenc_feats, data['labels']))

        del coords_flows

        return {
            'siam_feats': siam_feats,
            'pos': sp_pool(coord_maps, data['labels'])
        }

    def sync(self, encoder_, from_=1):

        for i in range(from_, len(encoder_.stages)):
            self.encoder.stages[i].load_state_dict(
                encoder_.stages[i].state_dict())


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
                 siamese='gcn',
                 skip_mode='aspp',
                 max_edges=10000,
                 dropout_fg=0,
                 dropout_adj=0.1,
                 dropout_reduc=0,
                 out_channels=3):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.siamese = siamese
        assert (siamese in ['gcn', 'unet', 'none',
                            'gcnunet']), 'siamese must be gcn ,unet, or none'

        self.dec = DEC(embedding_dims=embedded_dims,
                       cluster_number=cluster_number,
                       alpha=alpha,
                       backbone=backbone,
                       out_channels=out_channels)

        self.sigmoid = nn.Sigmoid()
        self.roi_pool = SupPixPool()

        self.rho_dec = ConvolutionalDecoder(out_channels=1,
                                            dropout_max=dropout_fg,
                                            start_filts=16,
                                            skip_mode='aspp',
                                            depth=5)

        self.cs_sigma = nn.Parameter(torch.tensor(1.))

        if (siamese == 'gcn'):
            in_dims = self.dec.autoencoder.filts_dims
            in_dims += [in_dims[-1]]
            self.locmotionapp = LocMotionAppearanceSigned(
                in_dims=in_dims,
                dropout_reduc=dropout_reduc,
                max_edges=max_edges)
        elif (siamese == 'unet'):
            self.locmotionapp = LocMotionAppearanceUNet(
                dropout_reduc=dropout_reduc)
        else:
            self.locmotionapp = nn.Identity()

        self.cs = nn.CosineSimilarity()

        # initialize weights
        # self.dec.autoencoder.encoder.apply(init_weights_xavier_normal)
        self.rho_dec.apply(init_weights_xavier_normal)
        self.locmotionapp.apply(init_weights_xavier_normal)

    def load_partial(self, state_dict):
        # filter out unnecessary keys
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict, strict=False)

    def forward(self, data, edges_nn=None):

        x, skips = self.dec.autoencoder.encoder(data['image'])
        feats = x
        pooled_feats = sp_pool(feats, data['labels'])
        # pooled_feats = F.normalize(pooled_feats, p=2, dim=1)
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
        res['rho_hat_pooled'] = sp_pool(rho_hat, data['labels'])
        res['rho_hat'] = rho_hat

        return res
