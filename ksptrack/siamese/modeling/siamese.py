import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.losses import structured_negative_sampling
from ksptrack.siamese.modeling.dil_unet import ConvolutionalEncoder, ResidualBlock, DilatedConvolutions
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


class LocMotionAppearanceGCN(nn.Module):
    """

    """
    def __init__(self, in_channels, filt_dims, dropout=0., mixing_coeff=0.):
        """
        """
        super(LocMotionAppearanceGCN, self).__init__()
        self.filt_dims = filt_dims
        self.sigma = 0.2

        in_dim = in_channels
        self.gcns = []
        for out_dim in filt_dims:
            gcn = gnn.GCNConv(in_dim, out_dim)
            self.gcns.append(gcn)
            in_dim = out_dim

        # cosine classification block
        self.lin1 = nn.Linear(256, 15, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.mixing_coeff = mixing_coeff
        self.roi_pool = SupPixPool()

    def forward(self, data, edges_nn, probas, feats):

        # add self loops
        edges, _ = gutls.add_self_loops(edges_nn)
        weights = torch.exp(-(probas[edges[0]] - probas[edges[1]]).abs() /
                            self.sigma)

        Z = sp_pool(data['image'], data['labels'])
        Z = F.relu(self.gcns[0](Z, edges, weights))

        for i, gcn in enumerate(self.gcns[1:]):
            # do GCNConv and average result with encoder features
            H_pool = sp_pool(feats[i], data['labels'])
            Z = self.mixing_coeff * H_pool + (1 - self.mixing_coeff) * Z
            Z = F.relu(gcn(Z, edges, weights))

        cs_r = F.normalize(Z, p=2, dim=1)

        # l2-normalize weights
        with torch.no_grad():
            self.lin1.weight.div_(
                torch.norm(self.lin1.weight, p=2, dim=1, keepdim=True))
        cs = self.lin1(cs_r)

        return cs, cs_r


class LocMotionAppearance(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0., mixing_coeff=0.):
        """
        """
        super(LocMotionAppearance, self).__init__()
        self.model = ConvolutionalEncoder(in_channels=5,
                                          start_filts=32,
                                          kernel_size=3,
                                          padding=1,
                                          depth=4,
                                          n_resblocks=1,
                                          dropout_min=0,
                                          dropout_max=0,
                                          blockObject=ResidualBlock,
                                          convObject=CoordConv2d,
                                          batchNormObject=nn.BatchNorm2d)
        self.dilatedConvs = DilatedConvolutions(256,
                                                n_convolutions=3,
                                                dropout=0.,
                                                convObject=CoordConv2d)

        # 128 output size?
        self.lin1 = nn.Linear(256, 15, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)
        self.mixing_coeff = mixing_coeff
        self.roi_pool = SupPixPool()

    def forward(self, data, dec_feats):

        # make location/motion input tensor
        batch_size, _, w, h = data['labels'].shape

        input = torch.cat((data['image'], data['fx'], data['fy']), dim=1)

        # get features
        x, skips = self.model(input)
        x, dilated_skips = self.dilatedConvs(x)
        for d in dilated_skips:
            x += d
        x += skips[-1]

        #upsample and pool to superpixels
        upsamp = nn.UpsamplingBilinear2d(data['labels'].size()[2:])
        x = upsamp(x)
        x = [
            self.roi_pool(x[b].unsqueeze(0), data['labels'][b]).squeeze().T
            for b in range(data['labels'].shape[0])
        ]
        x = torch.cat(x)
        x = F.normalize(x, p=2, dim=1)

        # merge features
        cs_r = self.mixing_coeff * dec_feats + (1 - self.mixing_coeff) * x
        cs_r = F.normalize(cs_r, p=2, dim=1)

        # l2-normalize weights
        with torch.no_grad():
            self.lin1.weight.div_(
                torch.norm(self.lin1.weight, p=2, dim=1, keepdim=True))
        cs = self.lin1(cs_r)

        return cs, cs_r


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

        if (backbone == 'drn'):
            self.in_dims = [3, 32, 64, 128, 256]
        else:
            self.in_dims = [3] + self.dec.autoencoder.filts_dims

        self.roi_pool = SupPixPool()

        self.locmotionapp = LocMotionAppearanceGCN(5,
                                                   self.in_dims,
                                                   dropout=0.1,
                                                   mixing_coeff=0.2)

    def load_partial(self, state_dict):
        # filter out unnecessary keys
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict, strict=False)

    def forward(self, data):

        res = self.dec(data)

        obj_pred = sp_pool(res['output'], data['labels'])
        res.update({'obj_pred': obj_pred})

        # feat and location/motion branches
        cs, cs_r = self.locmotionapp(data, res['pooled_feats'])
        res.update({'cs': cs})
        res.update({'cs_r': cs_r})

        return res
