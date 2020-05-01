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
                 in_channels,
                 dropout_min=0.,
                 dropout_max=0.2,
                 mixing_coeff=0.):
        """
        """
        super(LocMotionAppearance, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.lin1 = nn.Linear(in_features=256, out_features=256)

        self.conv1_merge = nn.Conv1d(in_channels=512,
                                     out_channels=512,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.lin1_merge = nn.Linear(in_features=512, out_features=256)
        self.lin2_merge = nn.Linear(in_features=256, out_features=128)

        self.lin1_dist = nn.Linear(128, 15, bias=False)
        self.nonlin = nn.CELU()
        self.dropout = nn.Dropout(dropout_max)

        self.lin_pred = nn.Linear(128, 1)
        self.apply(init_weights)

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

    def forward(self, data, dec_feats):

        # make location/motion input tensor
        batch_size, _, w, h = data['labels'].shape

        coord_maps = self.make_coord_map(batch_size, w,
                                         h).to(data['labels'].device)

        pooled_coords = sp_pool(coord_maps, data['labels'])
        pooled_fx = sp_pool(data['fx'], data['labels'])[..., None]
        pooled_fy = sp_pool(data['fy'], data['labels'])[..., None]

        n_samples = [torch.unique(l).numel() for l in data['labels']]
        time_idxs = torch.cat([
            torch.tensor(n * [idx / (N - 1)]).to(data['labels'].device) for n,
            idx, N in zip(n_samples, data['frame_idx'], data['n_frames'])
        ])[..., None]

        x = torch.cat((time_idxs, pooled_coords, pooled_fx, pooled_fy),
                      dim=1).T[None, ...]
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.lin1(x.squeeze(0).T)
        x = self.nonlin(x)
        x = self.dropout(x)

        x = F.normalize(x, p=2, dim=1)
        x = torch.cat((x, dec_feats), dim=1)

        x = self.conv1_merge(x.T.unsqueeze(0))
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.lin1_merge(x.squeeze(0).T)
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.lin2_merge(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        rho_hat = self.lin_pred(x)
        rho_hat = self.dropout(rho_hat)

        x = F.normalize(x, p=2, dim=1)

        # l2-normalize weights
        with torch.no_grad():
            self.lin1_dist.weight.div_(
                torch.norm(self.lin1_dist.weight, p=2, dim=1, keepdim=True))
        cs = self.lin1_dist(x)
        # cs = self.bn1(cs)
        cs = self.dropout(cs)

        return cs, x, rho_hat


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

        self.locmotionapp = LocMotionAppearance(5,
                                                dropout_min=0,
                                                dropout_max=0.2,
                                                mixing_coeff=0.2)

    def load_partial(self, state_dict):
        # filter out unnecessary keys
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict, strict=False)

    def forward(self, data):

        res = self.dec(data['image'], data['labels'])

        # feat and location/motion branches
        cs, cs_r, rho_hat = self.locmotionapp(data, res['pooled_feats'])

        res.update({'cs': cs})
        res.update({'cs_r': cs_r})
        res.update({'rho_hat': rho_hat})

        return res
