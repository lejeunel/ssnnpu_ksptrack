import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool


class AppearanceBranch(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.2):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(AppearanceBranch, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.lin = nn.Linear(in_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)

        self.roi_pool = SupPixPool()

    def forward(self, x):

        # pass through layers
        x = self.conv1(x.unsqueeze(-1))
        x = self.bn1(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        x = self.lin(x.squeeze())
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        # x = x / x.norm(dim=0)
        x = x / (x.norm(dim=1)[..., None] + 1e-8)

        return x


class LocMotionBranch(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.2):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(LocMotionBranch, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 128, 1)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.lin = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.roi_pool = SupPixPool()

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

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        out = torch.cat([xx_channel, yy_channel], dim=1)

        return out

    def forward(self, data):

        # make location/motion input tensor
        batch_size, _, w, h = data['labels'].shape
        coord_maps = self.make_coord_map(batch_size, w,
                                         h).to(data['labels'].device)
        pooled_coords = [
            self.roi_pool(coord_maps[b].unsqueeze(0),
                          data['labels'][b]).squeeze()
            for b in range(data['labels'].shape[0])
        ]

        pooled_flows = [
            self.roi_pool(flow.unsqueeze(0), labels).squeeze()[None, ...]
            for flow, labels in zip(data['flow'], data['labels'])
        ]

        n_samples = [s.shape[-1] for s in pooled_coords]
        time_idxs = [
            torch.tensor(n * [idx / (N - 1)]).unsqueeze(0).to(
                data['labels'].device) for n, idx, N in zip(
                    n_samples, data['frame_idx'], data['n_frames'])
        ]

        all_ = [
            torch.cat((t, f, c))
            for t, f, c in zip(time_idxs, pooled_flows, pooled_coords)
        ]

        x = torch.cat(all_, dim=-1).unsqueeze(0).T

        # pass through layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        x = self.lin(x.squeeze())
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        x = x / (x.norm(dim=1)[..., None] + 1e-8)

        return x


class SiameseModule(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.2):
        """
        """
        super(SiameseModule, self).__init__()
        self.appearance = AppearanceBranch(in_channels, dropout)
        self.locmotion = LocMotionBranch(4, dropout)

        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.lin1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, feats, data):
        locmotion = self.locmotion(data)
        appearance = self.appearance(feats)
        x = torch.cat((locmotion, appearance), dim=1)
        x = x / (x.norm(dim=1)[..., None] + 1e-8)

        x = self.lin1(x)
        x = self.bn1(x)
        # x = self.nonlin(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.bn2(x)
        # x = self.nonlin(x)
        # x = self.dropout(x)

        x = x / (x.norm(dim=1)[..., None] + 1e-8)

        return x


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

        self.dec = DEC(embedded_dims, cluster_number, alpha, backbone)

        self.sigmoid = nn.Sigmoid()

        if (backbone == 'drn'):
            self.in_dims = [3, 32, 64, 128, 256]
        else:
            self.in_dims = [3] + self.dec.autoencoder.filts_dims

        self.roi_pool = SupPixPool()

        self.siamese = SiameseModule(self.in_dims[-1])

    def load_partial(self, state_dict):
        # filter out unnecessary keys
        state_dict = {
            k: v
            for k, v in state_dict.items() if k in self.state_dict()
        }
        self.load_state_dict(state_dict, strict=False)

    def sp_pool(self, feats, labels):

        upsamp = nn.UpsamplingBilinear2d(labels.size()[2:])
        pooled_feats = [
            self.roi_pool(upsamp(feats[b].unsqueeze(0)),
                          labels[b].unsqueeze(0)).squeeze().T
            for b in range(labels.shape[0])
        ]
        pooled_feats = torch.cat(pooled_feats)
        return pooled_feats

    def forward(self, data):

        res = self.dec(data)

        obj_pred = self.sp_pool(res['output'], data['labels'])
        res.update({'obj_pred': obj_pred})

        # feat and location/motion branches
        locmotionapp = self.siamese(res['pooled_feats'], data)
        res.update({'locmotionapp': locmotionapp})

        return res
