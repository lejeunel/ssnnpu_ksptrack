import torch
import torch.nn as nn
from torch.nn import functional as F
from ksptrack.siamese.modeling.dec import DEC, RoIPooling
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import torch_geometric.utils as gutls
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool
from ksptrack.siamese.losses import structured_negative_sampling


class AppearanceBranch(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(AppearanceBranch, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.lin = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

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

        x = x / torch.clamp(x.norm(dim=1)[..., None], min=1e-8)

        return x


class LocMotionBranch(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(LocMotionBranch, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.lin = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

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

        pooled_fx = [
            self.roi_pool(flow.unsqueeze(0), labels).squeeze()[None, ...]
            for flow, labels in zip(data['fx'], data['labels'])
        ]

        pooled_fy = [
            self.roi_pool(flow.unsqueeze(0), labels).squeeze()[None, ...]
            for flow, labels in zip(data['fy'], data['labels'])
        ]

        n_samples = [s.shape[-1] for s in pooled_coords]
        time_idxs = [
            torch.tensor(n * [idx / (N - 1)]).unsqueeze(0).to(
                data['labels'].device) for n, idx, N in zip(
                    n_samples, data['frame_idx'], data['n_frames'])
        ]

        all_ = [
            torch.cat((t, fx, fy, c)) for t, fx, fy, c in zip(
                time_idxs, pooled_fx, pooled_fy, pooled_coords)
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

        x = x / torch.clamp(x.norm(dim=1)[..., None], min=1e-8)

        return x


class LocMotionAppearance(nn.Module):
    """

    """
    def __init__(self, in_channels, dropout=0.):
        """
        """
        super(LocMotionAppearance, self).__init__()
        self.appearance = AppearanceBranch(in_channels, dropout)
        self.locmotion = LocMotionBranch(5, dropout)

        self.dropout = nn.Dropout(dropout)
        self.nonlin = nn.ReLU()
        self.lin1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.lin2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)

    def forward(self, feats, data):
        locmotion = self.locmotion(data)
        appearance = self.appearance(feats)
        x = torch.stack((locmotion, appearance)).mean(dim=0)

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.nonlin(x)
        x = self.dropout(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.nonlin(x)
        x = self.dropout(x)

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

        self.locmotionapp = LocMotionAppearance(self.in_dims[-1], dropout=0.1)
        self.pw_clf = nn.Linear(2 * 16, 1)

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

    def forward(self, data, edges_list=None):

        res = self.dec(data)

        obj_pred = self.sp_pool(res['output'], data['labels'])
        res.update({'obj_pred': obj_pred})

        # feat and location/motion branches
        locmotionapp = self.locmotionapp(res['pooled_feats'], data)
        res.update({'locmotionapp': locmotionapp})

        if edges_list is not None:
            # relabel edges
            relabeled_edges_list = []
            max_node = 0
            for edges in edges_list:
                e_idx = edges.edge_index.clone()
                e_idx += max_node
                max_node = edges.n_nodes
                relabeled_edges_list.append(e_idx)

            relabeled_edges_list = torch.cat(relabeled_edges_list, dim=1)
            relabeled_edges_list.to(obj_pred.device)

            # do sampling
            i, j, k = structured_negative_sampling(relabeled_edges_list)

            # concatenate couples
            pos = torch.cat((locmotionapp[i], locmotionapp[j]), dim=1)
            neg = torch.cat((locmotionapp[i], locmotionapp[k]), dim=1)

            pw = self.sigmoid(self.pw_clf(locmotionapp))
            pw_pseudo_label = torch.cat(
                (torch.ones(pos.shape[0]),
                 torch.zeros(neg.shape[0]))).to(obj_pred.device)
            res.update({'pw': pw, 'pw_pseudo_label': pw_pseudo_label})

        return res
