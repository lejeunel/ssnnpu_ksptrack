import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ksptrack.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, aspp_out_dims=256):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.relu = nn.ReLU()

        self.aspp_out_dims = aspp_out_dims
        cat_dims = self.aspp_out_dims

        self.last_conv = nn.Sequential(
            nn.Conv2d(cat_dims,
                      256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), BatchNorm(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                      bias=True), BatchNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()
        self.feats_dim = cat_dims

    def forward(self, x):

        feats = F.interpolate(x,
                              scale_factor=4,
                              mode='bilinear',
                              align_corners=True)
        x = self.last_conv(feats)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
