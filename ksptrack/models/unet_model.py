import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            bias=True,
            groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1,
        padding=0,
        groups=groups, stride=1)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False)
        return x


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.modules = [self.conv1, self.bn1, self.conv2, self.bn2]

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.modules.append(self.pool)

    def train(self):
        for m in self.modules:
            m.training = True

    def eval(self):
        for m in self.modules:
            m.training = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

    def forward_no_last_relu(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='none',
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(
            self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        elif (self.merge_mode == 'add' or self.merge_mode == 'none'):
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.modules = [self.conv1, self.bn1, self.conv2, self.bn2]

    def train(self):
        for m in self.modules:
            m.train()

    def eval(self):
        for m in self.modules:
            m.eval()

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        elif self.merge_mode == 'add':
            x = from_up + from_down
        else:
            x = from_up

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
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
                 in_channels=3,
                 out_channels=3,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add', 'none'):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                "\"{}\" is not a valid mode for"
                "merging up and down paths. "
                "Only \"concat\", "
                "\"add\", and \"none\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        outs = None
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(
                ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.sigmoid = nn.Sigmoid()

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def train(self):
        for m in [self.down_convs, self.up_convs]:
            m.training = True

    def eval(self):
        for m in [self.down_convs, self.up_convs]:
            m.training = False

    def state_dict(self):
        return {'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'start_filts': self.start_filts,
                'depth': self.depth,
                'up_mode': self.up_mode,
                'merge_mode': self.merge_mode,
                'down_convs': self.down_convs.state_dict(),
                'up_convs': self.up_convs.state_dict(),
                'conv_final': self.conv_final.state_dict()}

    def load_state_dict(self, dict_):
        self.down_convs.load_state_dict(dict_['down_convs'])
        self.up_convs.load_state_dict(dict_['up_convs'])
        self.conv_final.load_state_dict(dict_['conv_final'])

    @classmethod
    def from_state_dict(cls, dict_):
        model = cls(dict_['in_channels'],
                     dict_['out_channels'],
                     depth=dict_['depth'],
                     start_filts=dict_['start_filts'],
                     up_mode=dict_['up_mode'],
                     merge_mode=dict_['merge_mode'],
                     cuda=False)
        model.load_state_dict(dict_)
        return model

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)

        # pass through sigmoid
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    """
    testing
    """
    model = UNet(3, depth=5, merge_mode='concat')
    x = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
