import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ksptrack.pu.modeling.coordconv import CoordConv2d


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
    if (type(m) == nn.Conv2d or type(m) == CoordConv2d):
        nn.init.kaiming_normal_(m.weight)


def auto_reshape(im, depth):
    w, h = im.shape[2:]
    do_resamp = False

    if w / (2**(depth - 1)) % 1 != 0:
        w = int(np.ceil(w / (2**(depth - 1))) * 2**(depth - 1))
        do_resamp = True
    if h / (2**(depth - 1)) % 1 != 0:
        h = int(np.ceil(h / (2**(depth - 1))) * 2**(depth - 1))
        do_resamp = True

    if do_resamp:
        resamp = nn.UpsamplingBilinear2d((w, h))
        im = resamp(im)

    return im


class MultiResBlock(nn.Module):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''
    def __init__(self,
                 in_channels,
                 num_filters,
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d):
        super(MultiResBlock, self).__init__()

        self.alpha = 1.67
        w = self.alpha * num_filters
        out_filt_shortcut = int(w * 0.167) + int(w * 0.333) + int(w * 0.5)
        out_filt_conv3x3 = int(w * 0.167)
        out_filt_conv5x5 = int(w * 0.333)
        out_filt_conv7x7 = int(w * 0.5)

        self.out_filts = out_filt_conv3x3 + out_filt_conv5x5 + out_filt_conv7x7

        self.shortcut = nn.Sequential(*[
            convObject(in_channels, out_filt_shortcut, 1),
        ])
        self.conv3x3 = nn.Sequential(*[
            convObject(in_channels, out_filt_conv3x3, 3, padding=1),
            nn.ReLU()
        ])
        self.conv5x5 = nn.Sequential(*[
            convObject(out_filt_conv3x3, out_filt_conv5x5, 3, padding=1),
            nn.ReLU(),
        ])
        self.conv7x7 = nn.Sequential(*[
            convObject(out_filt_conv5x5, out_filt_conv7x7, 3, padding=1),
            nn.ReLU(),
        ])

        self.bn0 = batchNormObject(out_filt_conv3x3 + out_filt_conv5x5 + \
                                   out_filt_conv7x7)
        self.bn1 = batchNormObject(out_filt_conv3x3 + out_filt_conv5x5 + \
                                   out_filt_conv7x7)

    def forward(self, input):
        shortcut = self.shortcut(input)
        conv3x3out = self.conv3x3(input)
        conv5x5out = self.conv5x5(conv3x3out)
        conv7x7out = self.conv7x7(conv5x5out)
        out = torch.cat((conv3x3out, conv5x5out, conv7x7out), dim=1)
        out = self.bn0(out)
        out += shortcut
        out = self.bn1(out)
        out = F.relu(out)
        return out


class ResidualPath(nn.Module):
    """
    Residual Path
    """
    def __init__(self,
                 num_filters,
                 length,
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d):
        super(ResidualPath, self).__init__()

        self.length = length
        self.shortcuts = nn.ModuleList([
            nn.Sequential(*[
                convObject(num_filters, num_filters, kernel_size=1),
                batchNormObject(num_filters)
            ]) for _ in range(length)
        ])

        self.convs = nn.ModuleList([
            nn.Sequential(*[
                convObject(num_filters, num_filters, kernel_size=3, padding=1),
                batchNormObject(num_filters),
                nn.ReLU()
            ]) for _ in range(length)
        ])
        self.bns = nn.ModuleList(
            [batchNormObject(num_filters) for _ in range(length)])

    def forward(self, x):
        for c, s, b in zip(self.convs, self.shortcuts, self.bns):
            x = b(F.relu(c(x) + s(x)))

        return x


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional Encoder providing skip connections
    """
    def __init__(self,
                 in_channels=3,
                 start_filts=32,
                 depth=5,
                 convObject=nn.Conv2d):
        """
        n_features_input (int): number of intput features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalEncoder, self).__init__()
        self.in_channels = in_channels

        self.filts_dims = [start_filts * (2**i) for i in range(depth)]

        self.stages = nn.ModuleList()

        # input convolution block
        block = [MultiResBlock(3, self.filts_dims[0], convObject=convObject)]
        self.stages.append(nn.Sequential(*block))

        # layers
        for i in range(len(self.filts_dims) - 1):
            # downsampling
            block = [
                nn.MaxPool2d(2),
                MultiResBlock(self.stages[i][-1].out_filts,
                              self.filts_dims[i + 1],
                              convObject=convObject)
            ]
            self.stages.append(nn.Sequential(*block))

    def forward(self, x):

        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)

        return x, skips

    def getInputShape(self):
        return (-1, self.n_features_input, -1, -1)

    def getOutputShape(self):
        return (-1, self.num_hidden_features[-1], -1, -1)


class ConvolutionalDecoder(nn.Module):
    """
    Convolutional Decoder taking skip connections
    """
    def __init__(self,
                 in_dims,
                 start_filts,
                 out_channels=3,
                 skip_mode='res',
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d):
        """
        n_features_output (int): number of output features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalDecoder, self).__init__()

        if skip_mode in ('none', 'res'):
            self.skip_mode = skip_mode
        else:
            raise ValueError(
                "\"{}\" is not a valid mode for"
                "merging up and down paths. "
                "Only \"none\", and \"res\" are allowed.".format(skip_mode))

        self.depth = 5
        self.out_channels = out_channels
        self.upConvolutions = nn.ModuleList()
        self.skipMergers = nn.ModuleList()
        self.skip_conns = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.upconv_out_dims = [
            start_filts * (2**i) for i in range(len(in_dims))
        ][::-1]
        # layers
        for i in range(len(in_dims) - 1):
            # downsampling
            self.upConvolutions.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dims[i],
                                       self.upconv_out_dims[i],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    batchNormObject(self.upconv_out_dims[i]), nn.ReLU()))

            self.skipMergers.append(nn.Identity())

            if self.skip_mode == 'res':
                self.skip_conns.append(
                    ResidualPath(in_dims[i + 1], length=i + 1))
                # residual blocks
                self.blocks.append(
                    MultiResBlock(in_dims[i + 1] + self.upconv_out_dims[i],
                                  self.upconv_out_dims[i + 1]))
            else:
                self.skip_conns.append(nn.Identity())
                # residual blocks
                self.blocks.append(
                    MultiResBlock(self.upconv_out_dims[i],
                                  self.upconv_out_dims[i + 1]))

        # output convolution block
        block = [
            convObject(in_dims[-1], out_channels, kernel_size=1, stride=1)
        ]
        self.output_convolution = nn.Sequential(*block)

    def forward(self, x, skips):
        for up, merge, skip_con, conv, skip in zip(self.upConvolutions,
                                                   self.skipMergers,
                                                   self.skip_conns,
                                                   self.blocks, skips):
            x = up(x)

            if (self.skip_mode == 'res'):
                skip = skip_con(skip)
                cat = torch.cat([x, skip], 1)
                x = merge(cat)
            x = conv(x)
        return self.output_convolution(x)

    def getInputShape(self):
        return (-1, self.num_hidden_features[0], -1, -1)

    def getOutputShape(self):
        return (-1, self.n_features_output, -1, -1)


class UNet(nn.Module):
    """
    U-Net model with dynamic number of layers, Residual Blocks
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 depth=5,
                 start_filts=32,
                 skip_mode='res',
                 use_coordconv=False):
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=3)
            out_channels (int): number of output channels (n_classes)
            num_hidden_features (list(int)): number of hidden features for each layer (the number of layer is the lenght of this list)
            padding (int): padding for the convolutions
            group_norm (bool): number of groups to use for Group Normalization, default is 32, if zero: use nn.BatchNorm2d
        """
        super(UNet, self).__init__()

        self.convObject = CoordConv2d if use_coordconv else nn.Conv2d

        self.filts_dims = [start_filts * (2**i) for i in range(depth)]
        self.last_feats_dims = self.filts_dims[-1]
        batchNormObject = nn.BatchNorm2d

        self.depth = depth

        self.encoder = ConvolutionalEncoder(in_channels=in_channels,
                                            start_filts=start_filts,
                                            depth=depth)

        filts_dims = [stage[-1].out_filts
                      for stage in self.encoder.stages][::-1]

        self.decoder = ConvolutionalDecoder(filts_dims,
                                            start_filts=start_filts,
                                            out_channels=out_channels,
                                            skip_mode=skip_mode)

        self.apply(init_weights)

    def forward(self, x):

        x = auto_reshape(x, self.depth)

        in_shape = x.shape
        x, skips = self.encoder(x)

        feats = x

        x = self.decoder(x, skips[:-1][::-1])
        x = F.interpolate(x,
                          size=in_shape[2:],
                          mode='bilinear',
                          align_corners=True)

        return {'output': x, 'feats': feats, 'skips': skips}

    def to_predictor(self):

        in_dim = self.decoder.filts_dims[-1]
        kernel_size = self.decoder.output_convolution[0].kernel_size
        padding = self.decoder.output_convolution[0].padding
        stride = self.decoder.output_convolution[0].stride
        new_out = self.convObject(in_dim, 1, kernel_size, stride, padding)
        self.decoder.output_convolution = torch.nn.Sequential(new_out)
