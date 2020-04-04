import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ksptrack.siamese.modeling.coordconv import CoordConv2d

@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self,
                 num_filters,
                 kernel_size,
                 padding,
                 nonlinearity=nn.ReLU,
                 dropout=0.2,
                 dilation=1,
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        num_hidden_filters = num_filters

        self.conv1 = convObject(num_filters,
                                num_hidden_filters,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                dilation=dilation)
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.conv2 = convObject(num_hidden_filters,
                                num_hidden_filters,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=padding,
                                dilation=dilation)
        self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        out = og_x + x
        out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional Encoder providing skip connections
    """
    def __init__(self,
                 in_channels=3,
                 n_resblocks=3,
                 start_filts=64,
                 depth=5,
                 kernel_size=3,
                 padding=1,
                 dropout_min=0,
                 dropout_max=0.2,
                 blockObject=ResidualBlock,
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d):
        """
        n_features_input (int): number of intput features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalEncoder, self).__init__()
        self.in_channels = in_channels

        self.filts_dims = [start_filts*(2**i) for i in range(depth)]

        self.stages = nn.ModuleList()
        dropout = [(1 - t) * dropout_min + t * dropout_max
                   for t in np.linspace(0, 1, depth)]
        # input convolution block
        block = [
            convObject(in_channels,
                      self.filts_dims[0],
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding)
        ]
        for _ in range(n_resblocks):
            block += [
                blockObject(self.filts_dims[0],
                            kernel_size,
                            padding,
                            dropout=dropout[0],
                            batchNormObject=batchNormObject)
            ]
        self.stages.append(nn.Sequential(*block))

        # layers
        for i in range(len(self.filts_dims) - 1):
            # downsampling
            block = [
                nn.MaxPool2d(2),
                convObject(self.filts_dims[i], self.filts_dims[i+1], kernel_size=1, padding=0),
                batchNormObject(self.filts_dims[i+1]),
                nn.ReLU()
            ]
            # residual blocks
            for _ in range(n_resblocks):
                block += [
                    blockObject(self.filts_dims[i+1],
                                kernel_size,
                                padding,
                                dropout=dropout[i+1],
                                batchNormObject=batchNormObject)
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
                 out_channels=3,
                 n_resblocks=3,
                 start_filts=64,
                 depth=5,
                 kernel_size=3,
                 padding=1,
                 dropout_min=0,
                 dropout_max=0.2,
                 blockObject=ResidualBlock,
                 convObject=nn.Conv2d,
                 batchNormObject=nn.BatchNorm2d,
                 skip_mode='conv'):
        """
        n_features_output (int): number of output features
        num_hidden_features (list(int)): number of features for each stage
        kernel_size (int): convolution kernel size
        padding (int): convolution padding
        n_resblocks (int): number of residual blocks at each stage
        dropout (float): dropout probability
        blockObject (nn.Module): Residual block to use. Default is ResidualBlock
        batchNormObject (nn.Module): normalization layer. Default is nn.BatchNorm2d
        """
        super(ConvolutionalDecoder, self).__init__()

        if skip_mode in ('conv', 'none'):
            self.skip_mode = skip_mode
        else:
            raise ValueError(
                "\"{}\" is not a valid mode for"
                "merging up and down paths. "
                "Only \"concat\", and \"none\" are allowed.".format(skip_mode))

        self.out_channels = out_channels
        self.filts_dims = [start_filts*(2**i) for i in range(depth)][::-1]
        self.upConvolutions = nn.ModuleList()
        self.skipMergers = nn.ModuleList()
        self.residualBlocks = nn.ModuleList()
        dropout = [(1 - t) * dropout_min + t * dropout_max
                   for t in np.linspace(0, 1, depth)][::-1]
        # layers
        for i in range(len(self.filts_dims) - 1):
            # downsampling
            self.upConvolutions.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.filts_dims[i],
                                       self.filts_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    batchNormObject(self.filts_dims[i+1]), nn.ReLU()))
            if(self.skip_mode == 'conv'):
                self.skipMergers.append(
                    convObject(2 * self.filts_dims[i+1],
                            self.filts_dims[i+1],
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding))
            else:
                self.skipMergers.append(nn.Identity())
            # residual blocks
            block = []
            p = next(iter(dropout))
            for _ in range(n_resblocks):
                block += [
                    blockObject(self.filts_dims[i+1],
                                kernel_size,
                                padding,
                                dropout=p,
                                batchNormObject=batchNormObject)
                ]
            self.residualBlocks.append(nn.Sequential(*block))
        # output convolution block
        block = [
            convObject(self.filts_dims[-1],
                      out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding)
        ]
        self.output_convolution = nn.Sequential(*block)

    def forward(self, x, skips):
        for up, merge, conv, skip in zip(self.upConvolutions, self.skipMergers,
                                         self.residualBlocks, skips):
            x = up(x)

            if(self.skip_mode == 'conv'):
                cat = torch.cat([x, skip], 1)
                x = merge(cat)
            x = conv(x)
        return self.output_convolution(x)

    def getInputShape(self):
        return (-1, self.num_hidden_features[0], -1, -1)

    def getOutputShape(self):
        return (-1, self.n_features_output, -1, -1)


class DilatedConvolutions(nn.Module):
    """
    Sequential Dilated convolutions
    """
    def __init__(self, n_channels, n_convolutions, dropout,
                 convObject=nn.Conv2d):
        super(DilatedConvolutions, self).__init__()
        kernel_size = 3
        padding = 1
        self.dropout = nn.Dropout2d(dropout)
        self.non_linearity = nn.ReLU(inplace=True)
        self.strides = [2**(k + 1) for k in range(n_convolutions)]
        convs = [
            convObject(n_channels,
                      n_channels,
                      kernel_size=kernel_size,
                      dilation=s,
                      padding=s) for s in self.strides
        ]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in convs:
            self.convs.append(c)
            self.bns.append(nn.BatchNorm2d(n_channels))

    def forward(self, x):
        skips = []
        for (c, bn, s) in zip(self.convs, self.bns, self.strides):
            x_in = x
            x = c(x)
            x = bn(x)
            x = self.non_linearity(x)
            x = self.dropout(x)
            x = x_in + x
            skips.append(x)
        return x, skips



class UNet(nn.Module):
    """
    U-Net model with dynamic number of layers, Residual Blocks, Dilated Convolutions, Dropout and Group Normalization
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 depth=5,
                 start_filts=32,
                 n_resblocks=1,
                 num_dilated_convs=3,
                 dropout_min=0,
                 dropout_max=0,
                 coordconv=True,
                 padding=1,
                 skip_mode='conv',
                 kernel_size=3,
                 group_norm=32):
        """
        initialize the model
        Args:
            in_channels (int): number of input channels (image=3)
            out_channels (int): number of output channels (n_classes)
            num_hidden_features (list(int)): number of hidden features for each layer (the number of layer is the lenght of this list)
            n_resblocks (int): number of residual blocks at each layer 
            num_dilated_convs (int): number of dilated convolutions at the last layer
            dropout (float): float in [0,1]: dropout probability
            padding (int): padding for the convolutions
            kernel_size (int): kernel size for the convolutions
            group_norm (bool): number of groups to use for Group Normalization, default is 32, if zero: use nn.BatchNorm2d
        """
        super(UNet, self).__init__()

        if coordconv:
            convObject = CoordConv2d
        else:
            convObject = nn.Conv2d

        self.filts_dims = [start_filts*(2**i) for i in range(depth)]
        self.last_feats_dims = self.filts_dims[-1]
        if group_norm > 0:
            for h in self.filts_dims:
                assert h % group_norm == 0, "Number of features at each layer must be divisible by 'group_norm'"
        batchNormObject = lambda n_features: nn.GroupNorm(
            group_norm, n_features) if group_norm > 0 else nn.BatchNorm2d

        self.encoder = ConvolutionalEncoder(in_channels=in_channels,
                                            start_filts=start_filts,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            depth=depth,
                                            n_resblocks=n_resblocks,
                                            dropout_min=dropout_min,
                                            dropout_max=dropout_max,
                                            blockObject=ResidualBlock,
                                            convObject=convObject,
                                            batchNormObject=batchNormObject)
        if num_dilated_convs > 0:
            self.dilatedConvs = DilatedConvolutions(
                self.filts_dims[-1], num_dilated_convs,
                dropout_max, convObject=convObject)
        else:
            self.dilatedConvs = None
        self.decoder = ConvolutionalDecoder(out_channels=out_channels,
                                            start_filts=start_filts,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            depth=depth,
                                            n_resblocks=n_resblocks,
                                            dropout_min=dropout_min,
                                            dropout_max=dropout_max,
                                            skip_mode=skip_mode,
                                            blockObject=ResidualBlock,
                                            convObject=convObject,
                                            batchNormObject=batchNormObject)

        self.apply(init_weights)

    def forward(self, x):
        in_shape = x.shape
        x, skips = self.encoder(x)
        if self.dilatedConvs is not None:
            x, dilated_skips = self.dilatedConvs(x)
            for d in dilated_skips:
                x += d
            x += skips[-1]
        feats = x
        x = self.decoder(x, skips[:-1][::-1])
        x = F.interpolate(x,
                          size=in_shape[2:],
                          mode='bilinear',
                          align_corners=True)
        feats = F.interpolate(feats,
                              size=in_shape[2:],
                              mode='bilinear',
                              align_corners=True)

        return {'output': x,
                'feats': feats,
                'layers': skips}

    def to_predictor(self):
        
        in_dim = self.decoder.output_convolution[0].in_channels
        kernel_size = self.decoder.output_convolution[0].kernel_size
        padding = self.decoder.output_convolution[0].padding
        stride = self.decoder.output_convolution[0].stride
        new_out = torch.nn.Conv2d(
            in_dim, 1, kernel_size, stride, padding)
        self.decoder.output_convolution = torch.nn.Sequential(new_out)

