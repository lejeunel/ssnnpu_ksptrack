import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from imgaug import augmenters as iaa
from ksptrack.utils.my_augmenters import rescale_augmenter, Normalize
from ksptrack.utils import my_utils as utls
import os
import matplotlib.pyplot as plt
import numpy as np
from ksptrack.models import drn
from ksptrack.models.decoder import Decoder
from ksptrack.models.aspp import build_aspp


class DeepLabv3Plus(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):

        super(DeepLabv3Plus, self).__init__()

        self.encoder = drn.__dict__.get('drn_d_22')(pretrained=pretrained,
                                                    num_classes=1000)
        self.encoder.out_middle = True
        self.aspp = build_aspp('drn',
                               output_stride=8,
                               BatchNorm=nn.BatchNorm2d)
        self.aspp_out_dims = 256

        self.decoder = Decoder(
            num_classes=num_classes,
            backbone='drn',
            BatchNorm=nn.BatchNorm2d,
            aspp_out_dims=256)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x, _ = self.encoder(input)
        aspp_feats = self.aspp(x)
        x = self.decoder(aspp_feats)
        x = F.interpolate(x,
                          size=input.size()[2:],
                          mode='bilinear',
                          align_corners=True)
        aspp_feats = F.interpolate(aspp_feats,
                                   size=input.size()[2:],
                                   mode='bilinear',
                                   align_corners=True)

        return {
            'output': x,
            'aspp_feats': aspp_feats,
        }

    def to_autoenc(self, in_channels=3, out_channels=3):
        self.decoder.last_conv[-1] = nn.Conv2d(256,
                                               out_channels,
                                               kernel_size=(1, 1),
                                               stride=(1, 1))

    def to_predictor(self, out_channels=1):
        self.model.last_conv[-1] = nn.Conv2d(256,
                                             out_channels,
                                             kernel_size=(1, 1),
                                             stride=(1, 1))


class DeepLabv3(nn.Module):
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N, 3, H, W), where N is the number of images, H and W are expected to be at least 224 pixels. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    The model returns an OrderedDict with two Tensors that are of the same height and width as the input Tensor, but with 21 classes. output['out'] contains the semantic masks, and output['aux'] contains the auxillary loss values per-pixel. In inference mode, output['aux'] is not useful. So, output['out'] is of shape (N, 21, H, W). More documentation can be found here.
    """
    def __init__(self):
        """
        """
        super(DeepLabv3, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.4.2',
                                    'deeplabv3_resnet101',
                                    pretrained=True)

        self.to_autoenc()
        self.sigmoid = nn.Sigmoid()

    def to_autoenc(self, in_channels=3, out_channels=3):
        self.model.aux_classifier[-1] = nn.Conv2d(256,
                                                  out_channels,
                                                  kernel_size=(1, 1),
                                                  stride=(1, 1))
        self.model.classifier[-1] = nn.Conv2d(256,
                                              out_channels,
                                              kernel_size=(1, 1),
                                              stride=(1, 1))

    def to_feat_extractor(self):
        self.model.aux_classifier = nn.Identity()
        self.model.classifier[-1] = nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def to_predictor(self, out_channels=1):
        self.model.aux_classifier[-1] = nn.Conv2d(256,
                                                  out_channels,
                                                  kernel_size=(1, 1),
                                                  stride=(1, 1))

    def forward(self, x):
        dict_ = self.model(x)
        dict_['aux'] = self.sigmoid(dict_['aux'])
        dict_['out'] = self.sigmoid(dict_['out'])

        return dict_


if __name__ == "__main__":
    model = DeepLabv3Plus()

    in_path = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'
    cuda = False

    transf = iaa.Sequential([
        rescale_augmenter,
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dl = LocPriorDataset(root_path=in_path, augmentations=transf)
    dataloader = DataLoader(dl,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=dl.collate_fn,
                            num_workers=0)

    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)

    for e in range(10):
        for i, sample in enumerate(dataloader):

            im = sample['image'].to(device)

            out = model(im)

            im_ = sample['image'][0].detach().cpu().numpy()
            im_ = np.rollaxis(im_, 0, 3)
            plt.imshow(im_)
            plt.show()
