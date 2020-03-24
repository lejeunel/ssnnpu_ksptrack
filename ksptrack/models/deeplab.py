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
from ksptrack.siamese.modeling.xception import xception
from ksptrack.models.decoder import Decoder
from ksptrack.models.aspp import build_aspp
from os.path import join as pjoin


class DeepLabv3Plus(nn.Module):
    def __init__(self,
                 pretrained=False,
                 do_skip=False,
                 num_classes=3):

        super(DeepLabv3Plus, self).__init__()
        self.do_skip = do_skip

        self.encoder = drn.__dict__.get('drn_d_22')(pretrained=pretrained,
                                                    num_classes=1000)
        self.encoder.out_middle = True

        self.aspp = build_aspp('drn',
                               output_stride=8,
                               BatchNorm=nn.BatchNorm2d)
        self.aspp_out_dims = 256

        if(self.do_skip):
            self.shortcut_conv = nn.Sequential(
                            nn.Conv2d(256, 48, 1, 1,
                                    padding=1//2,
                                    bias=True),
                            nn.BatchNorm2d(48),
                            nn.ReLU(inplace=True))	

        self.decoder = Decoder(
            num_classes=num_classes,
            backbone='drn',
            BatchNorm=nn.BatchNorm2d,
            aspp_out_dims=256+48 if do_skip else 256)

        #init weights
        if(not pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x, layers = self.encoder(input)
        aspp_feats = self.aspp(x)

        if(self.do_skip):
            feature_shallow = self.shortcut_conv(layers[4])
            feature_cat = torch.cat([aspp_feats,
                                     feature_shallow],1)
            x = self.decoder(feature_cat)
        else:
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
            'layers': layers,
            'output': x,
            'aspp_feats': aspp_feats,
        }


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
