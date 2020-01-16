import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ksptrack.models.dataset import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from imgaug import augmenters as iaa
from ksptrack.utils.my_augmenters import rescale_augmenter, Normalize
from ksptrack.utils import my_utils as utls
import os
import matplotlib.pyplot as plt
import numpy as np
from siamese_sp.modeling import drn
from siamese_sp.modeling.decoder import build_decoder
from siamese_sp.modeling.aspp import build_aspp

class DeepLabv3Plus(nn.Module):
    def __init__(self):

        super(DeepLabv3Plus, self).__init__()

        self.encoder = drn.__dict__.get('drn_d_22')(
            pretrained=True, num_classes=1000)
        self.encoder.out_middle = True
        self.aspp = build_aspp('drn', output_stride=8, BatchNorm=nn.BatchNorm2d)
        self.decoder = build_decoder(num_classes=3, backbone='drn', BatchNorm=nn.BatchNorm2d)

        self.sigmoid = nn.Sigmoid()

        self.low_level_feats_layer = 4

        self.feats_dim = 304

    def forward(self, input):
        x, low_level_feats = self.encoder(input)
        low_level_feat = low_level_feats[self.low_level_feats_layer]
        aspp_feats = self.aspp(x)
        x, cat_feats = self.decoder(aspp_feats, low_level_feat)
        x = F.interpolate(x,
                          size=input.size()[2:],
                          mode='bilinear',
                          align_corners=True)
        cat_feats = F.interpolate(cat_feats,
                                  size=input.size()[2:],
                                  mode='bilinear',
                                  align_corners=True)

        return x, cat_feats

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
    model = DeepLabv3()

    in_path = '/home/ubelix/lejeune/data/medical-labeling/Dataset00'
    locs_dir = 'gaze-measurements'
    csv_fname = 'video1.csv'
    frame_dir = 'input-frames'
    gt_dir = 'ground_truth-frames'
    sigma = 0.1
    cuda = False

    frame_fnames = utls.get_images(os.path.join(in_path, frame_dir))
    truth_fnames = utls.get_images(os.path.join(in_path, gt_dir))

    locs2d = utls.readCsv(os.path.join(in_path, locs_dir, csv_fname))
    transf = iaa.Sequential([rescale_augmenter,
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    ])
    dl = Dataset(im_paths=frame_fnames,
                truth_paths=truth_fnames,
                locs2d=locs2d,
                sig_prior=sigma,
                augmentations=transf)
    dataloader = DataLoader(dl,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=dl.collate_fn,
                            num_workers=0)

    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)
    model.to_feat_extractor()

    for e in range(10):
        for i, sample in enumerate(dataloader):

            im = sample['image'].to(device)
            truth = sample['label/segmentation'].to(device)
            prior = sample['prior'].to(device)

            out = model(im)

            import pdb; pdb.set_trace() ## DEBUG ##
            im_ = sample['image'][0].detach().cpu().numpy()
            im_ = np.rollaxis(im_, 0, 3)
            plt.imshow(im_);plt.show()
