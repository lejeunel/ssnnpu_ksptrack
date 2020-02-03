import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import operator
from ksptrack.models.my_augmenters import rescale_augmenter, Normalize
import im_utils as iutls
from loader import Loader
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from imgaug import augmenters as iaa
from itertools import chain
from sklearn.utils.class_weight import compute_class_weight
import random


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
                 dec,
                 embedding_dims):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.embedding_dims = embedding_dims

        self.dec = dec
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(self.embedding_dims,
                                 self.embedding_dims // 2,
                                 bias=True)
        self.linear2 = nn.Linear(self.embedding_dims // 2,
                                 1, bias=False)

    def forward(self, data, graph):

        import pdb; pdb.set_trace() ## DEBUG ##
        res = self.dec(data)

        # make (node0, node1, featdim) tensor for each edge in graph
        edges = torch.tensor([e for e in graph.edges()])
        X = torch.stack((res['pooled_aspp_feats'][0, edges[:, 0]],
                         res['pooled_aspp_feats'][1, edges[:, 1]]))

        out = []
        for br in range(2):
            x_ = self.linear1(X[br, ...])
            out.append(F.relu(x_))

        out = torch.abs(out[1] - out[0])
        out = self.linear2(out)
        out = self.sigmoid(out)

        out = out.squeeze()

        res = {}
        res['probas_preds'] = out

        return res

        return res


if __name__ == "__main__":
    path = '/home/ubelix/lejeune/data/medical-labeling/Dataset30/'
    transf = iaa.Sequential([rescale_augmenter])

    dl = Loader(path, augmentation=transf)

    dataloader_prev = DataLoader(dl,
                                 batch_size=2,
                                 shuffle=True,
                                 collate_fn=dl.collate_fn,
                                 num_workers=0)

    model = Siamese()

    for data in dataloader_prev:

        edges_to_pool = [[e for e in g.edges] for g in data['rag']]
        res = model(data['image'], data['rag'], data['labels'], edges_to_pool)
        fig = utls.make_grid_rag(
            data, [F.sigmoid(r) for r in res['similarities_labels']])

        # fig.show()
        fig.savefig('test.png', dpi=200)
        break
