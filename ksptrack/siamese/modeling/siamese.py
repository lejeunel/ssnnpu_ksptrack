import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import operator
from siamese_sp.my_augmenters import rescale_augmenter
from siamese_sp import utils as utls
from siamese_sp.loader import Loader
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
from imgaug import augmenters as iaa
from itertools import chain
from sklearn.utils.class_weight import compute_class_weight
import random



def compute_weights(edges):
    edges_ = np.array(edges)
    labels = np.unique(edges_[:, -1])
    weights = compute_class_weight('balanced', labels, edges_[:, -1])

    weights_ = np.zeros(edges_.shape[0])
    if (labels.size < 2):
        weights_[edges_[:, -1] == labels[0]] = weights[0]
    else:
        weights_[edges_[:, -1] == 0] = weights[0]
        weights_[edges_[:, -1] == 1] = weights[1]

    return weights_ / weights_.sum()


class SuperpixelPooling(nn.Module):
    """
    Given a RAG containing superpixel labels, this layer
    randomly samples couples of connected feature vector
    """
    def __init__(self, n_samples, balanced=False, use_max=False):

        super(SuperpixelPooling, self).__init__()

        self.n_samples = n_samples
        self.balanced = balanced
        self.use_max = use_max
        if (use_max):
            self.pooling = lambda x: torch.max(x, dim=1)
        else:
            self.pooling = lambda x: torch.mean(x, dim=1)

    def pool(self, x, dim):
        if (self.use_max):
            return x.max(dim=dim)[0]

        return x.mean(dim=dim)

    def forward(self, x, graphs, label_maps, edges_to_pool=None):

        X = [[] for _ in range(len(x))]
        Y = [[] for _ in range(len(x))]

        for i, (x_, g) in enumerate(zip(x, graphs)):
            if (edges_to_pool is None):
                edges = [(e[0], e[1], g.edges[e]['weight']) for e in g.edges]
                if (self.balanced):
                    weights = compute_weights(edges)
                else:
                    weights = None
                edges = [
                    edges[i] for i in np.random.choice(
                        len(edges), self.n_samples, p=weights)
                ]
            else:
                edges = edges_to_pool[i]

            for e in edges:
                X[i].append([
                    self.pool(x[i, ..., label_maps[i, 0, ...] == e[0]], dim=1),
                    self.pool(x[i, ..., label_maps[i, 0, ...] == e[1]], dim=1)
                ])
                Y[i].append([torch.tensor(e[-1]).float().to(x)])

        X = [(torch.stack([x_[0] for x_ in x],
                          dim=0), torch.stack([x_[1] for x_ in x], dim=0))
             for x in X]
        Y = [torch.stack([y_[0] for y_ in y], dim=0) for y in Y]
        return X, Y


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
                 in_channels=304,
                 start_filts=64,
                 n_edges=100,
                 with_batchnorm=False,
                 with_oflow=False,
                 balanced=True):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
        """
        super(Siamese, self).__init__()

        self.balanced = balanced
        self.n_edges = n_edges
        self.with_oflow = with_oflow

        self.in_channels = in_channels

        self.dec = dec
        self.sigmoid = nn.Sigmoid()

        # freeze weights of feature extractor (encoder + ASPP)
        # for param in chain(self.autoenc.encoder.parameters(),
        #                    self.autoenc.aspp.parameters()):
        #     param.requires_grad = False

        self.in_channels = in_channels

        self.linear1 = nn.Linear(self.in_channels,
                                 self.in_channels // 2,
                                 bias=True)
        self.linear2 = nn.Linear(self.in_channels // 2,
                                 1, bias=False)

    def grad_linears(self, switch):
        self.linear1.parameters().requires_grad = switch
        self.linear2.parameters().requires_grad = switch

    def grad_dec(self, switch):
        self.dec.parameters().requires_grad = switch

    def calc_probas(self, x):

        batch, branch, feats_dim, n_edges = x.shape

        # iterate on branches of siamese network
        res = []
        for br in range(branch):
            x_ = self.linear1(x[:, br, ...])
            res.append(F.relu(x_))

        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        res = self.sigmoid(res)

        res = res.squeeze()

        return res

    def forward(self, x, graphs, label_maps):

        res = self.dec(x, label_maps)

        X, Y = utls.sample_batch(graphs, res['clusters'],
                                 res['feats'],
                                 self.n_edges)

        edge_probas = self.calc_probas(X)

        res['similarities'] = edge_probas
        res['similarities_labels'] = Y
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
