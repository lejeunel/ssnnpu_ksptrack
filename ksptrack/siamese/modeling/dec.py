import torch
import torch.nn as nn
from ksptrack.siamese.modeling.cluster import ClusterAssignment
from ksptrack.models.deeplab import DeepLabv3Plus
from ksptrack.siamese.modeling.dil_unet import UNet
from torchvision.ops import RoIPool
from ksptrack.siamese.modeling.superpixPool.pytorch_superpixpool.suppixpool_layer import SupPixPool


class SuperpixelPooling(nn.Module):
    """
    Given a RAG containing superpixel labels, this layer
    randomly samples couples of connected feature vector
    """
    def __init__(self, use_max=False):

        super(SuperpixelPooling, self).__init__()

        self.use_max = use_max
        if (use_max):
            self.pooling = lambda x: torch.max(x, dim=1)
        else:
            self.pooling = lambda x: torch.mean(x, dim=1)

    def pool(self, x, dim):
        if (self.use_max):
            return x.max(dim=dim)[0]

        return x.mean(dim=dim)

    def forward(self, x, label_maps):

        X = []

        for i, (x_, labels) in enumerate(zip(x, label_maps)):
            X_ = [
                self.pooling(x_[:, labels[0, ...] == l])
                for l in torch.unique(labels[0, ...])
            ]
            X.append(torch.stack(X_))

        return X


class RoIPooling(nn.Module):
    """
    Given a RAG containing superpixel labels, this layer
    randomly samples couples of connected feature vector
    """
    def __init__(self, output_size, spatial_scale=1.0, use_max=False):

        super(RoIPooling, self).__init__()

        self.output_size = output_size
        self.spatial_scale = spatial_scale

        self.roi_pool = RoIPool(output_size, spatial_scale)
        self.use_max = use_max
        if (use_max):
            self.pooling = lambda x: torch.max(x, dim=1)
        else:
            self.pooling = lambda x: torch.mean(x, dim=1)

    def pool(self, x, dim):
        if (self.use_max):
            return x.max(dim=dim)[0]

        return x.mean(dim=dim)

    def forward(self, x, rois):

        X = self.roi_pool(x, rois)
        X = X.squeeze()

        return X


class DEC(nn.Module):
    def __init__(self,
                 embedding_dims,
                 cluster_number: int = 30,
                 alpha: float = 1.0,
                 backbone='drn'):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension, input to the encoder
        :param hidden_dimension: hidden dimension, output of the encoder
        :param autoencoder: autoencoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """

        super(DEC, self).__init__()
        if(backbone == 'unet'):
            self.autoencoder = UNet(depth=4, skip_mode='none')
        else:
            self.autoencoder = DeepLabv3Plus()

        self.cluster_number = cluster_number
        self.alpha = alpha
        self.embedding_dims = embedding_dims

        # self.roi_pool = RoIPooling((roi_size, roi_size), roi_scale)
        # self.roi_pool = PrRoIPool2D(roi_size, roi_size, roi_scale)
        self.roi_pool = SupPixPool()

        self.transform = nn.Linear(self.autoencoder.last_feats_dims,
                                   embedding_dims, bias=False)
        self.assignment = ClusterAssignment(cluster_number,
                                            embedding_dims,
                                            alpha)

    def set_clusters(self, clusters, requires_grad=True):
        self.assignment = ClusterAssignment(clusters.shape[0],
                                            self.embedding_dims,
                                            self.alpha)
        clusters.requires_grad = requires_grad
        self.assignment.cluster_centers = nn.Parameter(clusters)

    def set_transform(self, transform, requires_grad=True):
        transform.requires_grad = requires_grad
        self.transform.weight = nn.Parameter(transform)

    def grad_linears(self, switch):
        for param in self.linear1.parameters():
            param.requires_grad = switch
        for param in self.linear2.parameters():
            param.requires_grad = switch

    def grad_dec(self, switch):
        for param in self.autoencoder.parameters():
            param.requires_grad = switch
        for param in self.assignment.parameters():
            param.requires_grad = switch

    def forward(self, data):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """

        res = self.autoencoder(data['image'])
        feats = res['feats']

        pooled_feats = [self.roi_pool(feats[b].unsqueeze(0),
                                      data['labels'][b].unsqueeze(0)).squeeze().T
                        for b in range(data['labels'].shape[0])]
        pooled_feats = torch.cat(pooled_feats)
        res.update({'pooled_feats': pooled_feats})

        proj_pooled_feats = self.transform(pooled_feats)
        res.update({'proj_pooled_feats': proj_pooled_feats})

        clusters = self.assignment(proj_pooled_feats)
        res.update({'clusters': clusters})

        return res
