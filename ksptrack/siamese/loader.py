import os
from os.path import join as pjoin
from skimage import future, measure
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
from tqdm import tqdm
import pandas as pd
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.siamese import utils as utls
from ksptrack.siamese.modeling.siamese import Siamese
from skimage.future import graph as skg
import random
import networkx as nx
from torch._six import int_classes as _int_classes


class RandomBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, size, batch_size, shuffle=False):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle

    def __iter__(self):
        batch = []
        starts = list(range(self.size - (self.batch_size - 1)))
        if self.shuffle:
            random.shuffle(starts)

        for start in starts:
            for i in range(self.batch_size):
                batch.append(start + i)
            yield batch
            batch = []

    def __len__(self):
        return self.size


class Loader(LocPriorDataset):
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=None,
                 resize_shape=None,
                 csv_fname='video1.csv',
                 labels_fname='sp_labels.npz',
                 sig_prior=0.1,
                 nn_radius=0.1):
        """

        """
        super().__init__(root_path=root_path,
                         augmentations=augmentations,
                         normalization=normalization,
                         resize_shape=resize_shape,
                         csv_fname=csv_fname,
                         sig_prior=sig_prior)

        self.nn_radius = nn_radius

        self.hoof = pd.read_pickle(pjoin(root_path, 'precomp_desc', 'hoof.p'))

        graphs_path = pjoin(root_path, 'precomp_desc', 'siam_graphs.npz')
        if (not os.path.exists(graphs_path)):
            self.prepare_graphs()
            np.savez(graphs_path, **{'graphs': self.graphs})
        else:
            print('loading graphs at {}'.format(graphs_path))
            np_file = np.load(graphs_path, allow_pickle=True)
            self.graphs = np_file['graphs']

        self.labels = np.load(pjoin(root_path, 'precomp_desc',
                                    'sp_labels.npz'))['sp_labels']

        npzfile = np.load(pjoin(root_path, 'precomp_desc', 'flows.npz'))
        flows = dict()
        flows['bvx'] = npzfile['bvx']
        flows['fvx'] = npzfile['fvx']
        flows['bvy'] = npzfile['bvy']
        flows['fvy'] = npzfile['fvy']
        fx = np.rollaxis(
            np.concatenate((flows['fvx'], flows['fvx'][..., -1][..., None]),
                           axis=-1), -1, 0)
        fy = np.rollaxis(
            np.concatenate((flows['fvy'], flows['fvy'][..., -1][..., None]),
                           axis=-1), -1, 0)
        fv = np.sqrt(fx**2 + fy**2)
        self.fv = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fv]
        self.fx = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fx]
        self.fy = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fy]

    def prepare_graphs(self):

        self.graphs = []

        print('preparing graphs...')

        pbar = tqdm(total=len(self.imgs))
        for idx, (im, truth) in enumerate(zip(self.imgs, self.truths)):
            labels = self.labels[..., idx]
            graph = skg.RAG(label_image=labels)
            self.graphs.append(graph)
            pbar.update(1)
        pbar.close()

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        sample['graph'] = self.graphs[idx]

        fnorm = self.fv[idx][..., None]
        fnorm = self.reshaper_img.augment_image(fnorm)
        sample['fnorm'] = fnorm

        fx = self.fx[idx][..., None]
        fx = self.reshaper_img.augment_image(fx)
        sample['fx'] = fx

        fy = self.fy[idx][..., None]
        fy = self.reshaper_img.augment_image(fy)
        sample['fy'] = fy

        return sample

    def collate_fn(self, samples):
        out = super(Loader, Loader).collate_fn(samples)

        out['graph'] = [s['graph'] for s in samples]

        fnorm = [np.rollaxis(d['fnorm'], -1) for d in samples]
        fnorm = torch.stack([torch.from_numpy(f) for f in fnorm]).float()
        out['fnorm'] = fnorm

        fx = [np.rollaxis(d['fx'], -1) for d in samples]
        fx = torch.stack([torch.from_numpy(f) for f in fx]).float()
        out['fx'] = fx

        fy = [np.rollaxis(d['fy'], -1) for d in samples]
        fy = torch.stack([torch.from_numpy(f) for f in fy]).float()
        out['fy'] = fy

        return out


class StackLoader(LocPriorDataset):
    def __init__(self,
                 root_path,
                 depth=2,
                 augmentations=None,
                 normalization=None,
                 resize_shape=None,
                 csv_fname='video1.csv',
                 labels_fname='sp_labels.npz',
                 sig_prior=0.05,
                 nn_radius=0.1):
        """

        """
        super(StackLoader, self).__init__(root_path=root_path,
                                          augmentations=augmentations,
                                          normalization=normalization,
                                          resize_shape=resize_shape,
                                          csv_fname=csv_fname,
                                          sig_prior=sig_prior)

        self.nn_radius = nn_radius
        self.depth = depth

        self.hoof = pd.read_pickle(pjoin(root_path, 'precomp_desc', 'hoof.p'))

        graphs_path = pjoin(root_path, 'precomp_desc',
                            'graphs_depth_{}.npz'.format(self.depth))
        if (not os.path.exists(graphs_path)):
            self.prepare_graphs()
            np.savez(graphs_path, **{
                'centroids': self.centroids,
                'graphs': self.graphs
            })
        else:
            print('loading graphs at {}'.format(graphs_path))
            np_file = np.load(graphs_path, allow_pickle=True)
            self.centroids = np_file['centroids']
            self.graphs = np_file['graphs']

        self.labels = np.load(pjoin(root_path, 'precomp_desc',
                                    'sp_labels.npz'))['sp_labels']

        npzfile = np.load(pjoin(root_path, 'precomp_desc', 'flows.npz'))
        flows = dict()
        flows['bvx'] = npzfile['bvx']
        flows['fvx'] = npzfile['fvx']
        flows['bvy'] = npzfile['bvy']
        flows['fvy'] = npzfile['fvy']
        self.fx = np.rollaxis(
            np.concatenate((flows['fvx'], flows['fvx'][..., -1][..., None]),
                           axis=-1), -1, 0)
        self.fy = np.rollaxis(
            np.concatenate((flows['fvy'], flows['fvy'][..., -1][..., None]),
                           axis=-1), -1, 0)
        self.fv = np.sqrt(self.fx**2 + self.fy**2)
        # self.fv = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fv]
        # self.fx = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fx]
        # self.fy = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in fy]

    def prepare_graphs(self):

        self.centroids = []
        self.graphs = []

        print('preparing graphs...')
        pbar = tqdm(total=len(self.imgs) - (self.depth - 1))
        for i in range(self.labels.shape[-1] - self.depth + 1):
            labels = np.copy(self.labels[..., i:i + self.depth])
            for d in range(1, self.depth):
                labels[..., d] += labels[..., d - 1].max() + 1
            graph = skg.RAG(label_image=labels)
            self.graphs.append(graph)

            labels = self.labels[..., i:i + self.depth].copy()
            centroids = []
            for d in range(self.depth):
                regions = measure.regionprops(labels[..., d] + 1)
                centroids_ = np.array([
                    (p['centroid'][1] / labels[..., d].shape[1],
                     p['centroid'][0] / labels[..., d].shape[0])
                    for p in regions
                ])
                centroids.append(centroids_)
            self.centroids.append(np.concatenate(centroids))
            pbar.update(1)
        pbar.close()

    def __getitem__(self, idx):

        samples = [
            super(StackLoader, self).__getitem__(i)
            for i in range(idx, idx + self.depth)
        ]

        max_node = 0
        clicked = []
        for i in range(self.depth):
            fnorm = self.fv[idx + i][..., None]
            fnorm = self.reshaper_img.augment_image(fnorm)
            samples[i]['fnorm'] = fnorm

            fx = self.fx[idx + i][..., None]
            fx = self.reshaper_img.augment_image(fx)
            samples[i]['fx'] = fx

            fy = self.fy[idx + i][..., None]
            fy = self.reshaper_img.augment_image(fy)
            samples[i]['fy'] = fy

            clicked.append(np.array(samples[i]['labels_clicked']) + max_node)
            max_node += samples[i]['labels'].max() + 1

        clicked = np.concatenate(clicked)

        return samples, self.centroids[idx], self.graphs[idx], clicked

    def __len__(self):
        return len(self.imgs) - (self.depth - 1)
        # return len(self.imgs)

    def collate_fn(self, samples):
        out = dict()
        out['centroids'] = samples[0][1]
        out['graph'] = samples[0][2]
        clicked = samples[0][3]

        samples = samples[0][0]
        out_ = super(Loader, Loader).collate_fn(samples)
        out.update(out_)

        fnorm = [np.rollaxis(d['fnorm'], -1) for d in samples]
        fnorm = torch.stack([torch.from_numpy(f) for f in fnorm]).float()
        out['fnorm'] = fnorm

        fx = [np.rollaxis(d['fx'], -1) for d in samples]
        fx = torch.stack([torch.from_numpy(f) for f in fx]).float()
        out['fx'] = fx

        fy = [np.rollaxis(d['fy'], -1) for d in samples]
        fy = torch.stack([torch.from_numpy(f) for f in fy]).float()
        out['fy'] = fy

        out['clicked'] = torch.from_numpy(clicked)

        return out


if __name__ == "__main__":

    dset = StackLoader(root_path=pjoin(
        '/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset30'),
                       normalization='rescale',
                       depth=2,
                       resize_shape=512)

    device = torch.device('cuda')
    model = Siamese(embedded_dims=15,
                    cluster_number=15,
                    alpha=1,
                    backbone='unet').to(device)

    dl = DataLoader(dset, collate_fn=dset.collate_fn)
    labels_pos = dict()
    n_labels = dict()
    for s in dl:
        for i, f in enumerate(s['frame_idx']):
            labels_pos[f] = s['labels_clicked'][i]
            n_labels[f] = torch.unique(s['labels'][i]).numel()

    labels_pos_bool = []
    for f in sorted(n_labels.keys()):
        labels_pos_ = np.zeros(n_labels[f]).astype(bool)
        labels_pos_[labels_pos[f]] = True
        labels_pos_bool.append(labels_pos_)
