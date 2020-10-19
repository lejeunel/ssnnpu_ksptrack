import os
from os.path import join as pjoin
from skimage import future, measure
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
from torch.utils import data
from tqdm import tqdm
import pandas as pd
from ksptrack.utils.loc_prior_dataset import LocPriorDataset, loc_apply_augs
from ksptrack.utils.base_dataset import make_normalizer
from skimage.future import graph as skg
import random
import networkx as nx
from torch._six import int_classes as _int_classes
import pickle
from imgaug import augmenters as iaa
import imgaug as ia
import warnings


def apply_augs(im,
               labels,
               truth,
               keypoints,
               loc_prior,
               fx,
               fy,
               fnorm,
               aug,
               min_fx=-2,
               max_fx=2,
               min_fy=-2,
               max_fy=2):

    warnings.simplefilter("ignore", UserWarning)

    aug_norescale = iaa.Sequential(aug[:-1])
    fnorm = ia.HeatmapsOnImage(fnorm,
                               shape=fnorm.shape,
                               min_value=min_fx,
                               max_value=max_fx)
    fx = ia.HeatmapsOnImage(fx,
                            shape=fx.shape,
                            min_value=min_fx,
                            max_value=max_fx)
    fy = ia.HeatmapsOnImage(fy,
                            shape=fy.shape,
                            min_value=min_fy,
                            max_value=max_fy)

    fnorm = aug_norescale(heatmaps=fnorm).get_arr()
    fx = aug_norescale(heatmaps=fx).get_arr()
    fy = aug_norescale(heatmaps=fy).get_arr()

    fnorm = np.squeeze(fnorm)[..., None]
    fx = np.squeeze(fx)[..., None]
    fy = np.squeeze(fy)[..., None]
    im, labels, truth, keypoints, loc_prior = loc_apply_augs(
        im, labels, truth, keypoints, loc_prior, aug)

    return im, labels, truth, keypoints, loc_prior, fx, fy, fnorm


def linearize_labels(label_map, labels=None):
    """
    Converts a batch of label maps and labels each numbered starting from 0
    to have unique labels on the whole batch

    label_map: list of label maps
    labels: list of labels
    """
    if (labels is not None):
        assert (len(label_map) == len(labels)
                ), print('label_map and labels must have same length')

    new_labels = label_map.clone()

    max_label = 0
    for b in range(len(label_map)):

        new_labels[b] += max_label
        max_label += label_map[b].max() + 1
        if labels is not None:
            labels[b] = [l + max_label for l in labels[b]]

    if labels is not None:
        return new_labels, labels
    else:
        return new_labels


def delinearize_labels(label_map, labels):
    """
    Converts a batch of label maps and labels each with unique values to
    a batch where each start at 0

    label_map: list of label maps
    labels: list of labels
    """
    assert (len(label_map) == len(labels)
            ), print('label_map and labels must have same length')
    curr_min_label = 0
    for b in range(len(label_map)):

        label_map[b] -= curr_min_label
        labels[b] = [l - curr_min_label for l in labels[b]]
        curr_min_label -= label_map[b].max() + 1

    return label_map, labels


class Loader(LocPriorDataset):
    def __init__(self,
                 root_path,
                 augmentations=None,
                 normalization=None,
                 resize_shape=None,
                 csv_fname='video1.csv',
                 sp_labels_fname='sp_labels.npy',
                 sig_prior=0.1):
        """

        """
        super().__init__(root_path=root_path,
                         csv_fname=csv_fname,
                         sp_labels_fname=sp_labels_fname,
                         sig_prior=sig_prior)

        self.fvx = np.load(pjoin(root_path, 'precomp_desc', 'flows_fvx.npy'),
                           mmap_mode='r')
        self.fvy = np.load(pjoin(root_path, 'precomp_desc', 'flows_fvy.npy'),
                           mmap_mode='r')
        self.bvx = np.load(pjoin(root_path, 'precomp_desc', 'flows_bvx.npy'),
                           mmap_mode='r')
        self.bvx = np.load(pjoin(root_path, 'precomp_desc', 'flows_bvy.npy'),
                           mmap_mode='r')

        self.___augmentations = iaa.Noop()
        self.___reshaper = iaa.Noop()
        self.___normalization = iaa.Noop()

        self.___normalization = make_normalizer(
            normalization, map(lambda s: s['image'], self))

        if (augmentations is not None):
            self.___augmentations = augmentations

        if (resize_shape is not None):
            self.___reshaper = iaa.size.Resize(resize_shape)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        fx = self.fvx[..., min(len(self) - 2, idx)]
        fy = self.fvy[..., min(len(self) - 2, idx)]
        fnorm = np.linalg.norm(np.stack((fx, fy)), axis=0)

        aug = iaa.Sequential(
            [self.___reshaper, self.___augmentations, self.___normalization])
        aug = aug.to_deterministic()
        sample['image'], sample['labels'], sample[
            'label/segmentation'], sample['loc_keypoints'], sample[
                'loc_prior'], sample['fx'], sample['fy'], sample[
                    'fnorm'] = apply_augs(
                        sample['image'], sample['labels'].squeeze(),
                        sample['label/segmentation'].squeeze(),
                        sample['loc_keypoints'], sample['loc_prior'], fx, fy,
                        fnorm, aug)

        return sample

    def collate_fn(self, samples):
        out = super(Loader, Loader).collate_fn(samples)

        fnorm = [np.rollaxis(d['fnorm'], -1) for d in samples]
        fnorm = torch.stack([torch.from_numpy(f) for f in fnorm]).float()
        out['fnorm'] = fnorm

        fx = [np.rollaxis(d['fx'], -1) for d in samples]
        fx = torch.stack([torch.from_numpy(f) for f in fx]).float()
        out['fx'] = fx

        fy = [np.rollaxis(d['fy'], -1) for d in samples]
        fy = torch.stack([torch.from_numpy(f) for f in fy]).float()
        out['fy'] = fy

        out['pos_labels'] = pd.concat([s['pos_labels'] for s in samples])

        return out


class StackLoader(Loader):
    def __init__(self,
                 root_path,
                 depth=2,
                 augmentations=None,
                 normalization=None,
                 resize_shape=None,
                 csv_fname='video1.csv',
                 sp_labels_fname='sp_labels.npy',
                 sig_prior=0.05):
        """

        """
        super().__init__(root_path=root_path,
                         csv_fname=csv_fname,
                         sig_prior=sig_prior,
                         sp_labels_fname=sp_labels_fname)
        self.depth = depth

        self.graphs_path = pjoin(root_path, 'precomp_desc',
                                 'graphs_depth_{}'.format(self.depth))
        if (not os.path.exists(self.graphs_path)):
            self._prepare_graphs()

            os.makedirs(self.graphs_path)
            for i, g in enumerate(self.graphs):
                pickle.dump(
                    g,
                    open(pjoin(self.graphs_path, 'graph_{:04d}.p'.format(i)),
                         "wb"))

        self.____augmentations = iaa.Noop()
        self.____reshaper = iaa.Noop()
        self.____normalization = iaa.Noop()

        self.____normalization = make_normalizer(
            normalization, map(lambda s: s['image'], self))

        if (augmentations is not None):
            self.____augmentations = augmentations

        if (resize_shape is not None):
            self.____reshaper = iaa.size.Resize(resize_shape)

    def _prepare_graphs(self):

        from ilastikrag import rag
        import vigra

        self.graphs = []
        print('preparing graphs...')
        len_ = super(StackLoader, StackLoader).__len__(self) - (self.depth - 1)
        pbar = tqdm(total=len_)
        for i in range(len_):
            labels = np.array([
                super(StackLoader,
                      StackLoader).__getitem__(self, j)['labels'].squeeze()
                for j in range(i, i + self.depth)
            ])
            max_node = 0
            for d in range(self.depth):
                labels[d] += max_node
                max_node += labels[d].max() + 1

            labels_rag = np.rollaxis(labels, 0, 3)
            labels_rag = vigra.Volume(labels_rag, dtype=np.uint32)
            full_rag = rag.Rag(labels_rag)
            full_rag = full_rag.edge_ids.T.astype(np.int32)

            # add self loops
            loop_index = np.arange(0, labels.max())
            loop_index = loop_index[None, ...].repeat(2, axis=0)

            full_rag = np.concatenate([full_rag, loop_index], axis=1)

            self.graphs.append(full_rag)

            pbar.update(1)
        pbar.close()

    def __getitem__(self, idx):

        samples = [
            super(StackLoader, StackLoader).__getitem__(self, i)
            for i in range(idx, idx + self.depth)
        ]

        aug = iaa.Sequential([
            self.____reshaper, self.____augmentations, self.____normalization
        ])
        aug = aug.to_deterministic()

        for s in samples:

            s['image'], s['labels'], s['label/segmentation'], s[
                'loc_keypoints'], s['fx'], s['fy'], s['fnorm'] = apply_augs(
                    s['image'], s['labels'], s['label/segmentation'],
                    s['loc_keypoints'], s['fx'], s['fy'], s['fnorm'], aug)

        cat_sample = {}
        for k in samples[0].keys():
            cat_sample[k] = [samples[i][k] for i in range(len(samples))]

        cat_sample['graph'] = pickle.load(
            open(pjoin(self.graphs_path, 'graph_{:04d}.p'.format(idx)), "rb"))

        cat_sample['pos_labels'] = pd.concat(cat_sample['pos_labels'])

        return cat_sample

    def __len__(self):
        return super().__len__() - (self.depth - 1)

    def collate_fn(self, sample):
        to_cat = ['image', 'label/segmentation', 'labels', 'fnorm', 'fx', 'fy']

        assert len(sample) == 1, print(
            'set batch_size to 1 and modify depth instead')

        sample = sample[0]

        for k in sample.keys():
            if (k in to_cat):
                sample[k] = np.array([
                    np.moveaxis(sample[k][i], -1, 0)
                    for i in range(len(sample[k]))
                ])
                sample[k] = torch.from_numpy(sample[k]).float()

        return sample


if __name__ == "__main__":

    root_path = '/home/ubelix/artorg/lejeune'
    run_path = 'runs/siamese_dec/Dataset11'
    transf = iaa.Sequential(
        [iaa.Flipud(p=0.5),
         iaa.Fliplr(p=0.5),
         iaa.Rot90((1, 3))])
    import matplotlib.pyplot as plt

    dset = StackLoader(root_path=pjoin(root_path,
                                       'data/medical-labeling/Dataset11'),
                       depth=2,
                       augmentations=transf,
                       normalization='rescale',
                       resize_shape=512)
    dl = DataLoader(dset, collate_fn=dset.collate_fn, batch_size=1)

    for i, s in enumerate(dl):
        if i == 65:
            import pdb
            pdb.set_trace()  ## DEBUG ##
        labels = linearize_labels(s['labels'])

        print(s['pos_labels'])
