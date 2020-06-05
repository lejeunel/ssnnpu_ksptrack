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
import pickle


def _add_edge_filter(values, g):
    """Add an edge between first element in `values` and
    all other elements of `values` in the graph `g`.
    `values[0]` is expected to be the central value of
    the footprint used.

    Parameters
    ----------
    values : array
        The array to process.
    g : RAG
        The graph to add edges in.

    Returns
    -------
    0.0 : float
        Always returns 0.

    """
    values = values.astype(int)
    current = values[0]
    for value in values[1:]:
        g.add_edge(current, value)
    return 0.0


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
                            'graphs_depth_{}.p'.format(self.depth))
        if (not os.path.exists(graphs_path)):
            self.prepare_graphs()
            pickle.dump(self.graphs, open(graphs_path, "wb"))

        else:
            print('loading graphs at {}'.format(graphs_path))
            self.graphs = pickle.load(open(graphs_path, "rb"))

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

        from ilastikrag import rag
        import vigra

        self.graphs = []

        print('preparing graphs...')
        pbar = tqdm(total=len(self.imgs) - (self.depth - 1))
        for i in range(super(StackLoader, self).__len__() - self.depth + 1):
            labels = np.array([
                super(StackLoader, self).__getitem__(i)['labels'].squeeze()
                for i in range(i, i + self.depth)
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

        return samples, self.graphs[idx], clicked

    def __len__(self):
        return len(self.imgs) - (self.depth - 1)
        # return len(self.imgs)

    def collate_fn(self, samples):
        out = dict()
        out['graph'] = samples[0][1]
        clicked = samples[0][2]

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

    dset = LocPriorDataset(root_path=pjoin(
        '/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset30'),
                           normalization='rescale',
                           depth=2,
                           resize_shape=512)

    frames = [10, 11]
    label_stack = []
    max_node = 0
    for f in frames:
        labels = dset[f]['labels']
        label_stack.append(labels + max_node)
        max_node += labels.max() + 1
