import os
from os.path import join as pjoin
from skimage import future, measure
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from skimage.future import graph as skg
from random import shuffle


def scale_boxes(bboxes, factor):
    # boxes are (x0, y0, x1, y1)

    # align boxes on their centers
    offsets = np.concatenate((((bboxes[:, 2] - bboxes[:, 0]) // 2)[:, None],
                              ((bboxes[:, 3] - bboxes[:, 1]) // 2)[:, None]),
                             axis=1)
    offsets = np.concatenate((offsets[:, 0][:, None], offsets[:, 1][:, None],
                              offsets[:, 0][:, None], offsets[:, 1][:, None]),
                             axis=1)
    bboxes_shifted = bboxes - offsets
    bboxes_scaled = bboxes_shifted * factor

    bboxes_recentered = bboxes_scaled + offsets

    return bboxes_recentered


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
        self.flows = dict()
        self.flows['bvx'] = npzfile['bvx']
        self.flows['fvx'] = npzfile['fvx']
        self.flows['bvy'] = npzfile['bvy']
        self.flows['fvy'] = npzfile['fvy']
        self.fvx = np.rollaxis(
            np.concatenate(
                (self.flows['fvx'], self.flows['fvx'][..., -1][..., None]),
                axis=-1), -1, 0)
        self.fvy = np.rollaxis(
            np.concatenate(
                (self.flows['fvy'], self.flows['fvy'][..., -1][..., None]),
                axis=-1), -1, 0)
        self.fv = np.sqrt(self.fvx**2 + self.fvy**2)
        self.fv = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in self.fv]

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

        flow = self.fv[idx][..., None]
        flow = self.reshaper_img.augment_image(flow)
        sample['flow'] = flow
        return sample

    def collate_fn(self, samples):
        out = super(Loader, Loader).collate_fn(samples)

        out['graph'] = [s['graph'] for s in samples]

        flow = [np.rollaxis(d['flow'], -1) for d in samples]
        flow = torch.stack([torch.from_numpy(f) for f in flow]).float()
        out['flow'] = flow

        return out


class StackLoader(Dataset):
    def __init__(self, depth, *args, **kwargs):
        self.loader = Loader(*args, **kwargs)
        self.depth = depth

    def __getitem__(self, index):
        sample = [self.loader[i] for i in range(index, index + self.depth)]
        shuffle(sample)
        return sample

    def __len__(self):
        return len(self.loader) - (self.depth - 1)

    def collate_fn(self, samples):
        samples = samples[0]
        return self.loader.collate_fn(samples)


if __name__ == "__main__":

    dset = StackLoader(
        depth=2,
        root_path=pjoin(
            '/home/ubelix/lejeune/data/medical-labeling/Dataset00'))
    dl = DataLoader(dset, shuffle=True, collate_fn=dset.collate_fn)

    for s in dl:
        print(s['frame_idx'])
