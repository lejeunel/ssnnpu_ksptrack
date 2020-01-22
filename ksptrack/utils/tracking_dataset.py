from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.utils.loc_prior_dataset import collate_fn as base_collate
from os.path import join as pjoin
import os
import numpy as np
import torch


class SuperpixelDataset(LocPriorDataset):

    def __init__(self, root_path,
                 csv_fname='video1.csv',
                 sig_prior=0.1,
                 labels_fname='sp_labels.npz',
                 contours_fname='sp_labels_contours.npz'):
        super().__init__(root_path, csv_fname=csv_fname,
                         sig_prior=sig_prior)

        self.labels_path = pjoin(self.root_path, 'precomp_desc', labels_fname)

        assert(os.path.exists(self.labels_path)), print('{} not found'.format(self.labels_path))

        self.labels = np.load(self.labels_path)['sp_labels']

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        sample['labels'] = self.labels[..., idx][..., None]

    @staticmethod
    def collate_fn(data):

        out = base_collate(data)

        obj_prior = [np.rollaxis(d['labels'], -1) for d in data]
        obj_prior = torch.stack(
            [torch.from_numpy(i).float() for i in obj_prior])

        out['labels'] = obj_prior

        return out
