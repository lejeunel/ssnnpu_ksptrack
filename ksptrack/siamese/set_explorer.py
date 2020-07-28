#!/usr/bin/env python3
import os
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm

from ksptrack.siamese.loader import Loader
from ksptrack.utils.loc_prior_dataset import relabel
from skimage.measure import regionprops
from skimage import transform


class SetExplorer(data.Dataset):
    """
    """
    def __init__(self,
                 data_dir,
                 loader_class=Loader,
                 sp_labels_fname='sp_labels.npy',
                 *args,
                 **kwargs):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """

        super(SetExplorer, self).__init__()

        self.dl = loader_class(data_dir, *args, **kwargs)

        label_path = pjoin(data_dir, 'precomp_desc', sp_labels_fname)

        if not os.path.exists(label_path):
            labels = np.array([p['labels'] for p in self.pbex])
            np.save(label_path, labels)

        self.dl.labels = np.load(label_path, mmap_mode='r')

        self.unlabeled = None
        self.positives = None

    @property
    def labels(self):
        return self.dl.labels

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        # this will add "augmented set" to user-defined positives
        if self.positives is not None:
            sample['pos_labels'] = pd.concat(
                self.positives[self.positives['frame'] == f]
                for f in np.array(sample['frame_idx']).flat)
        else:
            sample['pos_labels']['from_aug'] = False

        return sample

    def __str__(self):
        if self.positives is not None:
            return 'SetExplorer. num. of positives: {}, num. of augmented: {}'.format(
                self.n_pos, self.n_aug)
        else:
            return 'SetExplorer. run make_candidates first!'

    def collate_fn(self, samples):

        return self.dl.collate_fn(samples)

    def make_candidates(self, predictions, labels):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """
        # build one merge tree per frame
        self.candidates = []
        self.positives = pd.concat([s['pos_labels'] for s in self.dl])
        self.positives['from_aug'] = False

        pbar = tqdm(total=len(self.dl))
        for i, s in enumerate(self.dl):

            pred_ = transform.resize(predictions[i],
                                     labels[i].shape,
                                     preserve_range=True)
            self.candidates.extend(
                [{
                    'frame': i,
                    'label': p['label'] - 1,
                    'weight': p['mean_intensity'],
                    'n_labels': np.unique(labels[i]).shape[0],
                    'from_aug': False
                } for p in regionprops(labels[i] + 1, intensity_image=pred_)
                 if p['label'] not in s['pos_labels']['label']])
            pbar.update(1)

        pbar.close()

        self.unlabeled = pd.DataFrame(self.candidates)
        # sort by weight
        self.unlabeled.sort_values(by=['weight'],
                                   ascending=False,
                                   inplace=True)

    @property
    def n_pos(self):
        if self.positives is not None:
            return self.positives[self.positives['from_aug'] == False].shape[0]
        else:
            return 0

    @property
    def n_aug(self):
        if self.positives is not None:
            return self.positives[self.positives['from_aug']].shape[0]
        else:
            return 0

    def make_unlabeled_candidates(self):

        pbar = tqdm(total=len(self.trees))
        self.unlabeled = []
        for i, t in enumerate(self.trees):
            for j, p in enumerate(t.parents_weights):
                dict_ = dict()
                dict_['frame'] = i
                dict_['weights'] = p['weight']
                dict_['n_labels'] = t.leaves.shape[0]
                dict_['parent_idx'] = j
                self.unlabeled.append(dict_)
            pbar.update(1)
        pbar.close()
        self.unlabeled = pd.DataFrame(self.unlabeled)

        # sort by weight
        self.unlabeled.sort_values(by=['weights'],
                                   ascending=self.ascending,
                                   inplace=True)

    def augment_positives(self, n_samples, pos_set=None):
        """
        if pos_set is None, take self.P
        n_samples: if float, will take it as a ratio of unlabeled
        This function will transfer samples from self.u_candidates to
        self.positives
        """

        if pos_set is None:
            assert (self.positives is
                    not None), 'give pos_set or run make_trees_and_positives'

        if (isinstance(n_samples, float)):
            n_samples = int(
                self.positives[self.positives['from_aug'] == False] *
                n_samples)

        nu = self.unlabeled.shape[0]
        if (nu < n_samples):
            warnings.warn(
                'not enough unlabeled candidates, will return {} samples'.
                format(nu))
            n_samples = nu

        # get one row per label
        augs = []
        added = 0
        idx_to_delete = []
        for i, row in self.unlabeled.iterrows():
            if (added >= n_samples):
                break
            dict_ = dict()
            dict_['frame'] = int(row['frame'])
            dict_['n_labels'] = int(row['n_labels'])
            dict_['label'] = int(row['label'])
            dict_['from_aug'] = True
            augs.append(dict_)
            added += 1
            idx_to_delete.append(i)

        augs = pd.DataFrame(augs)

        # remove them from unlabeled
        self.unlabeled.drop(idx_to_delete, inplace=True)

        # add to_aug to positives
        self.positives = pd.concat((self.positives, augs))

        return self.positives[self.positives['from_aug']]


if __name__ == "__main__":

    from ksptrack.utils.loc_prior_dataset import LocPriorDataset
    from ksptrack.siamese.loader import Loader
    import matplotlib.pyplot as plt
    from imgaug import augmenters as iaa
    from torch.utils.data import DataLoader
    import torch
    from tree_explorer import relabel

    transf = iaa.Sequential(
        [iaa.Flipud(p=0.5),
         iaa.Fliplr(p=0.5),
         iaa.Rot90((1, 3))])
    texp = TreeSetExplorer(
        data_dir='/home/ubelix/lejeune/data/medical-labeling/Dataset00',
        thr=0.5,
        thr_mode='upper',
        loader_class=Loader,
        normalization='rescale',
        sp_labels_fname='sp_labels_pb.npy',
        augmentations=transf)
    dl = DataLoader(texp,
                    collate_fn=texp.collate_fn,
                    batch_size=2,
                    shuffle=True)

    import pdb
    pdb.set_trace()  ## DEBUG ##
    for s in dl:

        plt.subplot(221)
        plt.imshow(np.moveaxis(s['image'][0].detach().cpu().numpy(), 0, -1))
        plt.subplot(222)
        plt.imshow(s['labels'][0].squeeze().cpu().numpy() ==
                   s['pos_labels'].iloc[0]['label'])
        plt.subplot(223)
        plt.imshow(np.moveaxis(s['image'][1].detach().cpu().numpy(), 0, -1))
        plt.subplot(224)
        plt.imshow(s['labels'][1].squeeze().cpu().numpy() ==
                   s['pos_labels'].iloc[1]['label'])
        plt.show()
