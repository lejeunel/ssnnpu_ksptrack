#!/usr/bin/env python3
import os
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm

from ksptrack.pu.loader import Loader
from ksptrack.pu.tree_explorer import TreeExplorer
from ksptrack.utils import pb_hierarchy_extractor as pbh
from ksptrack.utils.loc_prior_dataset import relabel
from ksptrack.selective_search.main import main as ssmain
from ksptrack.pu.set_explorer import is_sp_positive
import networkx as nx
from argparse import Namespace


class TreeSetExplorer(data.Dataset):
    """
    """
    def __init__(self,
                 data_dir,
                 trees_path='precomp_desc/ss',
                 ascending=False,
                 loader_class=Loader,
                 sp_labels_fname='sp_labels.npy',
                 *args,
                 **kwargs):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """

        super(TreeSetExplorer, self).__init__()

        self.dl = loader_class(data_dir, *args, **kwargs)

        label_path = pjoin(data_dir, 'precomp_desc', sp_labels_fname)

        path_ = pjoin(data_dir, trees_path)
        if not os.path.exists(path_):
            args_ = Namespace(data_dir=data_dir, out_dir=path_)
            ssmain(args_)

        self.trees_path = pjoin(data_dir, trees_path)

        self.dl.labels = np.load(label_path, mmap_mode='r')

        self.ascending = ascending
        self.trees = {}
        self.unlabeled = None

        self.positives = pd.concat([s['annotations'] for s in self.dl])
        self.positives['from_aug'] = False
        self.positives['tp'] = True
        self.positives['epoch'] = 0

    @property
    def labels(self):
        return self.dl.labels

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        if idx in self.trees.keys():
            sample['tree'] = self.trees[idx]
        else:
            fname = os.path.splitext(sample['frame_name'])[0]
            tree_path = pjoin(self.trees_path, fname + '.p')
            graph = nx.read_gpickle(tree_path)
            sample['tree'] = TreeExplorer(graph, sample['labels'],
                                          np.zeros_like(sample['labels']))

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
            return 'TreeSetExplorer. num. of positives: {}, num. of augmented: {}'.format(
                self.n_pos, self.n_aug)
        else:
            return 'TreeSetExplorer. run make_candidates first!'

    def collate_fn(self, samples):

        return self.dl.collate_fn(samples)

    def make_candidates(self, predictions):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """
        # build one merge tree per frame

        print('propagating values in merge trees')
        pbar = tqdm(total=len(self))
        for i, s in enumerate(self):
            positives = [s['annotations'].y, s['annotations'].x]

            labels_ = s['labels']
            t = TreeExplorer(s['tree'].tree,
                             labels_,
                             predictions[i],
                             positives,
                             ascending=self.ascending)
            self.trees[i] = t
            pbar.update(1)

        pbar.close()

        self.make_unlabeled_candidates()

    @property
    def n_pos(self):
        if self.positives is not None:
            return self.positives[self.positives['from_aug'] == False].shape[0]
        else:
            return 0

    @property
    def ratio_purity_augs(self):
        if self.positives is not None:
            augs = self.positives[self.positives['from_aug']]
            n_tp = augs[augs['tp']].shape[0]
            return n_tp / augs.shape[0]
        else:
            return 1

    @property
    def n_aug(self):
        if self.positives is not None:
            return self.positives[self.positives['from_aug']].shape[0]
        else:
            return 0

    def make_unlabeled_candidates(self):

        pbar = tqdm(total=len(self))
        self.unlabeled = []
        for i, s in enumerate(self):
            for j, p in enumerate(s['tree'].parents_weights):
                dict_ = dict()
                dict_['frame'] = i
                dict_['weights'] = p['weight']
                dict_['n_labels'] = s['tree'].leaves.shape[0]
                dict_['parent_idx'] = j
                self.unlabeled.append(dict_)
            pbar.update(1)
        pbar.close()
        self.unlabeled = pd.DataFrame(self.unlabeled)

        # sort by weight
        self.unlabeled.sort_values(by=['weights'],
                                   ascending=self.ascending,
                                   inplace=True)

    def reset_augs(self):

        self.positives = self.positives[self.positives['from_aug'] == False]

    def augment_positives(self, n_samples, priors, ratio, pos_set=None):
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

        # get one row per label
        augs = []
        added = 0
        for i, row in self.unlabeled.iterrows():
            if (added >= n_samples):
                break
            for j, n in enumerate(self.trees[int(row['frame'])][int(
                    row['parent_idx'])]['nodes']):
                frame = int(row['frame'])
                label = n

                already_there = ((self.positives['frame'] == frame)
                                 & (self.positives['label'] == n)).any()

                frame_has_enough = np.round(
                    row['n_labels'] * priors[frame] *
                    ratio) < (self.positives['frame'] == frame).sum()

                if not already_there and not frame_has_enough:
                    dict_ = dict()
                    dict_['frame'] = frame
                    dict_['n_labels'] = int(row['n_labels'])
                    dict_['label'] = n
                    dict_['from_aug'] = True
                    dict_['tp'] = is_sp_positive(
                        self.dl[dict_['frame']]['label/segmentation'],
                        self.dl[dict_['frame']]['labels'], dict_['label'])
                    self.positives = pd.concat(
                        (self.positives, pd.DataFrame([dict_])))
                    added += 1

        if (added < n_samples):
            warnings.warn(
                'I did not find demanded num of samples. Returned: {}'.format(
                    added))

        return self.positives[self.positives['from_aug']]


if __name__ == "__main__":

    from ksptrack.utils.loc_prior_dataset import LocPriorDataset
    from ksptrack.siamese.loader import Loader
    import matplotlib.pyplot as plt
    from ksptrack.siamese.modeling.unet import UNet
    from ksptrack.siamese import utils as utls
    from imgaug import augmenters as iaa
    from torch.utils.data import DataLoader
    import torch
    from ksptrack.siamese.im_utils import get_features

    cp_path = '/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset30/checkpoints/cp_pu_pimul_1.0_pr_bag.pth.tar'
    device = torch.device('cuda')
    state_dict = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model = UNet(out_channels=1).to(device)
    model.load_state_dict(state_dict)

    texp = TreeSetExplorer(
        data_dir='/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset21',
        normalization='rescale')
    dl = DataLoader(texp, collate_fn=texp.collate_fn)

    res = get_features(model, dl, device)

    losses = [-np.log(1 - o + 1e-8) for o in res['outs']]

    step = 15
    epoch = 1
    dl.dataset.make_candidates(losses)
    dl.dataset.augment_positives(step * epoch)
    import pdb
    pdb.set_trace()  ## DEBUG ##
    print('resetting augmented set')
    print(dl.dataset)
    dl.dataset.reset_augs()
    dl.dataset.augment_positives(step * (epoch + 1))
    print(dl.dataset)
