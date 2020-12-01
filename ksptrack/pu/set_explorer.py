#!/usr/bin/env python3
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm

from ksptrack.pu.loader import Loader
from ksptrack.utils.loc_prior_dataset import relabel


def is_sp_positive(truth, label_map, label, thr=0.5):

    area_sp = (label_map == label).sum()
    n_pix_pos = truth[label_map == label].sum()
    ratio = n_pix_pos / area_sp

    if ratio >= thr:
        return True
    else:
        return False


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
        """

        super(SetExplorer, self).__init__()

        self.dl = loader_class(data_dir, *args, **kwargs)

        label_path = pjoin(data_dir, 'precomp_desc', sp_labels_fname)

        self.unlabeled = None
        self.positives = pd.concat([s['annotations'] for s in self.dl])
        self.positives['from_aug'] = False
        self.positives['tp'] = True

    @property
    def labels(self):
        return self.dl.labels

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        # this will add "augmented set" to user-defined positives
        if self.positives is not None:
            sample['annotations'] = pd.concat(
                self.positives[self.positives['frame'] == f]
                for f in np.array(sample['frame_idx']).flat)
        else:
            sample['annotations']['from_aug'] = False

        return sample

    def __str__(self):
        if self.positives is not None:
            return 'SetExplorer. num. of positives: {}, num. of augmented: {}'.format(
                self.n_pos, self.n_aug)
        else:
            return 'SetExplorer. run make_candidates first!'

    def collate_fn(self, samples):

        return self.dl.collate_fn(samples)

    def make_candidates(self, predictions):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """

        # build one merge tree per frame
        self.candidates = []

        pbar = tqdm(total=len(self.dl))
        for i, s in enumerate(self.dl):
            labels = s['labels']

            lab_pred = np.concatenate(
                (np.unique(labels)[..., None], predictions[i][..., None]),
                axis=1)
            labs_to_remove = self.positives[self.positives['frame'] ==
                                            i]['label']
            idx_to_keep = np.array([
                lab_pred[:, 0] != l for l in labs_to_remove
            ]).sum(axis=0).astype(bool)
            lab_pred = lab_pred[idx_to_keep]
            n_labels = np.unique(labels).shape[0]

            self.candidates.extend([{
                'frame': i,
                'label': int(l),
                'weight': p,
                'n_labels': n_labels,
                'from_aug': False,
                'epoch': 0
            } for l, p in lab_pred])

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

    @property
    def ratio_purity_augs(self):
        if self.positives is not None:
            augs = self.positives[self.positives['from_aug']]
            n_tp = augs[augs['tp']].shape[0]
            return n_tp / augs.shape[0]
        else:
            return 1

    def reset_augs(self):

        self.unlabeled = pd.DataFrame(self.candidates)
        # sort by weight
        self.unlabeled.sort_values(by=['weight'],
                                   ascending=False,
                                   inplace=True)
        self.positives = self.positives[self.positives['from_aug'] == False]

    def augment_positives(self, n_samples, priors, ratio):
        """
        if pos_set is None, take self.P
        n_samples: if float, will take it as a ratio of unlabeled
        This function will transfer samples from self.u_candidates to
        self.positives
        """

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

            frame = int(row['frame'])
            frame_has_enough = np.round(
                row['n_labels'] * priors[frame] *
                ratio) < (self.positives['frame'] == frame).sum()

            if not frame_has_enough:
                dict_ = dict()
                dict_['frame'] = int(row['frame'])
                dict_['n_labels'] = int(row['n_labels'])
                dict_['label'] = int(row['label'])
                dict_['from_aug'] = True
                dict_['tp'] = is_sp_positive(
                    self.dl[dict_['frame']]['label/segmentation'],
                    self.dl[dict_['frame']]['labels'], dict_['label'])
                self.positives = pd.concat(
                    (self.positives, pd.DataFrame([dict_])))
                added += 1
                idx_to_delete.append(i)

        # remove them from unlabeled
        self.unlabeled.drop(idx_to_delete, inplace=True)

        return self.positives[self.positives['from_aug']]


def make_distrib_rho(rho, n_bins=100, range=[0., 1.]):
    levels = np.linspace(range[0], range[1], n_bins)
    counts = np.array([np.sum(rho >= l) for l in levels])
    counts = counts / counts.sum()

    return counts


if __name__ == "__main__":

    from ksptrack.pu.modeling.unet import UNet
    from ksptrack.pu import utils as utls
    from ksptrack.utils.my_utils import get_pm_array
    from torch.utils.data import DataLoader
    import torch
    from ksptrack.pu.im_utils import get_features
    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu, try_all_threshold
    from skimage import filters

    root_path = '/home/ubelix/artorg/lejeune'
    cp_path = pjoin(
        root_path,
        'runs/siamese_dec/Dataset32/pu_piovrs_1.8_ph0/cps/cp_0020.pth.tar')
    device = torch.device('cuda')
    state_dict = torch.load(cp_path, map_location=lambda storage, loc: storage)
    model = UNet(out_channels=1).to(device)
    model.load_state_dict(state_dict)

    texp = SetExplorer(data_dir=pjoin(root_path,
                                      'data/medical-labeling/Dataset32'),
                       normalization='rescale',
                       resize_shape=512)
    dl = DataLoader(texp, collate_fn=texp.collate_fn)

    res = get_features(model, dl, device)

    frame = 50
    n_bins = 100
    probas = res['outs']
    im = dl.dataset[frame]['image']
    labels = dl.dataset[frame]['labels']
    # dist_rho = make_distrib_rho(probas[frame], n_bins=n_bins)
    # thr = dist_rho.mean()

    probas_map = get_pm_array(labels[None, ...], [probas[frame]])[0]
    data = {
        'image': im,
        'labels': labels,
        'probas': probas,
        'probas_map': probas_map
    }
    np.savez('gc_data_br.npz', **data)
