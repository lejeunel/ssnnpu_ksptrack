#!/usr/bin/env python3
import numpy as np
from skimage.measure import regionprops
from skimage.transform import resize
import higra as hg
from ksptrack.utils import pb_hierarchy_extractor as pbh
from ksptrack.siamese.tree_explorer import TreeExplorer
from ksptrack.siamese.loader import Loader, StackLoader, linearize_labels, delinearize_labels
from tqdm import tqdm
import pandas as pd
import warnings
from torch.utils import data


class TreeSetExplorer(data.Dataset):
    """
    """
    def __init__(self,
                 data_dir,
                 thr=0.5,
                 thr_mode='upper',
                 ascending=False,
                 loader_class=Loader,
                 *args,
                 **kwargs):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """

        super(TreeSetExplorer, self).__init__()
        self.pbex = pbh.PbHierarchyExtractor(data_dir,
                                             normalization='rescale_histeq')

        self.dl = loader_class(data_dir, *args, **kwargs)

        # replace SLICs
        self.dl.labels = np.array([p['labels'] for p in self.pbex])

        self.thr = thr
        self.thr_mode = thr_mode
        self.ascending = ascending
        self.trees = []
        self.unlabeled = None
        self.positives = None

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        sample = self.dl[idx]

        # this will add "augmented set" to user-defined positives
        if self.positives is not None:
            sample['pos_labels'] = pd.concat(
                self.positives[self.positives['frame'] == f]
                for f in sample['frame_idx'])

        return sample

    def __str__(self):
        print('TreeSetExplorer')
        if self.positives is not None:
            print('num. of positives: {}'.format(
                self.positives[not self.positives['from_aug']]))
            print('num. of augmented: {}'.format(
                self.positives[self.positives['from_aug']]))
        else:
            print('no labels yet...')

    def collate_fn(self, samples):

        return self.dl.collate_fn(samples)

    def make_candidates(self, predictions, labels):
        """
        predictions is a [NxWxH] numpy array of foreground predictions (logits)
        """
        # build one merge tree per frame
        self.trees = []
        self.positives = []

        print('building all merge trees')
        pbar = tqdm(total=len(self.pbex))
        for i, s in enumerate(self.pbex):
            positives = np.fliplr(
                s['loc_keypoints'].to_xy_array()).astype(int).tolist()

            t = TreeExplorer(s['tree'],
                             labels[i],
                             predictions[i],
                             positives,
                             thr=self.thr,
                             thr_mode=self.thr_mode,
                             ascending=self.ascending)
            self.trees.append(t)
            clicked = [labels[i][p[0], p[1]] for p in positives]
            self.positives.extend([{
                'frame': i,
                'label': l,
                'n_labels': np.unique(labels[i]).shape[0],
                'from_aug': False
            } for l in clicked])
            pbar.update(1)

        pbar.close()

        self.positives = pd.DataFrame(self.positives)

    def make_unlabeled_candidates(self):

        pbar = tqdm(total=len(self.trees))
        self.unlabeled = []
        for i, t in enumerate(self.trees):
            for j, p in enumerate(t.parents_weights):
                dict_ = dict()
                dict_['frame'] = i
                dict_['weights'] = p['weight']
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
            for j, n in enumerate(self.trees[int(row['frame'])][int(
                    row['parent_idx'])]['nodes']):
                dict_ = dict()
                dict_['frame'] = int(row['frame'])
                dict_['label'] = n
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
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from ksptrack.siamese.modeling.siamese import Siamese
    import torch
    from ksptrack.siamese import utils as utls
    from skimage.transform import resize

    texp = TreeSetExplorer(
        data_dir='/home/ubelix/artorg/lejeune/data/medical-labeling/Dataset00',
        thr=0.5,
        thr_mode='upper',
        loader_class=LocPriorDataset,
        normalization='rescale')

    dl = DataLoader(texp, collate_fn=texp.collate_fn)

    cmap = plt.get_cmap('viridis')
    device = torch.device('cuda')
    model = Siamese(embedded_dims=15,
                    cluster_number=15,
                    backbone='unet',
                    siamese='none').to(device)

    cp = '/home/ubelix/artorg/lejeune/runs/siamese_dec/Dataset00/checkpoints/cp_pu.pth.tar'
    state_dict = torch.load(cp, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model.eval()
    predictions = []
    orig_shape = dl.dataset.pbex.dl[0]['image'].shape[:2]

    print('predicting object...')
    for s in dl:
        s = utls.batch_to_device(s, device)

        pred = model(s)['rho_hat'].sigmoid().squeeze().detach().cpu().numpy()

        labels = s['labels'].squeeze().int().detach().cpu().numpy() + 1

        props = regionprops(labels, intensity_image=pred)
        pred = np.array([p['mean_intensity'] for p in props])
        predictions.append(pred)

    texp.make_trees_and_positives(np.array(predictions))
    texp.make_unlabeled_candidates()
    texp.augment_positives(10)
    for s in texp:
        print(s['pos_labels'])
