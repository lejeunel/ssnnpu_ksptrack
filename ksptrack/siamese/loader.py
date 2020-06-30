import os
from os.path import join as pjoin
from skimage import future, measure
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, Sampler
from torch.utils import data
from tqdm import tqdm
import pandas as pd
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from ksptrack.utils.base_dataset import make_normalizer
from ksptrack.siamese import im_utils as iutls
from skimage.future import graph as skg
import random
import networkx as nx
from torch._six import int_classes as _int_classes
import pickle
from imgaug import augmenters as iaa
import imgaug as ia


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

        self.normalization = make_normalizer(normalization, self.imgs)

        self.reshaper = iaa.Noop()
        self.augmentations = iaa.Noop()

        if (augmentations is not None):
            self.augmentations = augmentations

        if (resize_shape is not None):
            self.reshaper = iaa.size.Resize(resize_shape)

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

        aug = iaa.Sequential(
            [self.reshaper, self.augmentations, self.normalization])
        aug_det = aug.to_deterministic()

        max_node = 0
        clicked = []
        shape = self.fv[0].shape

        fnorm = ia.HeatmapsOnImage(self.fv[idx], shape=shape)
        fx = ia.HeatmapsOnImage(self.fx[idx], shape=shape)
        fy = ia.HeatmapsOnImage(self.fy[idx], shape=shape)

        fnorm = aug_det(heatmaps=fnorm).get_arr()
        sample['fnorm'] = fnorm

        fx = aug_det(heatmaps=fx).get_arr()
        sample['fx'] = fx

        fy = aug_det(heatmaps=fy).get_arr()
        sample['fy'] = fy

        im = sample['image']
        shape = im.shape[:2]
        labels = ia.SegmentationMapsOnImage(sample['labels'].squeeze(),
                                            shape=shape)
        truth = ia.SegmentationMapsOnImage(
            sample['label/segmentation'].squeeze(), shape=shape)
        sample['image'] = aug_det(image=im)
        sample['labels'] = aug_det(
            segmentation_maps=labels).get_arr()[..., None] + max_node
        sample['label/segmentation'] = aug_det(
            segmentation_maps=truth).get_arr()[..., None]

        keypoints = aug_det.augment_keypoints(sample['loc_keypoints'])
        keypoints.clip_out_of_image_()
        keypoints.labels = [
            sample['labels'][k.y_int, k.x_int, 0] for k in keypoints.keypoints
        ]

        for kp in keypoints:
            kp_ = (kp.y_int, kp.x_int)
            clicked.append(sample['labels'][kp_[0], kp_[1], 0])
        max_node += sample['labels'].max() + 1

        sample['clicked'] = clicked

        sample['graph'] = self.graphs[idx]

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

        out['clicked'] = [s['clicked'] for s in samples]
        out['clicked'] = [
            item for sublist in out['clicked'] for item in sublist
        ]

        return out


class StackLoader(data.Dataset):
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
        super(StackLoader, self).__init__()

        self.single_loader = LocPriorDataset(root_path=root_path,
                                             csv_fname=csv_fname,
                                             sig_prior=sig_prior)

        if (resize_shape is not None):
            self.reshaper = iaa.size.Resize(resize_shape)

        self.normalization = make_normalizer(normalization,
                                             self.single_loader.imgs)

        self.augmentations = iaa.Noop()
        if (augmentations is not None):
            self.augmentations = augmentations

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
        self.fv = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in self.fv]
        self.fx = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in self.fx]
        self.fy = [(f - f.min()) / (f.max() - f.min() + 1e-8) for f in self.fy]

    def prepare_graphs(self):

        from ilastikrag import rag
        import vigra

        self.graphs = []

        print('preparing graphs...')
        pbar = tqdm(total=len(self.single_loader) - (self.depth - 1))
        for i in range(len(self.single_loader) - self.depth + 1):
            labels = np.array([
                self.single_loader[i]['labels'].squeeze()
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

        samples = [self.single_loader[i] for i in range(idx, idx + self.depth)]

        aug = iaa.Sequential(
            [self.reshaper, self.augmentations, self.normalization])
        aug_det = aug.to_deterministic()

        max_node = 0
        clicked = []
        shape = self.fv[0].shape
        for i in range(self.depth):
            fnorm = ia.HeatmapsOnImage(self.fv[idx + i], shape=shape)
            fx = ia.HeatmapsOnImage(self.fx[idx + i], shape=shape)
            fy = ia.HeatmapsOnImage(self.fy[idx + i], shape=shape)

            fnorm = aug_det(heatmaps=fnorm).get_arr()
            samples[i]['fnorm'] = fnorm

            fx = aug_det(heatmaps=fx).get_arr()
            samples[i]['fx'] = fx

            fy = aug_det(heatmaps=fy).get_arr()
            samples[i]['fy'] = fy

            im = samples[i]['image']
            shape = im.shape[:2]
            labels = ia.SegmentationMapsOnImage(samples[i]['labels'].squeeze(),
                                                shape=shape)
            truth = ia.SegmentationMapsOnImage(
                samples[i]['label/segmentation'].squeeze(), shape=shape)
            samples[i]['image'] = aug_det(image=im)
            samples[i]['labels'] = aug_det(
                segmentation_maps=labels).get_arr()[..., None] + max_node
            samples[i]['label/segmentation'] = aug_det(
                segmentation_maps=truth).get_arr()[..., None]

            keypoints = aug_det.augment_keypoints(samples[i]['loc_keypoints'])
            shape = im.shape[:2]
            keypoints = ia.KeypointsOnImage([
                ia.Keypoint(np.clip(k.x, a_min=0, a_max=shape[1] - 1),
                            np.clip(k.y, a_min=0, a_max=shape[0] - 1))
                for k in keypoints
            ], shape)

            keypoints.clip_out_of_image_()
            keypoints.labels = [
                samples[i]['labels'][k.y_int, k.x_int, 0]
                for k in keypoints.keypoints
            ]

            for kp in keypoints:
                kp_ = (kp.y_int, kp.x_int)
                clicked.append(samples[i]['labels'][kp_[0], kp_[1], 0])
            max_node += samples[i]['labels'].max() + 1

        return samples, self.graphs[idx], clicked

    def __len__(self):
        return len(self.single_loader.imgs) - (self.depth - 1)
        # return len(self.imgs)

    def collate_fn(self, samples):
        out = dict()
        out['graph'] = samples[0][1]
        clicked = samples[0][2]

        samples = samples[0][0]
        out_ = self.single_loader.collate_fn(samples)
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

        out['clicked'] = clicked

        im = [np.rollaxis(d['image'], -1) for d in samples]
        im = torch.stack([torch.from_numpy(i).float() for i in im])

        truth = [np.rollaxis(d['label/segmentation'], -1) for d in samples]
        truth = torch.stack([torch.from_numpy(i) for i in truth]).float()

        labels = [np.rollaxis(d['labels'], -1) for d in samples]
        labels = torch.stack([torch.from_numpy(i) for i in labels]).float()
        out['labels'] = labels
        out['label/segmentation'] = truth
        out['image'] = im

        return out


if __name__ == "__main__":
    from skimage import io
    from ksptrack.siamese import utils as utls
    from ksptrack.siamese.modeling.siamese import Siamese
    from ksptrack.siamese.losses import do_previews
    import matplotlib.pyplot as plt

    root_path = '/home/ubelix/artorg/lejeune'
    run_path = 'runs/siamese_dec/Dataset30'
    transf = iaa.Sequential(
        [iaa.Flipud(p=0.5),
         iaa.Fliplr(p=0.5),
         iaa.Rot90((1, 3))])

    dset = Loader(
        root_path=pjoin(root_path, 'data/medical-labeling/Dataset30'),
        # '/home/ubelix/lejeune/data/medical-labeling/Dataset30'),
        normalization='rescale',
        resize_shape=512)
    # dset = LocPriorDataset(root_path=pjoin(
    #     '/home/ubelix/lejeune/data/medical-labeling/Dataset00'),
    #                        normalization='rescale',
    #                        augmentations=transf,
    #                        resize_shape=512)
    dl = DataLoader(dset, collate_fn=dset.collate_fn)

    device = torch.device('cpu')

    model = Siamese(embedded_dims=15, cluster_number=15,
                    backbone='unet').to(device)
    cp = pjoin(root_path, run_path, 'checkpoints/init_dec.pth.tar')
    state_dict = torch.load(cp, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    cmap = plt.get_cmap('viridis')
    import pdb
    pdb.set_trace()  ## DEBUG ##
    for data in dl:

        im = (255 * np.rollaxis(data['image'].squeeze().detach().cpu().numpy(),
                                0, 3)).astype(np.uint8)
        labels = data['labels'].squeeze().detach().cpu().numpy()
        labels = (cmap((labels.astype(float) / labels.max() * 255).astype(
            np.uint8))[..., :3] * 255).astype(np.uint8)
        clusters = model(data)['clusters']
        clusters = iutls.make_clusters(labels, clusters)
        all_ = np.concatenate((im, labels, clusters))
        io.imsave(
            pjoin(root_path, run_path, '{}.png'.format(data['frame_idx'][0])),
            all_)
