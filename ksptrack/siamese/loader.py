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
    offsets = np.concatenate((offsets[:, 0][:, None],
                              offsets[:, 1][:, None],
                              offsets[:, 0][:, None],
                              offsets[:, 1][:, None]),
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
                 csv_fname='video1.csv',
                 labels_fname='sp_labels.npz',
                 nn_radius=0.1):
        """

        """
        super().__init__(root_path, augmentations, normalization, csv_fname)

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
        npzfile = np.load(pjoin(root_path, 'precomp_desc',
                                'flows.npz'))
        self.flows = dict()
        self.flows['bvx'] = npzfile['bvx']
        self.flows['fvx'] = npzfile['fvx']
        self.flows['bvy'] = npzfile['bvy']
        self.flows['fvy'] = npzfile['fvy']
        self.fvx = np.concatenate(
            (self.flows['fvx'], self.flows['fvx'][..., -1][..., None]), axis=-1)
        self.fvy = np.concatenate(
            (self.flows['fvy'], self.flows['fvy'][..., -1][..., None]), axis=-1)
        self.fv = np.sqrt(self.fvx**2 + self.fvy**2)


    def prepare_graphs(self):

        self.graphs = []

        print('preparing graphs...')

        pbar = tqdm(total=len(self.imgs))
        for idx, (im, truth) in enumerate(zip(self.imgs, self.truths)):
            labels = self.labels[..., idx]
            regions = measure.regionprops(labels + 1, intensity_image=truth)

            bboxes = [p['bbox'] for p in regions]
            bboxes = [(b[1], b[0], b[3], b[2]) for b in bboxes]
            centroids = [(p['centroid'][1] / labels.shape[1],
                          p['centroid'][0] / labels.shape[0]) for p in regions]
            truth_sp = [p['mean_intensity'] > 0.5 for p in regions]

            node_list = [[
                label,
                dict(truth=truth_,
                     labels=[label],
                     centroid=centroid,
                     bbox=bbox_)
            ] for label, truth_, centroid, bbox_ in zip(
                np.unique(labels), truth_sp, centroids, bboxes)]

            # region adjancency graph
            rag = future.graph.RAG(labels)
            rag.add_nodes_from(node_list)
            adj_edges = [(
                n0, n1,
                dict(
                    truth_sim=rag.nodes[n0]['truth'] == rag.nodes[n1]['truth'],
                    adjacent=True)) for n0, n1 in rag.edges()]
            rag.add_edges_from(adj_edges)

            for n in rag:
                rag.nodes[n].update({'pixel count': 0,
                                    'total color': np.array([0, 0, 0],
                                                            dtype=np.double)})
            # add mean colors to nodes
            for index in np.ndindex(labels.shape):
                current = labels[index]
                rag.nodes[current]['pixel count'] += 1
                rag.nodes[current]['total color'] += im[index]

            for n in rag:
                rag.nodes[n]['mean color'] = (rag.nodes[n]['total color'] /
                                              rag.nodes[n]['pixel count'])

            # make nearest neighbor graph based on centroid distances
            graph = rag.copy()
            node_label_list = [n[0] for n in node_list]
            nodes_ = np.array(np.meshgrid(node_label_list,
                                          node_label_list)).T.reshape(-1, 2)
            centroids_x = [n[1]['centroid'][0] for n in node_list]
            centroids_x = np.array(np.meshgrid(centroids_x,
                                               centroids_x)).T.reshape(-1, 2)
            centroids_y = [n[1]['centroid'][1] for n in node_list]
            centroids_y = np.array(np.meshgrid(centroids_y,
                                               centroids_y)).T.reshape(-1, 2)
            centroids_ = np.concatenate((centroids_x, centroids_y), axis=1)

            dists = np.sqrt((centroids_[:, 0] - centroids_[:, 1])**2 +
                            (centroids_[:, 2] - centroids_[:, 3])**2)
            inds = np.argwhere((dists < self.nn_radius) & (dists > 0)).ravel()

            edges = [(nodes_[i, 0], nodes_[i, 1],
                      dict(weight=graph.nodes[nodes_[i, 0]]['truth'] ==
                           graph.nodes[nodes_[i, 1]]['truth'],
                           adjacent=False)) for i in inds]
            graph.add_edges_from(edges)
            graph.add_edges_from(adj_edges)

            # edges for color similarity
            for x, y, d in graph.edges(data=True):
                diff = graph.nodes[x]['mean color'] - graph.nodes[y]['mean color']
                diff = np.linalg.norm(diff)
                d['col_sim'] = np.exp(-(diff ** 2)/255.0)

            # compute hoof intersections
            hoof_ = self.hoof.loc[self.hoof['frame'] == idx].to_numpy()
            ind_hoof = self.hoof.columns == 'hoof_forward'

            labels_edges = np.stack([(n0, n1) for n0, n1 in graph.edges()])
            hoof_0 = np.vstack(hoof_[labels_edges[:, 0], ind_hoof])
            hoof_1 = np.vstack(hoof_[labels_edges[:, 1], ind_hoof])
            stack = np.stack((hoof_0, hoof_1))
            mins = stack.min(axis=0)
            inters = mins.sum(axis=1)
            edges = [(n0, n1, dict(hoof_inter=inter))
                     for (n0, n1), inter in zip(graph.edges(), inters)]
            graph.add_edges_from(edges)

            self.graphs.append(graph)
            pbar.update(1)
        pbar.close()

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        # make tensor of bboxes
        # graph = copy.deepcopy(self.graphs[idx])
        graph = self.graphs[idx]
        bboxes = np.array([graph.nodes[n]['bbox'] for n in graph.nodes()])

        # flows = self.fv[..., idx][..., None]

        centroids = np.array(
            [graph.nodes[n]['centroid'] for n in graph.nodes()])

        shape = sample['image'].shape
        sample['graph'] = graph
        sample['labels'] = self.labels[..., idx][..., None]
        sample['bboxes'] = bboxes
        sample['centroids'] = centroids
        # sample['flows'] = flows
        return sample

    def collate_fn(self, samples):
        out = super(Loader, Loader).collate_fn(samples)

        bboxes = [
            np.concatenate((i * np.ones(
                (sample['bboxes'].shape[0], 1)), sample['bboxes']),
                           axis=1) for i, sample in enumerate(samples)
        ]
        bboxes = np.concatenate(bboxes, axis=0)
        out['bboxes'] = torch.from_numpy(bboxes).float()

        centroids = [torch.from_numpy(s['centroids']).float() for s in samples]
        out['centroids'] = torch.cat(centroids)

        out['graph'] = [s['graph'] for s in samples]

        # flows = [np.rollaxis(d['flows'], -1) for d in samples]
        # flows = torch.stack(
        #     [torch.from_numpy(i).float() for i in flows])
        # out['flows'] = flows

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
