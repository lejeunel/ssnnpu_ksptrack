import os
from os.path import join as pjoin
from skimage import io, segmentation, measure, future
import glob
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import torch
import networkx as nx
import itertools
from tqdm import tqdm
import pandas as pd
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
import copy


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

            # compute hoof intersections
            hoof_ = self.hoof.loc[self.hoof['frame'] == idx].to_numpy()
            ind_hoof = self.hoof.columns == 'hoof_forward'

            labels_edges = np.stack([(n0, n1) for n0, n1 in graph.edges()])
            hoof_0 = np.vstack(hoof_[labels_edges[:, 0], ind_hoof])
            hoof_1 = np.vstack(hoof_[labels_edges[:, 1], ind_hoof])
            stack = np.stack((hoof_0, hoof_1))
            mins = stack.min(axis=0)
            inters = mins.sum(axis=1)
            edges = [(n0, n1,
                      dict(hoof_inter=inter))
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

        centroids = np.array([graph.nodes[n]['centroid'] for n in graph.nodes()])

        kps = sample['loc_keypoints'].keypoints
        coords = [(np.round(kp.y).astype(int), np.round_(kp.x).astype(int))
                  for kp in kps]

        sample['labels_clicked'] = [self.labels[i, j, idx]
                                    for i,j in coords]
        sample['graph'] = graph
        sample['labels'] = self.labels[..., idx][..., None]
        sample['bboxes'] = bboxes
        sample['centroids'] = centroids
        return sample

    def collate_fn(self, samples):
        out = super(Loader, Loader).collate_fn(samples)

        bboxes = [
            np.concatenate((i * np.ones((sample['bboxes'].shape[0], 1)),
                            sample['bboxes']),
                           axis=1)
            for i, sample in enumerate(samples)
        ]
        bboxes = np.concatenate(bboxes, axis=0)
        out['bboxes'] = torch.from_numpy(bboxes).float()

        centroids = [torch.from_numpy(s['centroids']).float()
                     for s in samples]
        out['centroids'] = torch.cat(centroids)
        
        out['graph'] = [s['graph'] for s in samples]
        out['labels_clicked'] = [s['labels_clicked'] for s in samples]

        return out
