import logging
import numpy as np
from ksptrack.utils import my_utils as utls
import progressbar
import os
import pickle as pk
import matplotlib.pyplot as plt
import networkx as nx
import progressbar
import collections
import pandas as pd
import networkx as nx

class HOOFExtractor:

    """
    Computes Histograms of Oriented Optical Flow on superpixels
    We build HOOF descriptors on a coarse grid
    Descriptors are mapped to superpixel labels according to overlap priority
    """
    def __init__(self,
                 fvx,
                 fvy,
                 bvx,
                 bvy,
                 labels,
                 grid_size_ratio,
                 n_bins=30,
                 directions=['forward', 'backward']):

        self.fvx = fvx
        self.fvy = fvy
        self.bvx = bvx
        self.bvy = bvy

        self.directions = directions

        self.grid = make_grid_array(labels[..., 0].shape,
                                    grid_size_ratio)

        self.labels = labels

        self.bins_hoof = np.linspace(-np.pi/2,np.pi/2,n_bins+1)

        self.logger = logging.getLogger('HOOFExtractor')

        # make mapping from labels to grid on all frames
        self.logger.info('Making labels-to-grid mappings')
        mapping = dict()
        for f in range(labels.shape[-1]):
            print('{}/{}'.format(f+1, labels.shape[-1]))
            mapping[f] = dict()
            map_ = sp_to_grid_map(labels[..., f],
                                          self.grid).tolist()
            mapping[f] = {ls[0]:ls[1] for ls in map_}

        self.mapping = mapping

    def make_hoof_on_grid(self, path):
        """
        Compute HOOF on _grid_ labels
        """

        hoof = dict()
        file_hoof_grid = os.path.join(path, 'hoof_grid.npz')

        if(not os.path.exists(file_hoof_grid)):
            for dir_ in self.directions:
                self.logger.info('Computing HOOF\
                in {} direction on grid of {} elements'\
                                .format(dir_, np.unique(self.grid).size))

                frames = np.arange(self.labels.shape[-1]-1)
                if(dir_ == 'forward'):
                    vx = self.fvx
                    vy = self.fvy
                else:
                    vx = self.bvx
                    vy = self.bvy

                hoof[dir_] = list()

                with progressbar.ProgressBar(maxval=frames.size) as bar:
                    for f in frames:
                        bar.update(f)
                        hoof[dir_].append(make_hoof_labels(vx[..., f],
                                                        vy[..., f],
                                                        self.grid,
                                                        self.bins_hoof))
            self.logger.info('Saving HOOF on grid ...')
            np.savez(file_hoof_grid, **{'hoof': hoof, 'grid': self.grid})
            self.hoof_grid = hoof
        else:
            self.logger.info('Loading HOOF on grid ... (delete to re-run)')
            self.hoof_grid = np.load(file_hoof_grid)['hoof'][()]

        return self.hoof_grid

    def make_hoof_inters(self, path, g):
        """
        Compute HOOF intersection on sps
        Neighboring superpixels are given by undirected graph g
        """
        file_hoof_sps = os.path.join(path, 'hoof_inters_graph.npz')

        if(not os.path.exists(file_hoof_sps)):
            self.hoof_grid = self.make_hoof_on_grid(path)
            for dir_ in self.directions:
                self.logger.info('Computing HOOF\
                in {} direction on superpixels'\
                                .format(dir_))


                edges = g.edges()
                if(dir_ == 'forward'):
                    edges = [e for e in edges
                             if(e[1][0] > e[0][0])]
                    keys = ['fvx', 'fvy']
                else:
                    edges = [e for e in edges
                             if(e[0][0] > e[1][0])]
                    keys = ['bvx', 'bvy']

                frames = sorted(set([e[0][0] for e in edges]))

                with progressbar.ProgressBar(maxval=len(g.edges())) as bar:
                    for f in frames:
                        edges_ = [e for e in edges if(e[0][0] == f)]


                        for e in edges_:
                            if(dir_ == 'forward'):
                                s_start = e[0][1]
                                s_end = e[1][1]
                            else:
                                s_start = e[1][1]
                                s_end = e[0][1]

                            hoof_start = \
                                self.hoof_grid[dir_][f][self.mapping[f][s_start]]
                            hoof_end = \
                                self.hoof_grid[dir_][f][self.mapping[f][s_end]]

                            g[e[0]][e[1]][dir_] = utls.hist_inter(
                                hoof_start,
                                hoof_end)

                        bar.update(f)

                        self.logger.info('Saving HOOF on sps ...')
            np.savez(file_hoof_sps, **{'g': g})
            self.g = g
        else:
            self.logger.info('Loading HOOF on sps ... (delete to re-run)')
            self.g = np.load(file_hoof_sps)['g']

def sp_to_grid_map(labels, grid):
    """
    Performs the mapping function from label to grid values
    """

    map_ = list()

    concat_ = np.concatenate((labels[..., np.newaxis],
                              grid[..., np.newaxis]), axis=-1)
    concat_ = concat_.reshape((-1,2))
    concat_ = list(map(tuple, concat_))
    counts = collections.Counter(concat_)

    for l in np.unique(labels):
        candidates_with_counts = [(k[0], k[1], c)
                                  for k,c in counts.items() if k[0]==l]
        candidates_with_counts.sort(key=lambda tup: tup[-1], reverse=True)
        map_.append((l, candidates_with_counts[0][1]))

    return np.asarray(map_)


def make_grid_array(shape, grid_size_ratio):
    """
    Builds the grid with grid_size_ratio to determine size of cells
    """

    max_size = np.max(shape)
    block_size = int(grid_size_ratio*max_size)
    print('block_size: {}'.format(block_size))

    n_blocks = int(np.ceil((max_size**2)/(block_size**2)))
    print('n_blocks: {}'.format(n_blocks))

    blocks = np.empty((max_size, max_size), dtype=np.uint16)

    val = 1
    for i in range(n_blocks//2):
        for j in range(n_blocks//2):
            blocks[i*block_size:(i+1)*block_size,
                   j*block_size:(j+1)*block_size] = val
            val += 1
        
    print('n_blocks before reshape: {}'.format(np.unique(blocks).size))
    blocks = blocks[0:shape[0], 0:shape[1]]

    return blocks

def make_hoof_labels(fx, fy, labels, bins_hoof):

    unq_labels = np.unique(labels)
    bins_label = range(unq_labels.size + 1)
    angle = np.arctan2(fx,
                       fy)
    norm = np.linalg.norm(
        np.concatenate((fx[...,np.newaxis],
                        fy[...,np.newaxis]),
                                        axis=-1),
        axis=-1).ravel()

    # Get superpixel indices for each label
    l_mask = [np.where(labels.ravel() == l)[0].tolist()
            for l in unq_labels]

    # Get angle-bins for each pixel
    b_angles = np.digitize(angle.ravel(), bins_hoof).ravel()

    # Get angle-bins indices for each pixel
    b_mask = [np.where(b_angles.ravel() == b)[0].tolist()
            for b in range(1, len(bins_hoof))]

    # Sum norms for each bin and each label
    hoof__ = np.asarray([[np.sum(norm[list(set(l_ + b_))])
            for b_ in b_mask]
            for l_ in l_mask])

    # Normalize w.r.t. L1-norm
    l1_norm = np.sum(hoof__, axis = 1).reshape(
        (unq_labels.size, 1))
    hoof__ = np.nan_to_num(hoof__/l1_norm)

    # Store HOOF in dictionary
    hoof = {unq_labels[i]: hoof__[i, :]
            for i in range(len(unq_labels))}

    return hoof
