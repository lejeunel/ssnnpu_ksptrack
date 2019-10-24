import logging
import numpy as np
from ksptrack.utils import my_utils as utls
from ksptrack.utils import optical_flow_extractor as oflowx
import progressbar
import os
import pickle as pk
import matplotlib.pyplot as plt
import networkx as nx
import collections
import pandas as pd
import networkx as nx
import tqdm


class HOOFExtractor:
    """
    Computes Histograms of Oriented Optical Flow on superpixels
    We build HOOF descriptors on a coarse grid instead of superpixels as the
    latter will make the computation time prohibitive as the number of 
    superpixels increase...
    For improved precision, decrease the grid_size_ratio parameter
    Descriptors are mapped to superpixel labels according to overlap priority
    """

    def __init__(self,
                 conf,
                 flows_path,
                 labels,
                 grid_size_ratio,
                 bag_n_bins=30,
                 directions=['forward', 'backward']):

        self.directions = directions
        self.conf = conf
        self.labels = labels
        self.bins_hoof = np.linspace(-np.pi / 2, np.pi / 2, bag_n_bins + 1)
        self.logger = logging.getLogger('HOOFExtractor')
        self.flows_path = flows_path

        self.grid = self.make_grid_array(self.labels[..., 0].shape,
                                         grid_size_ratio,
                                         self.conf.precomp_desc_path)
        self.mapping = self.make_mapping(self.conf.precomp_desc_path)

    def make_mapping(self, path):
        # make mapping from labels to grid on all frames

        file_mapping = os.path.join(path, 'grid_mapping.npz')
        if (not os.path.exists(file_mapping)):
            self.logger.info('Making labels-to-grid mappings')
            mapping = dict()
            for f in range(self.labels.shape[-1]):
                print('{}/{}'.format(f + 1, self.labels.shape[-1]))
                mapping[f] = dict()
                map_ = sp_to_grid_map(self.labels[..., f], self.grid).tolist()
                mapping[f] = {ls[0]: ls[1] for ls in map_}
            np.savez(file_mapping, **{'mapping': mapping})
        else:
            mapping = np.load(file_mapping, allow_pickle=True)['mapping'][()]

        return mapping

    def make_hoof_on_grid(self):
        """
        Compute HOOF on _grid_ labels
        """

        hoof = dict()

        path_flow = self.flows_path

        if (not os.path.exists(path_flow)):
            self.calc_oflow(save=True, save_path=path_flow)

        flows = self.get_flows(path_flow)

        file_hoof_grid = os.path.join(self.conf.precomp_desc_path, 'hoof.npz')

        if (not os.path.exists(file_hoof_grid)):
            for dir_ in self.directions:
                self.logger.info('Computing HOOF in {} direction on grid of {} elements'\
                                .format(dir_, np.unique(self.grid).size))

                frames = np.arange(self.labels.shape[-1] - 1)
                if (dir_ == 'forward'):
                    vx = flows['fvx']
                    vy = flows['fvy']
                else:
                    vx = flows['bvx']
                    vy = flows['bvy']

                hoof[dir_] = list()

                bar = tqdm.tqdm(total=frames.size)
                for f in frames:
                    bar.update(1)
                    hoof[dir_].append(
                        make_hoof_labels(vx[..., f], vy[..., f], self.grid,
                                            self.bins_hoof))
                bar.close()
            self.logger.info('Saving HOOF on grid ...')
            np.savez(file_hoof_grid, **{'hoof': hoof, 'grid': self.grid})
            self.hoof_grid = hoof
        else:
            self.logger.info('Loading HOOF on grid {}'.format(file_hoof_grid))
            self.logger.info('... (delete to re-run)')
            self.hoof_grid = np.load(file_hoof_grid, allow_pickle=True)['hoof'][()]

        return self.hoof_grid

    def make_hoof_inters(self, g):
        """
        Compute HOOF intersection on sps
        Neighboring superpixels are given by undirected graph g
        """
        file_hoof_sps = os.path.join(self.conf.precomp_desc_path,
                                     'hoof_inters_graph.npz')

        if (not os.path.exists(file_hoof_sps)):
            self.hoof_grid = self.make_hoof_on_grid()
            for dir_ in self.directions:
                self.logger.info(
                    'Computing HOOF in {} direction on superpixels'.format(
                        dir_))

                edges = g.edges()
                if (dir_ == 'forward'):
                    keys = ['fvx', 'fvy']
                else:
                    keys = ['bvx', 'bvy']

                bar = tqdm.tqdm(total=len(g.edges()))
                for i, e in enumerate(edges):
                    i_0 = np.argmin((e[0][0], e[1][0]))
                    i_1 = np.argmax((e[0][0], e[1][0]))
                    f_0 = e[i_0][0]
                    f_1 = e[i_1][0]
                    s_0 = e[i_0][1]
                    s_1 = e[i_1][1]

                    hoof_0 = \
                        self.hoof_grid[dir_][f_0]\
                        [self.mapping[f_0][s_0]]
                    hoof_1 = \
                        self.hoof_grid[dir_][f_0]\
                        [self.mapping[f_1][s_1]]

                    g[e[0]][e[1]][dir_] = utls.hist_inter(hoof_0, hoof_1)

                    bar.update(1)
                bar.close()

            self.logger.info('Saving HOOF on sps ...')
            with open(file_hoof_sps, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
            self.g = g
        else:
            self.logger.info('Loading HOOF graph {} on sps'.format(file_hoof_sps))
            self.logger.info('... (delete to re-run)')
            with open(file_hoof_sps, 'rb') as f:
                self.g = pk.load(f)

        return self.g

    def make_grid_array(self, shape, grid_size_ratio, path):
        """
        Builds the grid with grid_size_ratio to determine size of cells
        """

        file_grid = os.path.join(path, 'hoof_grid.npz')
        if (not os.path.exists(file_grid)):
            max_size = np.max(shape)
            grid_size = int(grid_size_ratio * max_size)
            print('block_size: {}'.format(grid_size))

            n_blocks = int(np.ceil((max_size**2) / (grid_size**2)))
            print('n_blocks: {}'.format(n_blocks))

            grid = np.empty((max_size, max_size), dtype=np.uint16)

            val = 1
            for i in range(n_blocks // 2):
                for j in range(n_blocks // 2):
                    grid[i * grid_size:(i + 1) * grid_size, j *
                         grid_size:(j + 1) * grid_size] = val
                    val += 1

            print('n_blocks before reshape: {}'.format(np.unique(grid).size))
            grid = grid[0:shape[0], 0:shape[1]]
            np.savez(file_grid, **{'grid': grid})
        else:
            grid = np.load(file_grid)['grid']

        return grid

    def get_flows(self, path):
        flows = dict()
        self.logger.info('Loading optical flows...')
        npzfile = np.load(path)
        flows['bvx'] = npzfile['bvx']
        flows['fvx'] = npzfile['fvx']
        flows['bvy'] = npzfile['bvy']
        flows['fvy'] = npzfile['fvy']
        return flows

    def calc_oflow(self, save=True, save_path=None):

        oflow_extractor = oflowx.OpticalFlowExtractor(
            save_path, self.conf.oflow_alpha, self.conf.oflow_ratio,
            self.conf.oflow_min_width, self.conf.oflow_outer_iters,
            self.conf.oflow_inner_iters, self.conf.oflow_sor_iters)
        oflow_extractor.extract(self.conf.frameFileNames, save_path)


def sp_to_grid_map(labels, grid):
    """
    Performs the mapping function from label to grid values
    """

    map_ = list()

    concat_ = np.concatenate((labels[..., np.newaxis], grid[..., np.newaxis]),
                             axis=-1)
    concat_ = concat_.reshape((-1, 2))
    concat_ = list(map(tuple, concat_))
    counts = collections.Counter(concat_)

    for l in np.unique(labels):
        candidates_with_counts = [(k[0], k[1], c) for k, c in counts.items()
                                  if k[0] == l]
        candidates_with_counts.sort(key=lambda tup: tup[-1], reverse=True)
        map_.append((l, candidates_with_counts[0][1]))

    return np.asarray(map_)


def make_hoof_labels(fx, fy, labels, bins_hoof):

    unq_labels = np.unique(labels)
    bins_label = range(unq_labels.size + 1)
    angle = np.arctan2(fx, fy)
    norm = np.linalg.norm(
        np.concatenate((fx[..., np.newaxis], fy[..., np.newaxis]), axis=-1),
        axis=-1).ravel()

    # Get superpixel indices for each label
    l_mask = [np.where(labels.ravel() == l)[0].tolist() for l in unq_labels]

    # Get angle-bins for each pixel
    b_angles = np.digitize(angle.ravel(), bins_hoof).ravel()

    # Get angle-bins indices for each pixel
    b_mask = [
        np.where(b_angles.ravel() == b)[0].tolist()
        for b in range(1, len(bins_hoof))
    ]

    # Sum norms for each bin and each label
    hoof__ = np.asarray(
        [[np.sum(norm[list(set(l_ + b_))]) for b_ in b_mask] for l_ in l_mask])

    # Normalize w.r.t. L1-norm
    l1_norm = np.sum(hoof__, axis=1).reshape((unq_labels.size, 1))
    hoof__ = np.nan_to_num(hoof__ / l1_norm)

    # Store HOOF in dictionary
    hoof = {unq_labels[i]: hoof__[i, :] for i in range(len(unq_labels))}

    return hoof
