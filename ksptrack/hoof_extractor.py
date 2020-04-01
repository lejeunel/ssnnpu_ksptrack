import numpy as np
from ksptrack.utils import optical_flow_extractor as oflowx
from ksptrack.utils.regionprops import regionprops
import pickle as pk
import os
from os.path import join as pjoin
import networkx as nx
import pandas as pd
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
                 root_path,
                 desc_dir,
                 labels,
                 n_bins=30,
                 directions=['forward', 'backward']):

        self.directions = directions
        self.root_path = root_path
        self.desc_path = pjoin(root_path, desc_dir)
        self.path_flow = pjoin(self.desc_path, 'flows.npz')
        self.labels = labels
        self.n_bins_hoof = n_bins

    def make_hoof(self):
        """
        Compute HOOF on _grid_ labels
        """

        if (not os.path.exists(self.path_flow)):
            self.calc_oflow()

        file_hoof = os.path.join(self.desc_path, 'hoof.p')

        hoof = []

        if (not os.path.exists(file_hoof)):
            print('computing HOOF on superpixels with {} bins'.format(
                self.n_bins_hoof))
            flows = self.get_flows()
            fvx = np.concatenate(
                (flows['fvx'], flows['fvx'][..., -1][..., None]), axis=-1)
            fvy = np.concatenate(
                (flows['fvy'], flows['fvy'][..., -1][..., None]), axis=-1)
            bvx = np.concatenate(
                (flows['bvx'][..., 0][..., None], flows['bvx']), axis=-1)
            bvy = np.concatenate(
                (flows['bvy'][..., 0][..., None], flows['bvy']), axis=-1)
            pbar = tqdm.tqdm(total=self.labels.shape[0])
            for f in range(self.labels.shape[0]):

                regions_for = regionprops(self.labels[f] + 1,
                                          flow=np.stack((fvx[..., f],
                                                         fvy[..., f])),
                                          n_bins_hoof=self.n_bins_hoof)
                regions_back = regionprops(self.labels[f] + 1,
                                           flow=np.stack((bvx[..., f],
                                                          bvy[..., f])),
                                           n_bins_hoof=self.n_bins_hoof)

                hoof += [{
                    'frame': f,
                    'label': p_for.label - 1,
                    'hoof_forward': p_for.hoof,
                    'hoof_backward': p_back.hoof
                } for p_for, p_back in zip(regions_for, regions_back)]
                pbar.update(1)
            pbar.close()
            print('Saving HOOF to {}'.format(file_hoof))
            self.hoof = pd.DataFrame(hoof)
            self.hoof.to_pickle(file_hoof)
        else:
            print('Loading HOOF {}'.format(file_hoof))
            print('... (delete to re-run)')
            self.hoof = pd.read_pickle(file_hoof)

        return self.hoof

    def make_hoof_inters(self, g, file_out):
        """
        Compute HOOF intersection on sps
        Neighboring superpixels are given by undirected graph g
        """

        if (not os.path.exists(file_out)):
            self.hoof = self.make_hoof()

            edges_ = np.array([e for e in g.edges()]).astype(int)
            for dir_ in self.directions:
                print('Computing HOOF intersections in {} direction'.format(
                    dir_))

                if (dir_ == 'forward'):
                    ind_hoof = self.hoof.columns == 'hoof_forward'
                else:
                    ind_hoof = self.hoof.columns == 'hoof_backward'

                bar = tqdm.tqdm(total=self.labels.shape[0] - 1)
                for f in range(self.labels.shape[0] - 1):

                    # get indices of edges from from f
                    edges_idx_0 = (edges_[..., 0] == f)[:, 0].astype(bool)
                    edges_idx_1 = (edges_[..., 0] == f + 1)[:, 1].astype(bool)
                    edges_idx = np.argwhere(edges_idx_0 + edges_idx_1).ravel()

                    # labels of frame f are on first row
                    labels_0 = edges_[edges_idx, 0, 1]
                    # labels of frame f+1 are on second row
                    labels_1 = edges_[edges_idx, 1, 1]

                    # get HOOF of labels of frame f
                    hoof_0 = self.hoof.loc[self.hoof['frame'] == f].to_numpy()
                    hoof_0 = np.vstack(hoof_0[labels_0, ind_hoof])

                    # get HOOF of labels of frame f+1
                    hoof_1 = self.hoof.loc[self.hoof['frame'] == f +
                                           1].to_numpy()
                    hoof_1 = np.vstack(hoof_1[labels_1, ind_hoof])

                    stack = np.stack((hoof_0, hoof_1))
                    mins = stack.min(axis=0)
                    inters = mins.sum(axis=1)

                    edges_to_add = [((f, l0), (f + 1, l1), {
                        dir_: inter
                    }) for l0, l1, inter in zip(labels_0, labels_1, inters)]
                    g.add_edges_from(edges_to_add)

                    bar.update(1)
                bar.close()

            g = nx.Graph(g)
            print('Saving HOOF intersections to {}'.format(file_out))
            with open(file_out, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
            self.g = g
        else:
            print('Loading HOOF intersections at {}'.format(file_out))
            with open(file_out, 'rb') as f:
                self.g = pk.load(f)

        return self.g

    def get_flows(self):
        flows = dict()
        npzfile = np.load(self.path_flow)
        flows['bvx'] = npzfile['bvx']
        flows['fvx'] = npzfile['fvx']
        flows['bvy'] = npzfile['bvy']
        flows['fvy'] = npzfile['fvy']
        return flows

    def calc_oflow(self):

        oflow_extractor = oflowx.OpticalFlowExtractor(0.012, 0.75, 50., 7, 1,
                                                      30)
        oflow_extractor.extract(self.root_path, self.path_flow)
