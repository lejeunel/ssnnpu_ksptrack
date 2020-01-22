import logging
import numpy as np
from ksptrack.utils import my_utils as utls
from ksptrack.hoof_extractor import HOOFExtractor
import tqdm
import os
import pickle as pk
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tqdm


class SuperpixelManager:
    """
    Builds undirected graphs to decide which SP connects to which 
    Optionally computes Histograms of Oriented Optical Flow intersections
    """

    def __init__(self,
                 dm,
                 directions=['forward', 'backward'],
                 init_radius=0.15,
                 hoof_n_bins=30):

        self.dm = dm
        self.init_radius = init_radius
        self.labels = dm.labels
        self.c_loc = dm.centroids_loc
        self.hoof_n_bins = hoof_n_bins
        self.logger = logging.getLogger('SuperpixelManager')
        self.directions = directions
        self.desc_path = dm.desc_path
        self.graph = self.make_dicts()


    def make_dicts(self):
        #self.dict_init = self.make_init_dicts()
        self.graph = self.make_transition_constraint()
        file_hoof_sps = os.path.join(self.desc_path,
                                        'hoof_inters_graph.npz')

        hoof_extr = HOOFExtractor(self.dm.root_path,
                                  self.dm.desc_dir,
                                  self.labels,
                                  n_bins=self.hoof_n_bins)

        # This will add fields in original graph
        self.graph = hoof_extr.make_hoof_inters(self.graph, file_hoof_sps)

        return self.graph

    def make_transition_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.desc_path,
                                  'transition_constraint.p')
        if (not os.path.exists(file_graph)):
            f = self.c_loc['frame']
            s = self.c_loc['label']

            # this is a directed graph with lowest-frame first
            g = nx.DiGraph()

            frames = np.unique(f)

            frames_tup = [(frames[i], frames[i + 1])
                          for i in range(frames.shape[0] - 1)]
            dict_ = dict()

            self.logger.info('Building superpixel transition graph')
            bar = tqdm.tqdm(total=len(frames_tup))
            for i in range(len(frames_tup)):
                bar.update(1)

                f0 = frames_tup[i][0]
                f1 = frames_tup[i][1]
                df0 = self.c_loc.loc[self.c_loc['frame'] == f0].copy(
                    deep=False)
                df1 = self.c_loc.loc[self.c_loc['frame'] == f1].copy(
                    deep=False)
                df0.columns = ['frame_0', 'label_0', 'x0', 'y0']
                df1.columns = ['frame_1', 'label_1', 'x1', 'y1']
                df0.loc[:, 'key'] = 1
                df1.loc[:, 'key'] = 1
                df_combs = pd.merge(df0, df1, on='key').drop('key', axis=1)
                df_combs['rx'] = df_combs['x0'] - df_combs['x1']
                df_combs['ry'] = df_combs['y0'] - df_combs['y1']
                r = np.concatenate((df_combs['rx'].values.reshape(-1, 1),
                                    df_combs['ry'].values.reshape(-1, 1)),
                                    axis=1)
                dists = np.linalg.norm(r, axis=1)
                df_combs['dist'] = dists
                df_combs = df_combs.loc[
                    df_combs['dist'] < self.init_radius]

                # add edges with overlap=False by default, will change below
                edges = [((row[1], row[2]), (row[5], row[6]),
                          dict(dist=row[10], overlap=False))
                            for row in df_combs.itertuples()]
                g.add_edges_from(edges)

                # Find overlapping labels between consecutive frames
                l_0 = self.labels[..., frames_tup[i][0]][..., np.newaxis]
                l_1 = self.labels[..., frames_tup[i][1]][..., np.newaxis]
                concat_ = np.concatenate((l_0, l_1), axis=-1)
                concat_ = concat_.reshape((-1, 2))
                ovl = np.asarray(list(set(list(map(tuple, concat_)))))
                edges = [((f0, l[0]), (f1, l[1]), dict(overlap=True))
                         for l in ovl]
                g.add_edges_from(edges)

            bar.close()

            self.logger.info('Saving transition graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading transition graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                g = pk.load(f)
        self.graph_init = g
        return g

    def make_overlap_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.desc_path,
                                  'overlap_constraint.p')
        if (not os.path.exists(file_graph)):
            f = self.c_loc['frame']
            s = self.c_loc['label']

            g = nx.Graph()

            frames = np.unique(f)

            frames_tup = [(frames[i], frames[i + 1])
                          for i in range(frames.shape[0] - 1)]
            dict_ = dict()

            self.logger.info('Building SP overlap dictionary')

            bar = tqdm.tqdm(total=len(frames_tup))
            for i in range(len(frames_tup)):
                bar.update(1)

                # Find overlapping labels between consecutive frames
                l_0 = self.labels[..., frames_tup[i][0]][..., np.newaxis]
                l_1 = self.labels[..., frames_tup[i][1]][..., np.newaxis]
                f0 = frames_tup[i][0]
                f1 = frames_tup[i][1]
                concat_ = np.concatenate((l_0, l_1), axis=-1)
                concat_ = concat_.reshape((-1, 2))
                ovl = np.asarray(list(set(list(map(tuple, concat_)))))
                ovl_list = list()
                for l0 in np.unique(ovl[:, 0]):
                    ovl_ = ovl[ovl[:, 0] == l0, 1].tolist()
                    edges = [((f0, l0), (f1, l1)) for l1 in ovl_]
                    g.add_edges_from(edges)

            bar.close()

            self.logger.info('Saving overlap graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading overlap graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                g = pk.load(f)
        self.graph_init = g
        return g


    def __getstate__(self):
        d = dict(self.__dict__)
        keys_to_ignore = ['logger', 'labels', 'c_loc', 'dataset']
        for k in keys_to_ignore:
            if k in d.keys():
                del d[k]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
