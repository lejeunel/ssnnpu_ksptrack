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
                 dataset,
                 conf,
                 directions=['forward', 'backward'],
                 init_mode='overlap',
                 init_radius=0.07,
                 with_flow=False,
                 hoof_grid_ratio=0.07):

        self.dataset = dataset
        self.init_mode = init_mode
        self.init_radius = init_radius
        self.labels = dataset.labels
        self.c_loc = dataset.centroids_loc
        self.with_flow = with_flow
        self.logger = logging.getLogger('SuperpixelManager')
        self.norm_dist_t = 0.2  #Preliminary threshold (normalized coord.)
        self.hoof_grid_ratio = hoof_grid_ratio
        self.directions = directions
        self.conf = conf
        self.graph = self.make_dicts()

    def make_init_constraint(self):
        if (self.init_mode == 'overlap'):
            return self.make_overlap_constraint()
        elif (self.init_mode == 'radius'):
            return self.make_radius_constraint()

    def make_dicts(self):
        #self.dict_init = self.make_init_dicts()
        if (self.with_flow):
            file_hoof_sps = os.path.join(self.conf.precomp_desc_path,
                                         'hoof_inters_{}_graph.npz'.format(self.init_mode))

            self.graph = self.make_init_constraint()
            flows_path = os.path.join(self.conf.precomp_desc_path, 'flows.npz')

            hoof_extr = HOOFExtractor(self.conf, flows_path, self.labels,
                                      self.hoof_grid_ratio)

            # This will add fields in original graph
            self.graph = hoof_extr.make_hoof_inters(self.graph)
        else:
            self.graph = self.make_init_constraint()

        return self.graph

    def make_radius_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.conf.precomp_desc_path,
                                  'radius_constraint.p')
        if (not os.path.exists(file_graph)):
            f = self.c_loc['frame']
            s = self.c_loc['label']

            g = nx.Graph()

            frames = np.unique(f)

            frames_tup = [(frames[i], frames[i + 1])
                          for i in range(frames.shape[0] - 1)]
            dict_ = dict()

            self.logger.info('Building SP radius dictionary')
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
                edges = [((row[1], row[2]), (row[5], row[6]))
                            for row in df_combs.itertuples()]
                g.add_edges_from(edges)

            bar.close()

            self.logger.info('Saving radius graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading radius graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                g = pk.load(f)
        self.graph_init = g
        return g

    def make_overlap_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.conf.precomp_desc_path,
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
