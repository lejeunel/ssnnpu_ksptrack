import logging
import numpy as np
from ksptrack.hoof_extractor import HOOFExtractor
import tqdm
import os
import pickle as pk
import networkx as nx
import pandas as pd


class SuperpixelManager:
    """
    Builds undirected graphs to decide which SP connects to which 
    Optionally computes Histograms of Oriented Optical Flow intersections
    """
    def __init__(self,
                 root_path,
                 desc_dir,
                 labels,
                 desc_df,
                 directions=['forward', 'backward'],
                 init_radius=0.15,
                 hoof_n_bins=30):

        self.init_radius = init_radius
        self.labels = labels
        self.desc_df = desc_df
        self.hoof_n_bins = hoof_n_bins
        self.logger = logging.getLogger('SuperpixelManager')
        self.directions = directions
        self.root_path = root_path
        self.desc_dir = desc_dir
        self.graph = self.make_dicts()

    def make_dicts(self):
        #self.dict_init = self.make_init_dicts()
        self.graph = self.make_transition_constraint()
        file_hoof_sps = os.path.join(self.root_path,
                                     self.desc_dir,
                                     'hoof_inters_graph.npz')

        hoof_extr = HOOFExtractor(self.root_path,
                                  self.desc_dir,
                                  self.labels,
                                  n_bins=self.hoof_n_bins)

        # This will add fields in original graph
        self.graph = hoof_extr.make_hoof_inters(self.graph, file_hoof_sps)

        return self.graph

    def make_transition_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.root_path,
                                  self.desc_dir,
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

                # this compute distance between all combinations
                df0.columns = ['frame_0', 'label_0', 'x0', 'y0']
                df1.columns = ['frame_1', 'label_1', 'x1', 'y1']
                df0.loc[:, 'key'] = 1
                df1.loc[:, 'key'] = 1
                df_dists = pd.merge(df0, df1, on='key').drop('key', axis=1)
                df_dists['rx'] = df_dists['x0'] - df_dists['x1']
                df_dists['ry'] = df_dists['y0'] - df_dists['y1']
                r = np.concatenate((df_dists['rx'].values.reshape(
                    -1, 1), df_dists['ry'].values.reshape(-1, 1)),
                                   axis=1)
                dists = np.sqrt(r[:, 0]**2 + r[:, 1]**2)
                df_dists['dist'] = dists
                df_dists = df_dists.loc[df_dists['dist'] < self.init_radius]

                # Find overlapping labels between consecutive frames
                l_0 = self.labels[frames_tup[i][0]][..., np.newaxis]
                l_1 = self.labels[frames_tup[i][1]][..., np.newaxis]
                concat_ = np.concatenate((l_0, l_1), axis=-1)
                concat_ = concat_.reshape((-1, 2))
                ovl = np.asarray(list(set(list(map(tuple, concat_)))))
                df_overlap = pd.DataFrame(data=ovl, columns=['label_0',
                                                             'label_1'])
                df_overlap['frame_0'] = f0
                df_overlap['frame_1'] = f1
                df_overlap['overlap'] = True

                # set overlap values
                df_all = pd.merge(df_dists,
                                  df_overlap,
                                  how='left',
                                  on=['frame_0', 'label_0',
                                      'frame_1', 'label_1']).fillna(False)

                # add edges
                edges = np.stack((df_all['frame_0'].values,
                                  df_all['label_0'].values,
                                  df_all['frame_1'].values,
                                  df_all['label_1'].values,
                                  df_all['dist'].values,
                                  df_all['overlap'])).T.astype(np.float16)
                edges = [((e[0], e[1]), (e[2], e[3]),
                          dict(dist=e[4], overlap=e[5]))
                         for e in edges]
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

        file_graph = os.path.join(self.root_path,
                                  self.desc_dir,
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
                l_0 = self.labels[frames_tup[i][0]][..., np.newaxis]
                l_1 = self.labels[frames_tup[i][1]][..., np.newaxis]
                f0 = frames_tup[i][0]
                f1 = frames_tup[i][1]
                concat_ = np.concatenate((l_0, l_1), axis=-1)
                concat_ = concat_.reshape((-1, 2))
                ovl = np.asarray(list(set(list(map(tuple, concat_)))))
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
