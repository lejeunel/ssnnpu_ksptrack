import logging
import numpy as np
import tqdm
import os
import pickle as pk
import networkx as nx
import pandas as pd


class SuperpixelManager:
    """
    Builds undirected graphs to decide which SP connects to which 
    """
    def __init__(self,
                 root_path,
                 desc_dir,
                 labels,
                 desc_df,
                 directions=['forward', 'backward'],
                 init_radius=0.1):

        self.init_radius = init_radius
        self.labels = labels
        self.desc_df = desc_df
        self.directions = directions
        self.root_path = root_path
        self.desc_dir = desc_dir
        self.graph = self.make_dicts()

    def make_dicts(self):
        return self.make_transition_constraint()

    def make_transition_constraint(self):
        """
        Makes a first "gross" filtering of transition.
        """

        file_graph = os.path.join(self.root_path, self.desc_dir,
                                  'transition_constraint.p')
        if (not os.path.exists(file_graph)):
            f = self.desc_df['frame']
            s = self.desc_df['label']

            # this is a directed graph with lowest-frame first
            g = nx.Graph()

            frames = np.unique(f)

            frames_tup = [(frames[i], frames[i + 1])
                          for i in range(frames.shape[0] - 1)]
            dict_ = dict()

            print('Building superpixel transition graph')
            bar = tqdm.tqdm(total=len(frames_tup))
            for i in range(len(frames_tup)):
                bar.update(1)

                f0 = frames_tup[i][0]
                f1 = frames_tup[i][1]
                df0 = self.desc_df.loc[self.desc_df['frame'] == f0].copy(
                    deep=False)
                df1 = self.desc_df.loc[self.desc_df['frame'] == f1].copy(
                    deep=False)
                df0.drop(df0.columns.difference(['frame', 'label', 'x', 'y']),
                         1,
                         inplace=True)
                df1.drop(df1.columns.difference(['frame', 'label', 'x', 'y']),
                         1,
                         inplace=True)

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
                df_overlap = pd.DataFrame(data=ovl,
                                          columns=['label_0', 'label_1'])
                df_overlap['frame_0'] = f0
                df_overlap['frame_1'] = f1
                df_overlap['overlap'] = True

                # set overlap values
                df_all = pd.merge(
                    df_dists,
                    df_overlap,
                    how='left',
                    on=['frame_0', 'label_0', 'frame_1',
                        'label_1']).fillna(False)

                # add edges
                edges = np.stack(
                    (df_all['frame_0'].values, df_all['label_0'].values,
                     df_all['frame_1'].values, df_all['label_1'].values,
                     df_all['dist'].values,
                     df_all['overlap'])).T.astype(np.float16)
                edges = [((e[0], e[1]), (e[2], e[3]),
                          dict(dist=e[4], overlap=e[5])) for e in edges]
                g.add_edges_from(edges)

            bar.close()

            print('Saving transition graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            print('Loading transition graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                g = pk.load(f)
        return g

    def __getstate__(self):
        d = dict(self.__dict__)
        keys_to_ignore = ['labels', 'c_loc', 'dataset']
        for k in keys_to_ignore:
            if k in d.keys():
                del d[k]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
