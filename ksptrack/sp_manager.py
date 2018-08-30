import logging
import numpy as np
from ksptrack.utils import my_utils as utls
from ksptrack.hoof_extractor import HOOFExtractor
import progressbar
import os
import pickle as pk
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

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
                 with_flow = False,
                 hoof_grid_ratio=0.07):

        self.dataset = dataset
        self.init_mode = init_mode
        self.init_radius = init_radius
        self.labels = dataset.get_labels()
        self.c_loc = dataset.centroids_loc
        self.with_flow = with_flow
        self.logger = logging.getLogger('SuperpixelManager')
        self.norm_dist_t = 0.2 #Preliminary threshold (normalized coord.)
        self.hoof_grid_ratio = hoof_grid_ratio
        self.directions = directions
        self.conf = conf
        self.graph = self.make_dicts()

    def make_init_constraint(self):
        if(self.init_mode == 'overlap'):
            return self.make_overlap_constraint()
        elif(self.init_mode == 'radius'):
            return self.make_radius_constraint()
            
    def make_dicts(self):
        #self.dict_init = self.make_init_dicts()
        if(self.with_flow):
            self.graph = self.make_init_constraint()
            flows_path = os.path.join(self.conf.precomp_desc_path,
                                      'flows.npz')

            hoof_extr = HOOFExtractor(self.conf,
                                      flows_path,
                                      self.labels,
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
        if(not os.path.exists(file_graph)):
            f = self.c_loc['frame']
            s = self.c_loc['sp_label']

            g = nx.Graph()

            frames = np.unique(f)

            frames_tup = [(frames[i],frames[i+1])
                      for i in range(frames.shape[0]-1)]
            dict_ = dict()

            self.logger.info('Building SP radius dictionary')
            with progressbar.ProgressBar(maxval=len(frames_tup)) as bar:
                for i in range(len(frames_tup)):
                    bar.update(i)

                    f0 = frames_tup[i][0]
                    f1 = frames_tup[i][1]
                    df0 = self.c_loc.loc[self.c_loc['frame'] == f0].copy(deep=False)
                    df1 = self.c_loc.loc[self.c_loc['frame'] == f1].copy(deep=False)
                    df0.columns = ['frame_0', 'sp_label_0',
                                         'x0', 'y0']
                    df1.columns = ['frame_1', 'sp_label_1',
                                         'x1', 'y1']
                    df0.loc[:,'key'] = 1
                    df1.loc[:,'key'] = 1
                    df_combs = pd.merge(df0,df1,on='key').drop('key',axis=1)
                    df_combs['rx'] = df_combs['x0'] - df_combs['x1']
                    df_combs['ry'] = df_combs['y0'] - df_combs['y1']
                    r = np.concatenate((df_combs['rx'].values.reshape(-1,1),
                                            df_combs['ry'].values.reshape(-1,1)),
                                           axis=1)
                    dists = np.linalg.norm(r, axis=1)
                    df_combs['dist'] = dists
                    df_combs = df_combs.loc[df_combs['dist'] < self.init_radius]
                    edges = [((row[1], row[2]),
                              (row[5], row[6]))
                             for row in df_combs.itertuples()]
                    g.add_edges_from(edges)


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
        if(not os.path.exists(file_graph)):
            f = self.c_loc['frame']
            s = self.c_loc['sp_label']

            g = nx.Graph()

            frames = np.unique(f)

            frames_tup = [(frames[i],frames[i+1])
                      for i in range(frames.shape[0]-1)]
            dict_ = dict()

            self.logger.info('Building SP overlap dictionary')
            with progressbar.ProgressBar(maxval=len(frames_tup)) as bar:
                for i in range(len(frames_tup)):
                    bar.update(i)

                    # Find overlapping labels between consecutive frames
                    l_0 = self.labels[..., frames_tup[i][0]][..., np.newaxis]
                    l_1 = self.labels[..., frames_tup[i][1]][..., np.newaxis]
                    f0 = frames_tup[i][0]
                    f1 = frames_tup[i][1]
                    concat_ = np.concatenate((l_0,l_1), axis=-1)
                    concat_ = concat_.reshape((-1,2))
                    ovl = np.asarray(list(set(list(map(tuple, concat_)))))
                    ovl_list = list()
                    for l0 in np.unique(ovl[:, 0]):
                        ovl_ = ovl[ovl[:,0] == l0, 1].tolist()
                        edges = [((f0, l0), (f1, l1)) for l1 in ovl_]
                        g.add_edges_from(edges)

            self.logger.info('Saving overlap graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
        else:
            self.logger.info('Loading overlap graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                g = pk.load(f)
        self.graph_init = g
        return g

    def make_hoof_inter_graph(self):
        #We build an undirected graph where nodes are tuples (frame, sp_label)
        hoof = self.make_hoof_dict()

        file_graph = os.path.join(self.conf.precomp_desc_path,
                                 'hoof_constraint.p')

        frames = np.unique(self.c_loc['frame'])
        # Compute histograms
        if(not os.path.exists(file_graph)):
            for direction in self.directions:
                self.logger.info('Computing HOOF intersections in {} direction'\
                                 .format(direction))
                with progressbar.ProgressBar(maxval=len(g.nodes())) as bar:
                    for i, n in enumerate(g.nodes()):
                        bar.update(i)
                        for c in g[n]: # candidates
                            start_f = n[0]
                            end_f = c[0]
                            start_s = n[1] # start sp_label
                            end_s = c[1]
                            if(direction == 'forward'):
                                inter_hoof = utls.hist_inter(
                                    hoof[direction][start_f][start_s],
                                    hoof[direction][start_f][end_s])

                            else:
                                inter_hoof = utls.hist_inter(
                                    hoof[direction][end_f][start_s],
                                    hoof[direction][end_f][end_s])
                            g[n][c][direction] = inter_hoof

                self.logger.info('Saving hoof graph to ' + file_graph)
                with open(file_graph, 'wb') as f:
                    pk.dump(g, f, pk.HIGHEST_PROTOCOL)
                self.graph_hoof = g
            return self.graph_hoof
        else:
            self.logger.info('Loading graph... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                self.dict_hoof = f
                return pk.load(f)

    def make_hoof_dict(self):
        """
        Compute for each direction a HOOF dictionary
        """

        g = self.make_overlap_constraint()
        edges = g.edges()

        file_graph = os.path.join(self.conf.precomp_desc_path,
                                 'hoof_constraint.p')

        # Compute histograms
        if(not os.path.exists(file_graph)):
            for direction in self.directions:
                if(direction == 'forward'):
                    edges = [e for e in edges
                             if(e[1][0] > e[0][0])]
                    keys = ['fvx', 'fvy']
                else:
                    edges = [e for e in edges
                             if(e[0][0] > e[1][0])]
                    keys = ['bvx', 'bvy']

                frames = sorted(set([e[0][0] for e in edges]))

                hoof_ = dict()
                bins_hoof = np.linspace(-np.pi/2,np.pi/2,self.conf.n_bins_hoof+1)

                self.logger.info('Getting optical flows')
                flows = self.dataset.get_flows()

                # duplicate flows at limits
                fx = flows[keys[0]]
                fy = flows[keys[1]]

                self.logger.info('Computing HOOF in {} direction'.format(direction))

                with progressbar.ProgressBar(maxval=len(frames)) as bar:
                    for f, p in zip(frames, range(len(frames))):

                        edges_ = [e for e in edges if(e[0][0] == f)]

                        if(direction == 'forward'):
                            f_start = edges_[0][0][0]
                            f_end = edges_[0][1][0]
                        else:
                            f_start = edges_[0][1][0]
                            f_end = edges_[0][0][0]

                        hoof_start = make_hoof_labels(fx[..., p],
                                                    fy[..., p],
                                                    self.labels[..., f_start],
                                                    bins_hoof)
                        hoof_end = make_hoof_labels(fx[..., p],
                                                    fy[..., p],
                                                    self.labels[..., f_end],
                                                    bins_hoof)

                        for e in edges_:
                            g[e[0]][e[1]][direction] = utls.hist_inter(
                                hoof_start[e[0][1]],
                                hoof_end[e[1][1]])
                        bar.update(p)


            self.logger.info('Saving hoof graph to ' + file_graph)
            with open(file_graph, 'wb') as f:
                pk.dump(g, f, pk.HIGHEST_PROTOCOL)
            self.graph = g
            return self.graph
        else:
            self.logger.info('Loading hoof dict... (delete to re-run)')
            with open(file_graph, 'rb') as f:
                self.graph =  pk.load(f)
                return self.graph

