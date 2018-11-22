import progressbar
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import cvxopt
from scipy.optimize import minimize
from ksptrack.utils import csv_utils as csv
from ksptrack import tr
from ksptrack import tr_manager as trm
from ksptrack.utils import my_utils as utls
import logging
import functools
import pandas as pd
import pickle as pk
import os
from skimage.draw import circle
from ksptrack.utils.lfda import myLFDA
from sklearn.decomposition import PCA
from boostksp import libksp


class GraphTracking:
    """
    Connects, merges and add edges to graph. Also has KSP algorithm.
    """

    def __init__(self,
                 sps_man=None,
                 tol=0,
                 mode='edge',
                 tau_u=-1,
                 cxx_loglevel="info",
                 cxx_return_edges=True):
        self.logger = logging.getLogger('GraphTracking')

        self.nodeTypes = ['virtual', 'input', 'output', 'none']
        self.source = 0
        self.sink = 1

        self.cxx_loglevel = cxx_loglevel
        self.cxx_return_edges = cxx_return_edges
        self.kspSet = []
        self.tracklets = None
        self.costs = []
        self.nb_sets = 0
        self.mode = mode
        self.direction = None
        self.thr = 0.05  #For bernoulli costs
        self.tol = tol
        self.tau_u = tau_u  # Threshold for HOOF, set to -1 to disable constraint

        self.PCAs = []
        self.trans_transform = None

        self.tls_man = None
        self.sps_man = sps_man

        # This holds the networkx graph
        self.g = None

    def make_tracklets(self, sp_pm):
        #Loop through existing tracklets and add edges.

        tls = [t for t in self.tracklets if (t.blocked == False)]
        with progressbar.ProgressBar(maxval=len(tls)) as bar:
            for i, tl in enumerate(tls):
                bar.update(i)
                this_tracklet_probas = []
                for df_ix in tl.df_ix:
                    this_tracklet_probas.append(sp_pm['proba'][df_ix])
                this_tracklet_probas = np.asarray(this_tracklet_probas)
                this_tracklet_probas = np.clip(
                    this_tracklet_probas, a_min=self.thr, a_max=1 - self.thr)
                w = np.sum(-np.log(
                    np.asarray(this_tracklet_probas) /
                    (1 - np.asarray(this_tracklet_probas))))
                this_e = (tl.in_id, tl.out_id)
                self.g.add_edge(*this_e, weight=w, id_=tls[i].id_)

    def make_init_tracklets(self, sp_pm, labels, centroids_loc, thresh,
                            direction):
        #thresh: Block edges below this proba
        #This also populates the tracklet list

        self.tracklets = []
        tracklet_id = 0
        node_id = np.max((self.source, self.sink)) + 1
        with progressbar.ProgressBar(maxval=sp_pm.shape[0]) as bar:
            for i in range(sp_pm.shape[0]):
                bar.update(i)

                proba = sp_pm['proba'][i]
                if (proba > 1 - self.thr): proba = 1 - self.thr
                if (proba < self.thr): proba = self.thr

                #if (self.direction is not 'forward'):
                if (proba > thresh):
                    blocked = False
                else:
                    blocked = True

                w = -np.log(proba / (1 - proba))
                tl = tr.Tracklet(
                    tracklet_id,
                    node_id,
                    node_id + 1, [[(sp_pm['frame'][i], sp_pm['sp_label'][i])]],
                    [i],
                    direction,
                    length=1,
                    blocked=blocked,
                    marked=False)

                this_e = (tl.in_id, tl.out_id)

                self.tracklets.append(tl)
                if (not blocked):
                    self.g.add_edge(
                        this_e[0], this_e[1], weight=w, id_=tl.id_, sps=tl.sps)
                tracklet_id += 1
                node_id += 2

        self.logger.info('Building tracklet manager')
        self.tls_man = trm.TrackletManager(self.sps_man, self.direction,
                                           self.tracklets, self.n_frames)

    def unblock_tracklets(self, sp_pm, thresh):

        tls_blocked = [t for t in self.tracklets if (t.blocked == True)]
        tls_unblocked = []
        for t in tls_blocked:
            df_ix = t.df_ix
            proba = np.asarray([sp_pm['proba'][ix] for ix in df_ix])
            proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)

            if (np.mean(proba) > thresh):
                t.blocked = False
                tls_unblocked.append(t)
        self.logger.info('Unblocked tracklets: ' + str(len(tls_unblocked)))

        return tls_unblocked

    def merge_tracklets_temporally(self,
                                   tl_sets_to_merge,
                                   sp_pm,
                                   sp_desc,
                                   pm_thr=0.8):

        #Get id of last tracklet
        curr_ids_tls = np.asarray(
            [self.tracklets[i].id_ for i in range(len(self.tracklets))])
        max_id_tls = np.max(curr_ids_tls)
        new_id_tls = max_id_tls + 1

        self.logger.info('Building new tracklets')
        #Build new tracklets
        #this_set = self.kspSet[-1]
        new_tracklets = []
        ids_to_delete = []  #Store IDs of tracklets to delete after merging
        for tl_set in tl_sets_to_merge:
            #Remove source and sink edges
            #Get corresponding (frame,sp_label) tuples
            this_new_sps = [t.sps for t in tl_set]
            this_new_sps = [
                item for sublist in this_new_sps for item in sublist
            ]
            #Get corresponding df_ix indices
            this_new_df_ix = [t.df_ix for t in tl_set]
            this_new_df_ix = [
                item for sublist in this_new_df_ix for item in sublist
            ]
            this_new_sps_arr = np.asarray(this_new_sps).reshape(
                len(this_new_sps), 2)

            proba = np.asarray([sp_pm['proba'][ix] for ix in this_new_df_ix])
            proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)

            # Create new tracklet and corresponding edge to graph
            if (np.max(proba) > pm_thr):
                blocked = False
            else:
                blocked = True

            new_tracklets.append(
                tr.Tracklet(
                    new_id_tls,
                    tl_set[0].in_id,
                    tl_set[-1].out_id,
                    this_new_sps,
                    this_new_df_ix,
                    length=len(this_new_df_ix),
                    blocked=blocked,
                    direction=self.direction))

            new_id_tls += 1

            ids_to_delete.append([t.id_ for t in tl_set])

        #Get ids of tracklets to delete
        ids_to_delete = [item for sublist in ids_to_delete for item in sublist]

        #Remove old tracklets from list
        self.tracklets = [
            t for t in self.tracklets if (t.id_ not in ids_to_delete)
        ]

        # Add new tracklets
        self.tracklets += new_tracklets

        self.logger.info('Updating tracklet dictionary')
        self.tls_man.make_dict(self.tracklets, self.n_frames)

    def is_linkable_entrance_radius(self,
                                    centroids,
                                    loc_2d,
                                    tracklet,
                                    labels,
                                    radius=0.2):
        #Enters if centroid is in gaze "radius"

        centroid_sp = centroids.loc[
            (centroids['frame'] == tracklet.get_in_frame())
            & (centroids['sp_label'] == tracklet.get_in_label())]
        i_gaze, j_gaze = csv.coord2Pixel(loc_2d[0], loc_2d[1],
                                         labels[..., 0].shape[1],
                                         labels[..., 0].shape[0])
        mask = np.zeros(labels[..., 0].shape)
        rr, cc = circle(
            i_gaze,
            j_gaze,
            labels[..., 0].shape[0] * radius,
            shape=labels[..., 0].shape)
        mask[rr, cc] = 1
        centroid_i, centroid_j = utls.norm_to_pix(
            centroid_sp['pos_norm_x'], centroid_sp['pos_norm_y'],
            labels[..., 0].shape[1], labels[..., 0].shape[0])

        #calculate difference of norms for all combinations
        return mask[centroid_i, centroid_j]

    def is_linkable_entrance_sp(self, loc_2d, tracklet, labels):
        #Enters if gaze is on sp
        mask = np.sum(
            np.asarray([
                labels[..., tracklet.get_in_frame()] == tracklet.
                get_in_label()[i] for i in range(len(tracklet.get_in_label()))
            ]),
            axis=0)
        i_gaze, j_gaze = csv.coord2Pixel(loc_2d[0], loc_2d[1],
                                         labels[..., 0].shape[1],
                                         labels[..., 0].shape[0])

        #calculate difference of norms for all combinations
        return mask[i_gaze, j_gaze]

    def proba_trans(self, sp_desc, f1, l1, f2, l2):

        feat_2 = sp_desc.loc[((sp_desc['frame'] == f2) &
                              (sp_desc['sp_label'] == l2)), 'desc'].values[0]
        feat_2_proj = self.trans_transform.transform(feat_2.reshape(1, -1))
        feat_1 = sp_desc.loc[((sp_desc['frame'] == f1) &
                              (sp_desc['sp_label'] == l1)), 'desc'].values[0]
        feat_1_proj = self.trans_transform.transform(feat_1.reshape(1, -1))

        dist = np.linalg.norm(feat_2_proj - feat_1_proj)

        proba = np.exp(-dist**2)
        proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)

        return proba

    def proba_pca(self, sp_desc, f1, l1, f2, l2):

        feat_2 = sp_desc.loc[((sp_desc['frame'] == f2) &
                              (sp_desc['sp_label'] == l2)), 'desc'].values[0]
        feat_2_pca = self.PCAs[f2].transform(feat_2.reshape(1, -1))
        feat_1 = sp_desc.loc[((sp_desc['frame'] == f1) &
                              (sp_desc['sp_label'] == l1)), 'desc'].values[0]
        feat_1_pca = self.PCAs[f2].transform(feat_1.reshape(1, -1))

        dist = np.linalg.norm(feat_2_pca - feat_1_pca)

        proba = np.exp(-dist**2)
        proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)

        return proba

    def trans_probas_tracklets(self, tracklet1, tracklet2, sp_desc, direction,
                               mode):

        if (mode == 'tail'):  # Invert order
            t1 = tracklet2
            t2 = tracklet1
        else:
            t1 = tracklet1
            t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.proba_trans(sp_desc, frame_1, label_1, frame_2, label_2)

        return proba

    def makeFullGraph(self,
                      sp_desc,
                      sp_pom,
                      centroid_locs,
                      points_2d,
                      normNeighbor_in,
                      thresh_aux,
                      tau_u=0,
                      direction='forward',
                      labels=None):

        #Constructs graph from pre-computed costs
        self.n_frames = np.max(sp_desc['frame'].values) + 1
        self.logger.info("Making " + direction + " graph")

        self.g = nx.DiGraph()
        self.direction = direction

        #Auxiliary edges (appearance) creates tracklets, input/output nodes and weight
        self.logger.info('Making/connecting tracklets')
        if (self.tracklets is None):
            self.make_init_tracklets(sp_pom, labels, centroid_locs, thresh_aux,
                                     direction)
        else:
            self.make_tracklets(sp_pom)

        tls = [t for t in self.tracklets if (t.blocked == False)]

        self.make_edges_from_tracklets(tls, sp_desc, centroid_locs, points_2d,
                                       normNeighbor_in, tau_u, labels)

        self.orig_weights = nx.get_edge_attributes(self.g, 'weight')

    def make_trans_transform(self,
                             sp_desc,
                             pm,
                             thresh,
                             n_samps,
                             n_dims,
                             k,
                             pca=False,
                             n_comps_pca=3):

        # descs_cat = utls.concat_arr(sp_desc['desc'])
        descs_cat = np.vstack(sp_desc['desc'].values)
        if (descs_cat.shape[1] == sp_desc.shape[0]):
            descs_cat = descs_cat.T

        if (not pca):

            self.trans_transform = myLFDA(num_dims=n_dims, k=k,
                                          embedding_type='orthonormalized')
            self.trans_transform.fit(descs_cat,
                                     pm['proba'].values,
                                     thresh,
                                     n_samps,
                                     clean_zeros=True)
            self.logger.info(
                'Fitting LFDA (dims,k,n_samps): ({}, {}, {})'.format(
                    n_dims, k, n_samps))

        else:
            self.logger.info(
                'Fitting PCA with {} components'.format(n_comps_pca))
            self.trans_transform = PCA(n_components=n_comps_pca, whiten=True)
            self.trans_transform.fit(descs_cat)

    def connect_sources_cxx_dqn(self, loc2d, centroid_locs, labels, sp_desc,
                                normNeighbor_in):
        # tls = [t for t in self.tracklets if (t.blocked == False)]
        # frames = locs2d['frame']
        tls = [self.tls_man.dict_in[f] for f in np.unique(loc2d['frame'])]
        tls = [item for sublist in tls for item in sublist]
        tls = [t for t in tls if (t.blocked is False)]
        added_edges = list()
        got_bg = True

        edges_from_src = self.g_cxx.out_edges(self.source)
        for tl in tls:
            this_frame = tl.get_in_frame()

            # Check if we have a 2d location on this_frame
            i_gaze, j_gaze = utls.norm_to_pix(loc2d['x'], loc2d['y'],
                                              labels[..., 0].shape[1],
                                              labels[..., 0].shape[0])

            sp_gaze = labels[i_gaze, j_gaze, this_frame]
            e = (int(self.source), int(tl.in_id))

            if (self.is_linkable_entrance_radius(centroid_locs,
                                                 (loc2d['x'], loc2d['y']), tl,
                                                 labels, normNeighbor_in)):
                got_bg = False
                if ((e not in edges_from_src)):

                    added_edges.append(e)

                    proba = self.proba_trans(sp_desc, tl.get_in_frame(),
                                             tl.get_in_label(), this_frame,
                                             sp_gaze)
                    w = -np.log(proba / (1 - proba))
                    self.g_cxx.add_edge(*e, w, -1)

        return added_edges, got_bg

    def connect_sources_cxx(self, locs2d, centroid_locs, labels, sp_desc,
                            normNeighbor_in):
        # tls = [t for t in self.tracklets if (t.blocked == False)]
        # frames = locs2d['frame']
        tls = [self.tls_man.dict_in[f] for f in np.unique(locs2d['frame'])]
        tls = [item for sublist in tls for item in sublist]
        tls = [t for t in tls if (t.blocked is False)]

        added_edges = 0
        for tl in tls:
            this_frame = tl.get_in_frame()
            locs_ = locs2d[locs2d['frame'] == this_frame]

            # Check if we have a 2d location on this_frame
            for _, l in locs_.iterrows():
                i_gaze, j_gaze = utls.norm_to_pix(l['x'], l['y'],
                                                  labels[..., 0].shape[1],
                                                  labels[..., 0].shape[0])

                sp_gaze = labels[i_gaze, j_gaze, this_frame]
                e = (int(self.source), int(tl.in_id))

                if (self.is_linkable_entrance_radius(centroid_locs,
                                                     (l['x'], l['y']), tl,
                                                     labels, normNeighbor_in)):

                    added_edges += 1

                    proba = self.proba_trans(sp_desc, tl.get_in_frame(),
                                             tl.get_in_label(), this_frame,
                                             sp_gaze)
                    w = -np.log(proba / (1 - proba))
                    self.g_cxx.add_edge(*e, w, -1)
                    #self.g.add_edge(*this_e, weight=w, id_=-1)

        return added_edges

    def make_edges_from_tracklets(self, tls, sp_desc, loc, points_2d,
                                  normNeighbor_in, tau_u, labels):

        #Exit edges (time lapse)
        self.logger.info('Connecting exit edges')
        for i in range(len(tls)):
            this_e = (tls[i].out_id, self.sink)
            self.g.add_edge(*this_e, weight=0, id_=-1)

        #Entrance edges (location)
        self.logger.info('Connecting entrance edges')

        for i in range(len(tls)):
            this_frame = loc['frame'][tls[i].df_ix[0]]
            # Check if we have a 2d location on this_frame
            if (this_frame in points_2d[:, 0]):
                loc_2d = points_2d[points_2d[:, 0] == this_frame, 3:5].ravel()
                i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0], loc_2d[1],
                                                  labels[..., 0].shape[1],
                                                  labels[..., 0].shape[0])

                sp_gaze = labels[i_gaze, j_gaze, this_frame]

                #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
                if (self.is_linkable_entrance_radius(loc, loc_2d, tls[i],
                                                     labels, normNeighbor_in)):

                    this_e = (int(self.source), int(tls[i].in_id))

                    proba = self.proba_trans(sp_desc, tls[i].get_in_frame(),
                                             tls[i].get_in_label(), this_frame,
                                             sp_gaze)
                    w = -np.log(proba / (1 - proba))
                    self.g.add_edge(*this_e, weight=w, id_=-1)

        # Transition edges
        self.logger.info('Connecting transition edges')

        #if(self.direction == 'forward'):
        mode = 'head'
        #else:
        #    mode = 'tail'
        with progressbar.ProgressBar(maxval=len(tls)) as bar:
            for i in range(len(tls)):
                bar.update(i)
                this_tracklet = tls[i]

                linkable_tracklets = self.tls_man.get_linkables(
                    this_tracklet,
                    tau_u=tau_u,
                    mode=mode,
                    direction=self.direction)

                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(
                        this_tracklet,
                        linkable_tracklets[j],
                        sp_desc,
                        self.direction,
                        mode='head')
                    w = -np.log(probas / (1 - probas))
                    this_e = (this_tracklet.out_id,
                              linkable_tracklets[j].in_id)
                    self.g.add_edge(*this_e, weight=w, id_=-1)

    def run(self):
        self.copy_cxx()
        self.kspSet = self.g_cxx.run()
        return self.kspSet

    def run_nocopy(self):
        self.kspSet = self.g_cxx.run()
        return self.kspSet

    def copy_cxx(self):
        # This holds the c++ graph
        self.g_cxx = libksp.ksp()
        self.g_cxx.config(
            self.source,
            self.sink,
            loglevel=self.cxx_loglevel,
            min_cost=True,
            return_edges=self.cxx_return_edges)

        self.logger.info('Copying graph...')
        for e in self.g.edges():
            self.g_cxx.add_edge(
                int(e[0]), int(e[1]), self.g[e[0]][e[1]]['weight'],
                int(self.g[e[0]][e[1]]['id_']))
        self.logger.info('done.')

    def save_graph(self, path):
        nx.write_gpickle(self.g, path)

    def load_graph(self, path):
        self.g = nx.read_gpickle(path)

    def save_all(self, path):
        graph_save_path = path + '_graph.p'
        sps_man_save_path = path + '_sp_man.p'
        tls_man_save_path = path + '_tls_man.p'
        self.logger.info('Saving graph to {}'.format(graph_save_path))
        self.save_graph(graph_save_path)
        self.logger.info('Saving sp manager to {}'.format(sps_man_save_path))
        pk.dump(self.sps_man, open(sps_man_save_path, 'wb'))
        self.logger.info('Saving tls manager to {}'.format(tls_man_save_path))
        pk.dump(self.tls_man, open(tls_man_save_path, 'wb'))

    def load_all(self, path):
        self.load_graph(path + '_graph.p')
        self.sps_man = pk.load(open(path + '_sp_man.p', 'rb'))
        self.tls_man = pk.load(open(path + '_tls_man.p', 'rb'))
