import progressbar
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from labeling.utils import my_utils as utls
import labeling.tr_manager as trm
import itertools as it
import cvxopt
from scipy.optimize import minimize
from labeling.utils import csv_utils as csv
import logging
import functools
import pandas as pd
import pickle as pk
import os
from labeling import tr
from labeling import tr_manager
from skimage.draw import circle
from metric_learn import LFDA
from sklearn.decomposition import PCA
from boostksp import libksp

class GraphTracking:
    """
    Connects, merges and add edges to graph. Also has KSP algorithm.
    """

    def __init__(self,sps_mans, tol=0, mode='edge', tau_u = -1):
        self.logger = logging.getLogger('GraphTracking')

        self.nodeTypes = ['virtual', 'input', 'output', 'none']
        self.source = 0
        self.sink = 1
        self.curr_node_id = 2
        self.kspSet = None
        self.tracklets = None
        self.cost = None
        self.nb_sets = 0
        self.mode = mode
        self.direction = None
        self.thr = 0.05 #For bernoulli costs
        self.tol = tol
        self.tau_u = tau_u # Threshold for HOOF, set to -1 to disable constraint

        self.means_feats = []
        self.covs_inv_feats = []
        self.PCAs = []
        self.trans_transform = None

        self.tls_man = None
        self.sps_mans = sps_mans

        self.g = libksp.ksp()
        self.g.config(self.source,
                      self.sink,
                      loglevel="info",
                      min_cost=True)
                      #l_max=1)

    def run(self):
        self.kspSet = self.g.run()

        return self.kspSet

    def make_tracklets(self, sp_pm, thresh,direction):
        #Loop through existing (already merged) tracklets and assign mean proba.
        #self.tracklets contains both merge and not-merge (length 1) tracklets.
        #sp_pm is "re-looped" to add previously discarded tracklets that now
        #correspond to proba > self.thr

        tls = [t for t in self.tracklets if(t.blocked == False)]
        with progressbar.ProgressBar(maxval=len(tls)) as bar:
            for i in range(len(tls)):
                bar.update(i)
                this_tracklet_probas = []
                for df_ix in tls[i].df_ix:
                    this_tracklet_probas.append(sp_pm['proba'][df_ix])
                this_tracklet_probas = np.asarray(this_tracklet_probas)
                mean_proba = np.mean(this_tracklet_probas)
                if(mean_proba > 1 - self.thr): mean_proba = 1 - self.thr
                if(mean_proba < self.thr): mean_proba = self.thr
                if(mean_proba > thresh):
                    this_tracklet_probas = np.clip(this_tracklet_probas,
                                                a_min=self.thr,
                                                a_max=1-self.thr)
                    w = np.sum(-np.log(np.asarray(this_tracklet_probas) / (1 - np.asarray(this_tracklet_probas))))
                    self.g.add_edge(int(tls[i].in_id),
                                    int(tls[i].out_id),
                                    weight=w,
                                    id=tls[i].id_)

        #Get df_ix that are left after previous construction
        taken_df_ix = [t.df_ix for t in tls]
        taken_df_ix = [item for sublist in taken_df_ix for item in sublist]
        all_df_ix = sp_pm.index.values
        remaining_df_ix = [df_ix for df_ix in all_df_ix if(df_ix not in taken_df_ix)]

        max_id = self.tracklets[-1].id_
        curr_id = max_id+1

        with progressbar.ProgressBar(maxval=len(remaining_df_ix)) as bar:
            for i in range(len(remaining_df_ix)):
                bar.update(i)
                proba = sp_pm['proba'][remaining_df_ix[i]]
                if (proba > 1 - self.thr): proba = 1 - self.thr
                if (proba < self.thr): proba = self.thr

                #if (self.direction is not 'forward'):
                if(proba > thresh):
                    w = -np.log(proba / (1 - proba))
                    this_tracklet = tr.Tracklet(curr_id,
                                            [[(sp_pm['frame'][remaining_df_ix[i]],
                                        sp_pm['sp_label'][remaining_df_ix[i]])]],
                                            [remaining_df_ix[i]],
                                            direction,
                                            blocked=False,
                                            marked=False)
                    self.tracklets.append(this_tracklet)

                    self.g.add_edge(int(this_tracklet.in_id),
                                    int(this_tracklet.out_id),
                                    weight=w,
                                    id=this_tracklet.id_)
                    curr_id +=1

    def make_init_tracklets(self,
                            sp_pm,
                            labels,
                            centroids_loc,
                            thresh,direction):
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
                if(proba > thresh):
                    blocked = False
                else:
                    blocked = True

                w = -np.log(proba / (1 - proba))
                tl = tr.Tracklet(tracklet_id,
                                 node_id,
                                 node_id+1,
                                 [[(sp_pm['frame'][i],
                                    sp_pm['sp_label'][i])]],
                                 [i],
                                 direction,
                                 scale=1,
                                 length=1,
                                 blocked=blocked,
                                 marked=False)

                self.tracklets.append(tl)
                if(not blocked):
                    self.g.add_edge(int(tl.in_id),
                                    int(tl.out_id),
                                    weight=w,
                                    id=tl.id_)
                tracklet_id +=1
                node_id += 2

        self.logger.info('Building tracklet manager')
        self.tls_man = trm.TrackletManager(self.sps_mans,
                                       self.tracklets,
                                       self.n_frames)


    def unblock_tracklets(self,sp_pm,thresh):

        tls_blocked = [t for t in self.tracklets if(t.blocked == True)]
        tls_unblocked = []
        for t in tls_blocked:
            df_ix = t.df_ix
            proba = np.asarray([sp_pm['proba'][ix] for ix in df_ix])
            proba = np.clip(proba, a_min=self.thr,a_max=1-self.thr)

            if(np.mean(proba) > thresh):
                t.blocked = False
                tls_unblocked.append(t)
        self.logger.info('Unblocked tracklets: ' + str(len(tls_unblocked)))

        return tls_unblocked

    def merge_tracklets_temporally(self,
                                   loc,
                                   sp_pm,
                                   sp_desc,
                                   points_2d,
                                   normNeighbor_in,
                                   thresh,
                                   labels,
                                   tau_u):

        #Get id of last tracklet
        curr_ids = np.asarray([self.tracklets[i].id_ for i in range(len(self.tracklets))])
        max_id = np.max(curr_ids)
        new_id = max_id + 1

        self.logger.info('Building new tracklets and adding to graph')
        #Build new tracklets
        paths = self.kspSet # this gives ids of tracklets
        new_tracklets = []
        tls_to_delete = [] #Store IDs of tracklets to delete after merging
        for p in paths:
            tls = [t for t in self.tracklets if(t.id_ in p)]
            #Get corresponding (frame,sp_label) tuples
            new_sps = [t.sps for t in tls]
            new_sps = [item for sublist in new_sps for item in sublist]
            #Get corresponding df_ix indices
            new_df_ix = [t.df_ix for t in tls]
            new_df_ix = [item for sublist in new_df_ix for item in sublist]
            new_sps_arr = np.asarray(new_sps).reshape(len(new_sps),2)
            arg_sort_frames = np.argsort(new_sps_arr[:,0]).tolist()
            if(self.direction == 'backward'):
                arg_sort_frames = arg_sort_frames[::-1]
                new_sps = [new_sps[i] for i in arg_sort_frames]
                new_df_ix = [new_df_ix[i] for i in arg_sort_frames]
            else:
                arg_sort_frames = arg_sort_frames
                new_sps = [new_sps[i] for i in arg_sort_frames]
                new_df_ix = [new_df_ix[i] for i in arg_sort_frames]

            proba = np.asarray([sp_pm['proba'][ix] for ix in new_df_ix])
            proba = np.clip(proba, a_min=self.thr,a_max=1-self.thr)

            # Create new tracklet and corresponding edge to graph
            if(np.mean(proba) > thresh):
                blocked = False
            else:
                blocked = True

            new_tracklets.append(tr.Tracklet(new_id,
                                             tls[0].in_id,
                                             tls[-1].out_id,
                                            new_sps,
                                            new_df_ix,
                                            length=len(new_df_ix),
                                            blocked=blocked,
                                            direction=self.direction))

            if(blocked == False):
                w = np.sum([-np.log(p/(1-p)) for p in proba])
                if(len(tls)>1):

                    w_trans = 0
                    for i in range(len(tls)-1):
                        t1 = tls[i]
                        t2 = tls[i+1]
                        p = self.trans_probas_tracklets(t1,
                                                        t2,
                                                        sp_desc,
                                                        self.direction,
                                                        mode='head')
                        w_trans += -np.log(p/(1-p))
                else:
                    w_trans = 0

                self.g.add_edge(int(new_tracklets[-1].in_id),
                                int(new_tracklets[-1].out_id),
                                weight=w+w_trans,
                                id=int(new_id))
            new_id += 1

            tls_to_delete.append(tls)

        #Get ids of tracklets to delete
        tls_to_delete = [item for sublist in tls_to_delete for item in sublist]

        #Remove nodes of old tracklets (removes all edges connected to it by default)
        self.logger.info('Removing old tracklets')
        for t in tls_to_delete:
            self.g.remove_edge(int(t.in_id), int(t.out_id))

        #Remove old tracklets from list
        ids_to_delete = [t.id_ for t in tls_to_delete]
        self.tracklets = [t for t in self.tracklets if(t.id_ not in ids_to_delete)]

        # Unblock tracklets with probas above thresh
        tls_unblocked = self.unblock_tracklets(sp_pm,thresh)

        #Make new list of tracklets and update tracklet manager
        self.tracklets += new_tracklets

        new_and_unblocked_tls = new_tracklets + tls_unblocked

        self.logger.info('Connecting entrance edges on new tracklets')
        for i in range(len(new_tracklets)):
            frame = loc['frame'][new_tracklets[i].df_ix[0]]
            loc_2d = points_2d[frame,3:5]
            i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                              loc_2d[1],
                                              labels[...,0].shape[1],
                                              labels[...,0].shape[0])

            sp_gaze = labels[i_gaze,j_gaze,frame]

            #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
            if(self.is_linkable_entrance_radius(loc,
                                                loc_2d,
                                                new_tracklets[i],
                                                labels,
                                                normNeighbor_in)):

                this_e = (int(self.source), int(new_tracklets[i].in_id))

                proba = self.proba_trans(sp_desc,
                                       new_tracklets[i].get_in_frame(),
                                       new_tracklets[i].get_in_label(),
                                       frame,
                                       sp_gaze)
                w = -np.log(proba / (1 - proba))
                self.g.add_edge(*this_e, weight=w)

        #Entrance edges (location)
        self.logger.info('Connecting entrance edges on unblocked tracklets')
        for i in range(len(tls_unblocked)):
            frame = loc['frame'][tls_unblocked[i].df_ix[0]]
            loc_2d = points_2d[frame,3:5]
            i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                              loc_2d[1],
                                              labels[...,0].shape[1],
                                              labels[...,0].shape[0])

            sp_gaze = labels[i_gaze,j_gaze,frame]

            #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
            if(self.is_linkable_entrance_radius(loc,
                                                loc_2d,
                                                tls_unblocked[i],
                                                labels,
                                                normNeighbor_in)):

                this_e = (int(self.source), int(tls_unblocked[i].in_ind))

                proba = self.proba_trans(sp_desc,
                                       tls_unblocked[i].get_in_frame(),
                                       tls_unblocked[i].get_in_label(),
                                       frame,
                                       sp_gaze)
                w = -np.log(proba / (1 - proba))
                self.g.add_edge(*this_e, weight=w)

        self.logger.info('Connecting exit edges on new and unblocked tracklets')
        for t in new_and_unblocked_tls:
            this_e = (int(t.out_id), int(self.sink))
            self.g.add_edge(*this_e, weight=0.)

        tls = [t for t in self.tracklets if(t.blocked == False)]
        self.logger.info('Updating tracklet dictionary')
        self.tls_man.make_dict(self.tracklets,self.n_frames)

        # Transition edges
        #ids_new_unblocked =
        self.logger.info('Connecting transition edges on new and unblocked tracklets')
        with progressbar.ProgressBar(maxval=len(new_and_unblocked_tls)) as bar:
            for tl in new_and_unblocked_tls:

                bar.update(i)
                linkable_tracklets = self.tls_man.get_linkables(
                    tl,
                    tau_u=tau_u,
                    direction=self.direction,
                    mode='head')

                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(tl,
                                                         linkable_tracklets[j],
                                                         sp_desc,
                                                         self.direction,
                                                         mode='head')
                    w = -np.log(probas / (1 - probas))
                    this_e = (int(tl.out_id),
                              int(linkable_tracklets[j].in_id))
                    self.g.add_edge(*this_e, w)

                linkable_tracklets = self.tls_man.get_linkables(
                    tl,
                    tau_u=tau_u,
                    direction=self.direction,
                    mode='tail')
                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(tl,
                                                         linkable_tracklets[j],
                                                         sp_desc,
                                                         self.direction,
                                                         mode='tail')
                    w = -np.log(probas / (1 - probas))
                    this_e = (int(linkable_tracklets[j].out_id),
                                int(tl.in_id))
                    self.g.add_edge(*this_e, w)

    def is_linkable_entrance_radius(self,centroids,
                                    loc_2d,
                                    tracklet,
                                    labels,
                                    radius=0.2):
        #Enters if centroid is in gaze "radius"

        centroid_sp = centroids.loc[(centroids['frame'] == tracklet.get_in_frame()) & (centroids['sp_label'] == tracklet.get_in_label())]
        i_gaze,j_gaze = csv.coord2Pixel(loc_2d[0],
                                        loc_2d[1],
                                        labels[...,0].shape[1],
                                        labels[...,0].shape[0])
        mask = np.zeros(labels[...,0].shape)
        rr, cc = circle(i_gaze,j_gaze,
                        labels[...,0].shape[0]*radius,
                        shape=labels[...,0].shape)
        mask[rr,cc] = 1
        centroid_i, centroid_j = utls.norm_to_pix(centroid_sp['pos_norm_x'],
                                                  centroid_sp['pos_norm_y'],
                                                  labels[...,0].shape[1],
                                                  labels[...,0].shape[0])

        #calculate difference of norms for all combinations
        return mask[centroid_i,centroid_j]


    def is_linkable_entrance_sp(self,loc_2d, tracklet, labels ):
        #Enters if gaze is on sp
        mask = np.sum(np.asarray([labels[...,tracklet.get_in_frame()] == tracklet.get_in_label()[i] for i in range(len(tracklet.get_in_label()))]),axis=0)
        i_gaze,j_gaze = gaze.gazeCoord2Pixel(loc_2d[0],loc_2d[1],labels[...,0].shape[1],labels[...,0].shape[0])

        #calculate difference of norms for all combinations
        return mask[i_gaze,j_gaze]

    def proba_trans(self,
                  sp_desc,
                  f1,
                  l1,
                  f2,
                  l2):

        feat_2 = sp_desc.loc[((sp_desc['frame'] == f2) & (sp_desc['sp_label'] == l2)), 'desc'].as_matrix()[0]
        feat_2_proj = self.trans_transform.transform(feat_2.reshape(1,-1))
        feat_1 = sp_desc.loc[((sp_desc['frame'] == f1) & (sp_desc['sp_label'] == l1)), 'desc'].as_matrix()[0]
        feat_1_proj = self.trans_transform.transform(feat_1.reshape(1,-1))

        dist = np.linalg.norm(feat_2_proj - feat_1_proj)

        proba = np.exp(-dist**2)
        proba = np.clip(proba, a_min=self.thr, a_max=1-self.thr)

        return proba

    def proba_pca(self,
                  sp_desc,
                  f1,
                  l1,
                  f2,
                  l2):

        feat_2 = sp_desc.loc[((sp_desc['frame'] == f2) & (sp_desc['sp_label'] == l2)), 'desc'].as_matrix()[0]
        feat_2_pca = self.PCAs[f2].transform(feat_2.reshape(1,-1))
        feat_1 = sp_desc.loc[((sp_desc['frame'] == f1) & (sp_desc['sp_label'] == l1)), 'desc'].as_matrix()[0]
        feat_1_pca = self.PCAs[f2].transform(feat_1.reshape(1,-1))

        dist = np.linalg.norm(feat_2_pca - feat_1_pca)

        proba = np.exp(-dist**2)
        proba = np.clip(proba, a_min=self.thr, a_max=1-self.thr)

        return proba

    def trans_probas_tracklets(self,
                                    tracklet1,
                                    tracklet2,
                                    sp_desc,
                                    direction,
                                    mode):

        if(mode == 'tail'): # Invert order
            t1 = tracklet2
            t2 = tracklet1
        else:
            t1 = tracklet1
            t2 = tracklet2

        frame_1 = t1.get_out_frame()
        label_1 = t1.get_out_label()
        frame_2 = t2.get_in_frame()
        label_2 = t2.get_in_label()

        proba = self.proba_trans(sp_desc,frame_1,label_1,frame_2,label_2)

        return proba

    def makeFullGraph(self,
                      sp_desc,
                      sp_pom,
                      loc,
                      points_2d,
                      normNeighbor_in,
                      thresh_aux,
                      tau_u = 0,
                      direction='forward',
                      labels=None):

        #Constructs graph from pre-computed costs
        self.n_frames = np.max(sp_desc['frame'].values)+1
        self.logger.info("Making " + direction + " graph")

        self.direction = direction

        #Auxiliary edges (appearance) creates tracklets, input/output nodes and weight
        self.logger.info('Making/connecting tracklets')
        if(self.tracklets is None):
            self.make_init_tracklets(sp_pom,labels,loc,thresh_aux,direction)
        else:
            self.make_tracklets(sp_pom,thresh_aux,direction)

        tls = [t for t in self.tracklets if(t.blocked == False)]

        self.make_edges_from_tracklets(tls,
                                       sp_desc,
                                       loc,
                                       points_2d,
                                       normNeighbor_in,
                                       tau_u,
                                       labels)

        #self.orig_weights = nx.get_edge_attributes(self.g, 'weight')


    def make_trans_transform(self, sp_desc, pm, thresh, n_samps, n_dims, k, pca=False):

        descs_cat = utls.concat_arr(sp_desc['desc'])

        if(not pca):
            y = (pm['proba'] > thresh).as_matrix().astype(int)

            self.trans_transform = LFDA(dim=n_dims, k=k)

            rand_idx_pos = np.random.choice(np.where(y > 0)[0],size=n_samps)
            rand_idx_neg = np.random.choice(np.where(y == 0)[0],size=n_samps)
            rand_descs_pos = descs_cat[rand_idx_pos,:]
            rand_descs_neg = descs_cat[rand_idx_neg,:]
            rand_y_pos = y[rand_idx_pos]
            rand_y_neg = y[rand_idx_neg]
            rand_descs = np.concatenate((rand_descs_pos,rand_descs_neg),axis=0)
            rand_y = np.concatenate((rand_y_pos,rand_y_neg),axis=0)

            self.logger.info('Fitting LFDA (dims: {},k: {},n_samps: {})'.format( n_dims,k,n_samps))

            self.trans_transform.fit(rand_descs, rand_y)
        else:
            self.logger.info('Fitting PCA with {} components'.format(n_dims))
            self.trans_transform = PCA(n_components=n_dims, whiten=True)
            self.trans_transform.fit(descs_cat)


    def make_edges_from_tracklets(self,
                                  tls,
                                  sp_desc,
                                  loc,
                                  points_2d,
                                  normNeighbor_in,
                                  tau_u,
                                  labels):

        # get max id of tracklets (will increment)
        max_e_id = int(np.max([t.id_ for t in tls]))

        #Exit edges (time lapse)
        self.logger.info('Connecting exit edges')
        for i in range(len(tls)):
            max_e_id += 1
            this_e = (int(tls[i].out_id), int(self.sink))
            #self.g.add_edge(*this_e, 0., max_e_id)
            self.g.add_edge(*this_e, 0.)

        #Entrance edges (location)
        self.logger.info('Connecting entrance edges')
        for i in range(len(tls)):
            frame = loc['frame'][tls[i].df_ix[0]]
            loc_2d = points_2d[frame,3:5]
            i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                              loc_2d[1],
                                              labels[...,0].shape[1],
                                              labels[...,0].shape[0])

            sp_gaze = labels[i_gaze,j_gaze,frame]

            #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
            if(self.is_linkable_entrance_radius(loc,
                                                loc_2d,
                                                tls[i],
                                                labels,
                                                normNeighbor_in)):

                this_e = (int(self.source), int(tls[i].in_id))

                proba = self.proba_trans(sp_desc,
                                       tls[i].get_in_frame(),
                                       tls[i].get_in_label(),
                                       frame,
                                       sp_gaze)
                w = -np.log(proba / (1 - proba))
                max_e_id += 1
                #self.g.add_edge(*this_e, w, max_e_id)
                self.g.add_edge(*this_e, w)

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

                linkable_tracklets = self.tls_man.get_linkables(this_tracklet,
                                                                tau_u=tau_u,
                                                                mode=mode,
                                                                direction=self.direction)

                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(this_tracklet,
                                                        linkable_tracklets[j],
                                                        sp_desc,
                                                            self.direction,
                                                              mode='head')
                    w = -np.log(probas / (1 - probas))
                    this_e = (int(this_tracklet.out_id),
                              int(linkable_tracklets[j].in_id))
                    max_e_id += 1
                    #self.g.add_edge(*this_e, w, max_e_id)
                    self.g.add_edge(*this_e, w)

    def lp_min_cost_flow_pow(init_flow=1):

         return minimize(self.lp_make_graph_and_run, init_flow, args=(self.g,self.source,self.sink), method='Powell', tol=None, callback=None, options={'disp': True, 'return_all': False, 'maxiter': None, 'direc': None, 'func': None, 'maxfev': None, 'xtol': 0.0001, 'ftol': 0.0001})

    def lp_make_graph_and_run(fin,g,source,sink):

        e = g.edges()
        n = g.nodes()

        e_aux = [e for e in g.edges() if((e[0] != source) and (e[1] != sink))]

        # equality constraint matrix
        A_eq = nx.incidence_matrix(self.g,n,e,oriented=True)
        A_eq_m = cvxopt.matrix(A_eq.toarray()) #flow conservation/demand matrix

        b_ub = [self.g[e[i][0]][e[i][1]]['scale'] for i in range(len(e))]
        b_lb = np.zeros((len(e)))
        b_ineq = cvxopt.matrix(np.concatenate((b_ub,b_lb)))

        A_ub_m = cvxopt.spmatrix(1.0, range(len(e)), range(len(e)))
        A_lb_m = cvxopt.spmatrix(-1.0, range(len(e)), range(len(e)))
        A_ineq = cvxopt.sparse([A_ub_m,A_lb_m])
        b_ub_m = cvxopt.matrix(b_ub,tc='d')

        #cost vector
        costs_dict = nx.get_edge_attributes(self.g,'weight')
        costs = np.asarray([costs_dict[e] for e in self.g.edges()])
        c_m = cvxopt.matrix(costs.astype(float))

        demand_dict = {n:0 for n in self.g.nodes()}
        demand_dict['s'] = -fin[-1]
        demand_dict['t'] = fin[-1]

        nx.set_node_attributes(self.g,'demand',demand_dict)
        demands = nx.get_node_attributes(self.g,'demand')

        #capas = nx.get_edge_attributes(self.g,'scale')

        b_eq = [demands[n] for n in self.g.nodes()]
        b_eq_m = cvxopt.matrix(b_eq, tc='d')


        cvxopt.solvers.options['LPX_K_MSGLEV'] = 0
        cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF' #Disable stdout
        cvxopt.solvers.options['msg_levels'] = 'GLP_MSG_OFF'
        lp_sol = cvxopt.solvers.lp(c_m,
                                        A_ineq,
                                        b_ineq,
                                        A_eq_m,
                                        b_eq_m,
                                        solver='glpk')

        lp_cost = np.asarray(c_m.T*lp_sols[-1]['x'])[0]

        return lp_cost

    def lp_min_cost_flow(self,init_flow=1,gamma_0=0.9,max_iter=8):

        fin = [init_flow]

        gamma = gamma_0

        lp_sols = []
        lp_costs = []
        i = 0

        e = self.g.edges()
        n = self.g.nodes()

        e_aux = [e for e in self.g.edges() if((e[0] != self.source) and (e[1] != self.sink))]

        # equality constraint matrix
        A_eq = nx.incidence_matrix(self.g,n,e,oriented=True)
        A_eq_m = cvxopt.matrix(A_eq.toarray()) #flow conservation/demand matrix

        b_ub = [self.g[e[i][0]][e[i][1]]['scale'] for i in range(len(e))]
        b_lb = np.zeros((len(e)))
        b_ineq = cvxopt.matrix(np.concatenate((b_ub,b_lb)))

        A_ub_m = cvxopt.spmatrix(1.0, range(len(e)), range(len(e)))
        A_lb_m = cvxopt.spmatrix(-1.0, range(len(e)), range(len(e)))
        A_ineq = cvxopt.sparse([A_ub_m,A_lb_m])
        b_ub_m = cvxopt.matrix(b_ub,tc='d')

        #cost vector
        costs_dict = nx.get_edge_attributes(self.g,'weight')
        costs = np.asarray([costs_dict[e] for e in self.g.edges()])
        c_m = cvxopt.matrix(costs.astype(float))

        while(True):

            self.logger.info('-----')
            self.logger.info('Iter. ' + str(i) + '. Flow: ' + str(fin[-1]))
            self.logger.info('-----')
            demand_dict = {n:0 for n in self.g.nodes()}
            demand_dict['s'] = -fin[-1]
            demand_dict['t'] = fin[-1]

            nx.set_node_attributes(self.g,'demand',demand_dict)
            demands = nx.get_node_attributes(self.g,'demand')

            #capas = nx.get_edge_attributes(self.g,'scale')

            b_eq = [demands[n] for n in self.g.nodes()]
            b_eq_m = cvxopt.matrix(b_eq, tc='d')


            cvxopt.solvers.options['LPX_K_MSGLEV'] = 0
            cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF' #Disable stdout
            cvxopt.solvers.options['msg_levels'] = 'GLP_MSG_OFF'
            lp_sols.append(cvxopt.solvers.lp(c_m,
                                         A_ineq,
                                         b_ineq,
                                         A_eq_m,
                                         b_eq_m,
                                         solver='glpk'))

            lp_costs.append(c_m.T*lp_sols[-1]['x'])

            self.logger.info('-----')
            self.logger.info('Iter. ' + str(i) + '. Flow: ' + str(fin[-1]))
            self.logger.info('gamma: ' + str(gamma))
            self.logger.info('Obj: ' + str(lp_costs[-1][0]))
            self.logger.info('-----')

            if(i>3):
                if(lp_costs[-1][0] == lp_costs[-2][0]):
                    self.logger.info('Reached minimum after iter.: ' + str(i))
                    break

            if(i>max_iter):
                self.logger.info('Reached max num. of iter.: ' + str(max_iter))
                break

            i += 1
            #fin += 1
            if(i<3):
                fin.append(fin[-1]+1)
            else:
                diff_dL = np.asarray((lp_costs[-1]-lp_costs[-2]) - (lp_costs[-2]-lp_costs[-3]))[0][0]
                #gamma = (fin[-1] - fin[-2])*(diff_dL)/np.linalg.norm(diff_dL)**2
                gamma = gamma_0
                fin.append(int(fin[-1] - gamma*np.asarray(lp_costs[-1]-lp_costs[-2])[0][0]))

        #lp_sols = lp_sols[:-1]
        #lp_costs = lp_costs[:-1]

        self.lp_sols = []
        for i in range(len(lp_sols)):
            this_sols = dict()
            this_sols['x'] = np.asarray(lp_sols[i]['x'])
            this_sols['x'] = np.asarray(lp_sols[i]['x'])
            self.lp_sols.append(this_sols)

        self.lp_costs = lp_costs
