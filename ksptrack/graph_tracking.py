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
from metric_learn import LFDA
from sklearn.decomposition import PCA
from boostksp import libksp

class GraphTracking:
    """
    Connects, merges and add edges to graph. Also has KSP algorithm.
    """

    def __init__(self,sps_man, tol=0, mode='edge', tau_u = -1):
        self.logger = logging.getLogger('GraphTracking')

        self.nodeTypes = ['virtual', 'input', 'output', 'none']
        self.source = 0
        self.sink = 1
        self.kspSet = []
        self.tracklets = None
        self.costs = []
        self.sp = []  #Used for cost transformation
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
        self.sps_man = sps_man

        # This holds the networkx graph
        self.g = None


    def make_tracklets(self, sp_pm, thresh,direction):
        #Loop through existing (already merged) tracklets and assign mean proba.
        #self.tracklets contains both merge and not-merge (length 1) tracklets.
        #sp_pm is "re-looped" to add previously discarded tracklets that now correspond to proba > self.thr

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
                    this_e = ((tls[i].id_, 0),
                            (tls[i].id_, 1))
                    self.g.add_edge(*this_e, weight=w, scale=tls[i].scale)

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
                    this_e = ((curr_id, 0),
                            (curr_id, 1))
                    this_tracklet = tr.Tracklet(curr_id,
                                            [[(sp_pm['frame'][remaining_df_ix[i]],
                                            sp_pm['sp_label'][remaining_df_ix[i]])]],
                                            [remaining_df_ix[i]],
                                            direction,
                                            scale=1,
                                            length=1,
                                            blocked=False,
                                            marked=False)
                    self.tracklets.append(this_tracklet)

                    self.g.add_edge(this_e[0], this_e[1], weight=w)
                    curr_id +=1

    def make_init_tracklets(self, sp_pm, labels, centroids_loc, thresh,direction):
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
                              length=1,
                              blocked=blocked,
                              marked=False)

                this_e = (tl.in_id, tl.out_id)

                self.tracklets.append(tl)
                if(not blocked):
                    self.g.add_edge(this_e[0], this_e[1],
                                    weight=w,
                                    id_=tl.id_)
                tracklet_id +=1
                node_id += 2

        self.logger.info('Building tracklet manager')
        self.tls_man = trm.TrackletManager(self.sps_man,
                                           self.direction,
                                       self.tracklets,
                                       self.n_frames)


    def unblock_tracklets(self,sp_pm,thresh):

        tls_blocked = [t for t in self.tracklets if(t.blocked == True)]
        tls_unblocked = []
        for t in tls_blocked:
            df_ix = t.df_ix
            proba = np.asarray([sp_pm['proba'][ix] for ix in df_ix])
            proba = np.clip(proba,                                           a_min=self.thr,a_max=1-self.thr)

            if(np.mean(proba) > thresh):
                t.blocked = False
                tls_unblocked.append(t)
        self.logger.info('Unblocked tracklets: ' + str(len(tls_unblocked)))

        return tls_unblocked

    def merge_tracklets_temporally(self,
                                   tracklet_set,
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
        #this_set = self.kspSet[-1]
        new_tracklets = []
        ids_to_delete = [] #Store IDs of tracklets to delete after merging
        for tl_set in tracklet_set:
            #Remove source and sink edges
            #Get corresponding (frame,sp_label) tuples
            this_new_sps = [t.sps for t in tl_set]
            this_new_sps = [item for sublist in this_new_sps for item in sublist]
            #Get corresponding df_ix indices
            this_new_df_ix = [t.df_ix for t in tl_set]
            this_new_df_ix = [item for sublist in this_new_df_ix for item in sublist]
            this_new_sps_arr = np.asarray(this_new_sps).reshape(len(this_new_sps),2)

            proba = np.asarray([sp_pm['proba'][ix] for ix in this_new_df_ix])
            proba = np.clip(proba,                                           a_min=self.thr,a_max=1-self.thr)

            # Create new tracklet and corresponding edge to graph
            if(np.mean(proba) > thresh):
                blocked = False
            else:
                blocked = True

            new_tracklets.append(tr.Tracklet(new_id,
                                                tl_set[0].in_id,
                                                tl_set[-1].out_id,
                                                this_new_sps,
                                                this_new_df_ix,
                                                length=len(this_new_df_ix),
                                                blocked=blocked,
                                                direction=self.direction))

            if(blocked == False):
                w = np.sum([-np.log(p/(1-p)) for p in proba])
                # Need to add transition costs
                if(len(tl_set)>1):

                    w_trans = 0
                    for i in range(len(tl_set)-1):
                        t1 = tl_set[i]
                        t2 = tl_set[i+1]
                        p = self.trans_probas_tracklets(t1,
                                                        t2,
                                                        sp_desc,
                                                        self.direction,
                                                        mode='head')
                        w_trans += -np.log(p/(1-p))
                else:
                    w_trans = 0
                this_e = (new_tracklets[-1].in_id,
                          new_tracklets[-1].out_id)
                self.g.add_edge(this_e[0],
                                this_e[1],
                                weight=w+w_trans,
                                id_=new_id)
            new_id += 1

            ids_to_delete.append([t.id_ for t in tl_set])

        #Get ids of tracklets to delete
        ids_to_delete = [item for sublist in ids_to_delete for item in sublist]

        #Remove nodes of old tracklets (removes all edges connected to it by default)
        self.logger.info('Removing old tracklets')
        tls_to_delete = [t for t in self.tracklets if(t.id_ in ids_to_delete)]
        for t in tls_to_delete:
            self.g.remove_edge(t.in_id, t.out_id)

        #Remove old tracklets from list
        self.tracklets = [t for t in self.tracklets if(t.id_ not in ids_to_delete)]

        # Unblock tracklets with probas above thresh
        tls_unblocked = self.unblock_tracklets(sp_pm,thresh)

        #Make new list of tracklets and update tracklet manager
        self.tracklets += new_tracklets

        new_and_unblocked_tls = new_tracklets + tls_unblocked

        self.logger.info('Connecting entrance edges on new tracklets')
        for i in range(len(new_tracklets)):

            this_frame = loc['frame'][new_tracklets[i].df_ix[0]]
            # Check if we have a 2d location on this_frame
            if(this_frame in points_2d[:, 0]):
                loc_2d = points_2d[points_2d[:, 0] == this_frame, 3:5].ravel()
                i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                                loc_2d[1],
                                                labels[...,0].shape[1],
                                                labels[...,0].shape[0])

                sp_gaze = labels[i_gaze,j_gaze,this_frame]

                #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
                if(self.is_linkable_entrance_radius(loc,
                                                    loc_2d,
                                                    new_tracklets[i],
                                                    labels,
                                                    normNeighbor_in)):

                    this_e = (self.source, new_tracklets[i].in_id)

                    proba = self.proba_trans(sp_desc,
                                        new_tracklets[i].get_in_frame(),
                                        new_tracklets[i].get_in_label(),
                                        this_frame,
                                        sp_gaze)
                    w = -np.log(proba / (1 - proba))
                    self.g.add_edge(*this_e,
                                    weight=w, id_=-1)

        #Entrance edges (location)
        self.logger.info('Connecting entrance edges on unblocked tracklets')
        for i in range(len(tls_unblocked)):
            this_frame = loc['frame'][tls_unblocked[i].df_ix[0]]
            if(this_frame in points_2d[:, 0]):
                loc_2d = points_2d[points_2d[:, 0] == this_frame, 3:5].ravel()
                i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                                loc_2d[1],
                                                labels[...,0].shape[1],
                                                labels[...,0].shape[0])

                sp_gaze = labels[i_gaze,j_gaze,this_frame]

                #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
                if(self.is_linkable_entrance_radius(loc,
                                                    loc_2d,
                                                    tls_unblocked[i],
                                                    labels,
                                                    normNeighbor_in)):

                    this_e = (int(self.source),
                              int(tls_unblocked[i].in_id))

                    proba = self.proba_trans(sp_desc,
                                        tls_unblocked[i].get_in_frame(),
                                        tls_unblocked[i].get_in_label(),
                                        this_frame,
                                        sp_gaze)
                    w = -np.log(proba / (1 - proba))
                    self.g.add_edge(*this_e,
                                    weight=w, id_=-1)

        self.logger.info('Connecting exit edges on new and unblocked tracklets')
        for t in new_and_unblocked_tls:
            this_e = (t.out_id, self.sink)
            self.g.add_edge(*this_e, weight=0, id_=-1) 

        tls = [t for t in self.tracklets if(t.blocked == False)]
        self.logger.info('Updating tracklet dictionary')
        self.tls_man.make_dict(self.tracklets,self.n_frames)

        # Transition edges
        self.logger.info('Connecting transition edges on new and unblocked tracklets')
        import pdb; pdb.set_trace()
        with progressbar.ProgressBar(maxval=len(new_and_unblocked_tls)) as bar:
            for i in range(len(new_and_unblocked_tls)):

                bar.update(i)
                this_tracklet = new_and_unblocked_tls[i]
                linkable_tracklets = self.tls_man.get_linkables(this_tracklet,
                                                                tau_u=tau_u,
                                                                direction=self.direction,
                                                                mode='head')

                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(this_tracklet,
                                                            linkable_tracklets[j],
                                                            sp_desc,
                                                            self.direction,
                                                                mode='head')
                    w = -np.log(probas / (1 - probas))
                    this_e = (this_tracklet.out_id,
                                linkable_tracklets[j].in_id)
                    self.g.add_edge(*this_e, weight=w, id_=-1)

                linkable_tracklets = self.tls_man.get_linkables(this_tracklet,
                                                                tau_u=tau_u,
                                                                direction=self.direction,
                                                                mode='tail')
                for j in range(len(linkable_tracklets)):

                    probas = self.trans_probas_tracklets(this_tracklet,
                                                              linkable_tracklets[j],
                                                              sp_desc,
                                                              self.direction,
                                                              mode='tail')
                    w = -np.log(probas / (1 - probas))
                    this_e = (linkable_tracklets[j].out_id,
                                this_tracklet.in_id)
                    self.g.add_edge(*this_e, weight=w, id_=-1)
                self.logger.debug('new_tracklets ind: ' + str(i))

        #self.logger.debug('Neg cost cycle: ' + str(self.has_neg_cost_cycle()))
        self.orig_weights = nx.get_edge_attributes(self.g, 'weight')

    def is_linkable_entrance_radius(self,centroids,
                                    loc_2d,
                                    tracklet,
                                    labels,
                                    radius=0.2):
        #Enters if centroid is in gaze "radius"

        centroid_sp = centroids.loc[(centroids['frame'] == tracklet.get_in_frame()) & (centroids['sp_label'] == tracklet.get_in_label())]
        i_gaze,j_gaze = csv.coord2Pixel(loc_2d[0],loc_2d[1],labels[...,0].shape[1],labels[...,0].shape[0])
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
        i_gaze,j_gaze = csv.coord2Pixel(loc_2d[0],loc_2d[1],labels[...,0].shape[1],labels[...,0].shape[0])

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

        self.g = nx.DiGraph()
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

            # Check for features with all zeros
            inds_ = np.where(np.sum(rand_descs, axis=0) != 0)[0]
            rand_descs = rand_descs[:, inds_]

            rand_y = np.concatenate((rand_y_pos,rand_y_neg),axis=0)

            self.logger.info('Fitting LFDA (dims,k,n_samps): (' + str(n_dims) +
                            ',' + str(k) + ',' + str(n_samps) + ')')

            self.trans_transform.fit(rand_descs, rand_y)
        else:
            self.logger.info('Fitting PCA with {} components'.format(n_comps_pca))
            self.trans_transform = PCA(n_components=n_comps_pca, whiten=True)
            self.trans_transform.fit(descs_cat)


    def make_edges_from_tracklets(self,
                                  tls,
                                  sp_desc,
                                  loc,
                                  points_2d,
                                  normNeighbor_in,
                                  tau_u,
                                  labels):

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
            if(this_frame in points_2d[:, 0]):
                loc_2d = points_2d[points_2d[:, 0] == this_frame, 3:5].ravel()
                i_gaze, j_gaze = utls.norm_to_pix(loc_2d[0],
                                                loc_2d[1],
                                                labels[...,0].shape[1],
                                                labels[...,0].shape[0])

                sp_gaze = labels[i_gaze,j_gaze,this_frame]

                #if(self.is_linkable_entrance_sp(loc_2d,tls[i],labels)):
                if(self.is_linkable_entrance_radius(loc,
                                                    loc_2d,
                                                    tls[i],
                                                    labels,
                                                    normNeighbor_in)):

                    this_e = (int(self.source),
                              int(tls[i].in_id))

                    proba = self.proba_trans(sp_desc,
                                        tls[i].get_in_frame(),
                                        tls[i].get_in_label(),
                                        this_frame,
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
                    this_e = (this_tracklet.out_id,
                                linkable_tracklets[j].in_id)
                    self.g.add_edge(*this_e,
                                    weight=w,
                                    id_=-1)

    def run(self):

        # This holds the c++ graph
        self.g_cxx = libksp.ksp()
        self.g_cxx.config(self.source,
                      self.sink,
                      loglevel="info",
                      min_cost=True)

        self.logger.info('Copying graph...')
        for e in self.g.edges():
            self.g_cxx.add_edge(int(e[0]),
                                int(e[1]),
                                self.g[e[0]][e[1]]['weight'],
                                int(self.g[e[0]][e[1]]['id_']))
        self.kspSet =  self.g_cxx.run()
        return self.kspSet
