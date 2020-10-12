import tqdm
import networkx as nx
import numpy as np
from ksptrack import tr
from ksptrack import tr_manager as trm
import logging
import pickle as pk
from boostksp import libksp


class GraphTracking:
    """
    Connects, merges and add edges to graph. Also has KSP algorithm.
    """
    def __init__(self,
                 link_agent,
                 sps_man=None,
                 tol=10e-12,
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
        self.direction = None
        self.tol = tol

        self.thr = 0.05

        self.PCAs = []
        self.trans_transform = None

        self.tls_man = None
        self.sps_man = sps_man

        # This holds the networkx graph
        self.g = None

        self.link_agent = link_agent

    def make_tracklets(self, sp_pm):
        #Loop through existing tracklets and add edges.

        tls = [t for t in self.tracklets if (t.blocked == False)]
        bar = tqdm.tqdm(total=len(tls))
        for i, tl in enumerate(tls):
            this_tracklet_probas = []
            for df_ix in tl.df_ix:
                this_tracklet_probas.append(sp_pm['proba'][df_ix])
            this_tracklet_probas = np.asarray(this_tracklet_probas)
            this_tracklet_probas = np.clip(this_tracklet_probas,
                                           a_min=self.thr,
                                           a_max=1 - self.thr)
            w = np.sum(-np.log(
                np.asarray(this_tracklet_probas) /
                (1 - np.asarray(this_tracklet_probas))))
            this_e = (tl.in_id, tl.out_id)
            self.g.add_edge(*this_e, weight=w, id_=tls[i].id_)
            bar.update(1)
        bar.close()

    def make_init_tracklets(self, sp_pm, thresh, direction):
        #thresh: Block edges below this proba
        #This also populates the tracklet list

        n_unblocked = 0
        self.tracklets = []
        tracklet_id = 0
        node_id = np.max((self.source, self.sink)) + 1
        bar = tqdm.tqdm(total=sp_pm.shape[0])
        for i in range(sp_pm.shape[0]):
            bar.update(1)

            proba = sp_pm['proba'][i]

            proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)

            if (proba >= thresh):
                blocked = False
                n_unblocked += 1
            else:
                blocked = True

            w = -np.log(proba / (1 - proba))
            tl = tr.Tracklet(tracklet_id,
                             node_id,
                             node_id + 1,
                             [[(sp_pm['frame'][i], sp_pm['label'][i])]], [i],
                             direction,
                             length=1,
                             blocked=blocked,
                             marked=False)

            this_e = (tl.in_id, tl.out_id)

            self.tracklets.append(tl)
            if (not blocked):
                self.g.add_edge(this_e[0],
                                this_e[1],
                                weight=w,
                                id_=tl.id_,
                                sps=tl.sps)
            tracklet_id += 1
            node_id += 2

        bar.close()
        self.logger.info('Got {} unblocked tracklets. '.format(n_unblocked))

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
                tr.Tracklet(new_id_tls,
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

    def makeFullGraph(self,
                      sp_desc,
                      sp_pom,
                      thresh_aux,
                      hoof_tau_u=0,
                      rel_radius=1.,
                      direction='forward',
                      labels=None):

        #Constructs graph from pre-computed costs
        self.n_frames = np.max(sp_desc['frame'].values) + 1
        self.logger.info("Making " + direction + " graph")

        self.g = nx.DiGraph()
        self.direction = direction

        #Auxiliary edges (appearance) creates tracklets, input/output nodes and weight
        self.logger.info(
            'Making/connecting tracklets (thr: {})'.format(thresh_aux))
        if (self.tracklets is None):
            self.make_init_tracklets(sp_pom, thresh_aux, direction)
        else:
            self.make_tracklets(sp_pom)

        tls = [t for t in self.tracklets if (t.blocked == False)]

        self.make_edges_from_tracklets(tls, sp_desc, hoof_tau_u, rel_radius,
                                       labels)

        self.orig_weights = nx.get_edge_attributes(self.g, 'weight')

    def make_edges_from_tracklets(self, tls, sp_desc, hoof_tau_u, rel_radius,
                                  labels):

        #Exit edges (time lapse)
        self.logger.info('Connecting exit edges')
        for i in range(len(tls)):
            this_e = (tls[i].out_id, self.sink)
            self.g.add_edge(*this_e, weight=0, id_=-1)

        #Entrance edges (location)
        self.logger.info('Connecting entrance edges')

        added = 0
        for tl in tls:
            tl_loc = sp_desc.loc[tl.df_ix[0]]

            if (self.link_agent.is_entrance(tl_loc['frame'], tl_loc['label'])):
                # print('frame/label: {}/{} is entrance'.format(tl_loc['frame'],
                #                                               tl_loc['label']))

                this_e = (int(self.source), int(tl.in_id))

                proba = self.link_agent.get_proba_entrance(tl_loc)

                proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)
                w = -np.log(proba / (1 - proba))
                self.g.add_edge(*this_e, weight=w, id_=-1)
                added += 1
        self.logger.info('Added {} entrance edges'.format(added))

        # Transition edges
        self.logger.info(
            'Connecting transition edges. hoof_tau: {}'.format(hoof_tau_u))

        bar = tqdm.tqdm(total=len(tls))
        added = 0
        for tl in tls:

            linkable_tracklets = self.tls_man.get_linkables(
                tl,
                rel_radius=rel_radius,
                hoof_tau_u=hoof_tau_u,
                direction=self.direction)

            for linkable_tl in linkable_tracklets:

                proba = self.link_agent.get_proba_inter_frame(tl, linkable_tl)
                proba = np.clip(proba, a_min=self.thr, a_max=1 - self.thr)
                w = -np.log(proba / (1 - proba))
                this_e = (tl.out_id, linkable_tl.in_id)
                self.g.add_edge(*this_e, weight=w, id_=-1)
                added += 1
            bar.update(1)
        bar.close()
        self.logger.info('Added {} transition edges'.format(added))

    def run(self):
        from_source = [e for e in self.g.edges() if (e[0] == self.source)]
        if (len(from_source) == 0):
            self.logger.info('Found 0 entrance edges. Skipping.')
        else:
            self.copy_cxx()
            self.kspSet = self.g_cxx.run()
        return self.kspSet

    def run_nocopy(self):
        res = self.g_cxx.run()
        return res

    def copy_cxx(self):
        # This holds the c++ graph
        self.g_cxx = libksp.ksp()
        self.g_cxx.config(self.source,
                          self.sink,
                          loglevel=self.cxx_loglevel,
                          min_cost=True,
                          tol=self.tol,
                          return_edges=self.cxx_return_edges)

        self.logger.info('Copying graph with {} edges...'.format(
            len(self.g.edges())))
        # weights = [e['weight'] for e in self.g.edges]
        # self.logger.info('num. nan: {}'.format(np.isnan(weights).sum()))
        # self.logger.info('num. inf: {}'.format(np.isinf(weights).sum()))
        # self.logger.info('max weight: {}'.format(np.max(weights)))
        # self.logger.info('min weight: {}'.format(np.min(weights)))

        for e in self.g.edges():
            self.g_cxx.add_edge(int(e[0]), int(e[1]),
                                self.g[e[0]][e[1]]['weight'],
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
