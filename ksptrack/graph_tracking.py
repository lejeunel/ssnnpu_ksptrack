import tqdm
import networkx as nx
import numpy as np
from ksptrack import tr
from ksptrack import tr_manager as trm
import logging
import pickle as pk
import pyksp


def paths_to_sps(g, paths):
    sps = []
    for p in paths:
        p = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
        for e in p:
            if ('sps' in g[e[0]][e[1]].keys()):
                sps.append(g[e[0]][e[1]]['sps'][0][0])

    return sps


class GraphTracking:
    """
    Connects, merges and add edges to graph. Also has KSP algorithm.
    """
    def __init__(self,
                 link_agent,
                 sps_man=None,
                 tol=10e-7,
                 cxx_loglevel="info",
                 cxx_return_edges=True):
        self.logger = logging.getLogger('GraphTracking')

        self.nodeTypes = ['virtual', 'input', 'output', 'none']

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
        node_id = 0
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
                node_id += 2
            tracklet_id += 1

        bar.close()
        print('Got {} unblocked tracklets. '.format(n_unblocked))

        print('Building tracklet manager')
        self.tls_man = trm.TrackletManager(self.sps_man, self.direction,
                                           self.tracklets, self.n_frames)

    def make_graph(self,
                   sp_desc,
                   sp_pom,
                   thresh_aux,
                   rel_radius,
                   direction='forward',
                   labels=None):

        #Constructs graph from pre-computed costs
        self.n_frames = np.max(sp_desc['frame'].values) + 1
        print("Making " + direction + " graph")

        self.g = nx.DiGraph()
        self.direction = direction

        #Auxiliary edges creates tracklets, input/output nodes and weight
        print('Making/connecting tracklets (thr: {})'.format(thresh_aux))
        self.make_init_tracklets(sp_pom, thresh_aux, direction)

        tls = [t for t in self.tracklets if (t.blocked == False)]

        self.make_edges_from_tracklets(tls, sp_desc, rel_radius, labels)

    def make_edges_from_tracklets(self, tls, sp_desc, rel_radius, labels):

        assert len(
            tls
        ) > 0, 'graph must contain tracklets. Call make_init_tracklets first!'

        # set source and sink nodes as max(n_id) and max(n_id)+1 respectively
        nodes = list(self.g.nodes)
        source_id = np.max(nodes) + 1
        sink_id = np.max(nodes) + 2
        self.g.add_node(source_id, is_source=True)
        self.g.add_node(sink_id, is_sink=True)

        #Entrance edges (location)
        print('Connecting entrance edges')
        added = 0
        for tl in tls:
            tl_loc = sp_desc.loc[tl.df_ix[0]]

            if (self.link_agent.is_entrance(tl_loc['frame'], tl_loc['label'])):

                this_e = (source_id, int(tl.in_id))

                self.g.add_edge(*this_e, weight=0, id_=-1)
                added += 1
        print('Added {} entrance edges'.format(added))

        #Exit edges (time lapse)
        print('Connecting exit edges')
        for i in range(len(tls)):
            this_e = (tls[i].out_id, sink_id)
            self.g.add_edge(*this_e, weight=0., id_=-1)

        # Transition edges
        print('Connecting transition edges.')
        bar = tqdm.tqdm(total=len(tls))
        added = 0
        for tl in tls:

            linkable_tracklets = self.tls_man.get_linkables(
                tl, rel_radius=rel_radius, direction=self.direction)

            for linkable_tl in linkable_tracklets:
                this_e = (tl.out_id, linkable_tl.in_id)
                self.g.add_edge(*this_e, weight=0., id_=-1)
                added += 1
            bar.update(1)
        bar.close()
        print('Added {} transition edges'.format(added))

    def run(self, return_sps=True):
        source = [
            n for n, a in self.g.nodes(data=True) if 'is_source' in a.keys()
        ][0]
        sink = [
            n for n, a in self.g.nodes(data=True) if 'is_sink' in a.keys()
        ][0]

        from_source = [e for e in self.g.edges() if (e[0] == source)]
        if (len(from_source) == 0):
            print('Found 0 entrance edges. Skipping.')
            return

        edges = np.array([e for e in self.g.edges(data='weight')])
        tracker = pyksp.PyKsp(edges[:, 0], edges[:, 1], edges[:, 2],
                              nx.number_of_nodes(self.g), source, sink)

        tracker.config(min_cost=True)
        paths = tracker.run()

        self.kspSet = paths

        if return_sps:
            return paths_to_sps(self.g, self.kspSet)

        return self.kspSet

    def save_graph(self, path):
        nx.write_gpickle(self.g, path)

    def load_graph(self, path):
        self.g = nx.read_gpickle(path)

    def save_all(self, path):
        graph_save_path = path + '_graph.p'
        sps_man_save_path = path + '_sp_man.p'
        tls_man_save_path = path + '_tls_man.p'
        print('Saving graph to {}'.format(graph_save_path))
        self.save_graph(graph_save_path)
        print('Saving sp manager to {}'.format(sps_man_save_path))
        pk.dump(self.sps_man, open(sps_man_save_path, 'wb'))
        print('Saving tls manager to {}'.format(tls_man_save_path))
        pk.dump(self.tls_man, open(tls_man_save_path, 'wb'))

    def load_all(self, path):
        self.load_graph(path + '_graph.p')
        self.sps_man = pk.load(open(path + '_sp_man.p', 'rb'))
        self.tls_man = pk.load(open(path + '_tls_man.p', 'rb'))
