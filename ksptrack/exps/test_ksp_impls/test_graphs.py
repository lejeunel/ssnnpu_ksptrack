import os
import yaml
import pandas as pd
import numpy as np
from ksptrack.cfgs import cfg
import ksptrack.graph_tracking as gtrack
import ksptrack.graph_tracking_cxx as gtrack_cxx
from ksptrack.utils import my_utils as utls
from ksptrack.utils.data_manager import DataManager
import logging
import pickle
import networkx as nx
import time
import inspect

# Get path of this dir
path = os.path.split(inspect.stack()[0][1])[0]

g_for = gtrack.GraphTracking(None,
                             tol=0,
                             mode='edge')

g_for_cxx = gtrack_cxx.GraphTracking(None,
                                tol=0,
                                mode='edge')

g_for_cxx.g.config(g_for.source,
                   g_for.sink,
                   loglevel="info",
                   min_cost=True)

utls.setup_logging('.')
g_for.g = pickle.load(open(os.path.join(path, 'nxgraph.p'), 'rb'))
g_for.orig_weights = nx.get_edge_attributes(g_for.g, 'weight')

print('num. edges: {}'.format(nx.number_of_edges(g_for.g)))
print('num. nodes: {}'.format(nx.number_of_nodes(g_for.g)))

id_e = 0
for e in g_for.g.edges():
    g_for_cxx.g.add_edge(int(e[0]),
                         int(e[1]),
                         g_for.g[e[0]][e[1]]['weight'],
                         id_e)
    id_e += 1

start_cxx = time.time()
res_cxx = g_for_cxx.g.run()
end_cxx = time.time()

start_py = time.time()
g_for.disjointKSP(K=None, verbose=True)
end_py = time.time()
res_py = g_for.kspSet[-1]

time_cxx = end_cxx - start_cxx
time_py = end_py - start_py
print('cxx time: {}'.format(time_cxx))
print('py time: {}'.format(time_py))
print('speed-up: {:.2f}%'.format(time_py/time_cxx*100))

edges_py = [item for sublist in res_py for item in sublist]
edges_cxx = [item for sublist in res_cxx for item in sublist]

print('num edges with cxx: {}'.format(len(edges_cxx)))
print('num edges with py: {}'.format(len(edges_py)))
