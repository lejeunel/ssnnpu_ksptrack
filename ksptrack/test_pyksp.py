#!/usr/bin/env python3
import networkx as nx
import scipy.sparse as sparse
import numpy as np
import pyksp

g = nx.read_gpickle('pu/tmp.p')
source = [n for n, a in g.nodes(data=True) if 'is_source' in a.keys()][0]
sink = [n for n, a in g.nodes(data=True) if 'is_sink' in a.keys()][0]
edges = np.array([e for e in g.edges(data='weight')])
tracker = pyksp.PyKsp(edges[:, 0], edges[:, 1], edges[:, 2],
                      nx.number_of_nodes(g), source, sink)

tracker.config(min_cost=True)
paths = tracker.run()
sps = []
for p in paths:
    p = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
    for e in p:
        if ('sps' in g[e[0]][e[1]].keys()):
            sps.append(g[e[0]][e[1]]['sps'][0][0])

# print(sps)
print('num superpixels: ', len(sps))
