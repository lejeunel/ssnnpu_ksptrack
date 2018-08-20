import networkx as nx
import graphksp as gksp

def makeTestGraph2():

    g = nx.DiGraph()
    g.add_edge('a', 'g', weight = 4,label=0)
    g.add_edge('g', 'z', weight = 2,label=0)
    g.add_edge('a', 'f', weight = 5,label=0)
    g.add_edge('f', 'z', weight = 2,label=0)

    g.add_edge('a', 'd', weight = 1,label=0)
    g.add_edge('d', 'e', weight = 2,label=0)
    g.add_edge('e', 'z', weight = 2,label=0)

    g.add_edge('a', 'b', weight = 1,label=0)
    g.add_edge('b', 'c', weight = 1,label=0)
    g.add_edge('c', 'z', weight = 1,label=0)

    source = 'a'
    sink = 'z'

    return g,source,sink

def makeTestGraphTuples():

    g = nx.DiGraph()
    g.add_edge((0,1), (0,2), weight = 1)
    g.add_edge((0,2), (0,3), weight = 0.5)
    g.add_edge((0,3), (0,4), weight = 1)
    g.add_edge((0,4), (0,8), weight = 1)
    g.add_edge((0,1), (0,7), weight = 7)
    g.add_edge((0,3), (0,7), weight = 1)
    g.add_edge((0,7), (0,8), weight = 2)
    g.add_edge((0,1), (0,5), weight = 1)
    g.add_edge((0,5), (0,2), weight = 1)
    g.add_edge((0,5), (0,6), weight = 3)
    g.add_edge((0,2), (0,6), weight = 1)
    g.add_edge((0,6), (0,4), weight = 1)
    g.add_edge((0,6), (0,8), weight = 4)
    nx.set_edge_attributes(g,'occupied',0)
    nx.set_edge_attributes(g,'label',0)


    for e in g.edges():
        g[e[0]][e[1]]['capacity'] = 1

    source = (0,1)
    sink = (0,8)

    return g,source,sink

def makeTestGraph():

    g = nx.DiGraph()
    g.add_edge('a', 'b', weight = 1)
    g.add_edge('b', 'c', weight = 1)
    g.add_edge('c', 'd', weight = 1)
    g.add_edge('d', 'z', weight = 1)
    g.add_edge('a', 'g', weight = 7)
    g.add_edge('c', 'g', weight = 1)
    g.add_edge('g', 'z', weight = 2)
    g.add_edge('a', 'e', weight = 1)
    g.add_edge('e', 'b', weight = 1)
    g.add_edge('e', 'f', weight = 3)
    g.add_edge('b', 'f', weight = 1)
    g.add_edge('f', 'd', weight = 1)
    g.add_edge('f', 'z', weight = 4)
    nx.set_edge_attributes(g,'occupied',0)
    nx.set_edge_attributes(g,'label',0)

    source = 'a'
    sink = 'z'

    return g,source,sink

g,source,sink = makeTestGraphTuples()
#g,source,sink = makeTestGraph()
g_obj = gksp.GraphKSP(g,source,sink,'edge')
g_obj.disjointKSP(1)
print("Edge-disjoint (K=1): ", g_obj.kspSet[0])

g.node[(0,1)]['demand'] = -1
g.node[(0,8)]['demand'] = 1
flowCost, flowDict = nx.network_simplex(g)
print("Min-cost-flow (K=1): ",sorted([(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0]))

print("----")
print("Edge-disjoint (K=2)")
g_obj.disjointKSP(2)
print("1st-shortest path:", g_obj.kspSet[1][0])
print("2nd-shortest path:", g_obj.kspSet[1][1])

g,source,sink = makeTestGraphTuples()
g.node[(0,1)]['demand'] = -2
g.node[(0,8)]['demand'] = 2
flowCost, flowDict = nx.network_simplex(g)
print("Min-cost-flow (K=2): ",sorted([(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0]))

print("----")
print("Edge-disjoint (K=3)")
g_obj.disjointKSP(3)
print("1st-shortest path:", g_obj.kspSet[2][0])
print("2nd-shortest path:", g_obj.kspSet[2][1])
print("3rd-shortest path:", g_obj.kspSet[2][2])


g_obj_nd = gksp.GraphKSP(g,source,sink,'node')
print("----")
print("Node-disjoint (K=1)")
g_obj_nd.disjointKSP(1)
print("1st-shortest path:", g_obj_nd.kspSet[0])

print("----")
print("Node-disjoint (K=2)")
g_obj_nd.disjointKSP(2)
print("1st-shortest path:", g_obj_nd.kspSet[1][0])
print("2nd-shortest path:", g_obj_nd.kspSet[1][1])

print("----")
print("Node-disjoint (K=3)")
g_obj_nd.disjointKSP(3)
print("1st-shortest path:", g_obj_nd.kspSet[2][0])
print("2nd-shortest path:", g_obj_nd.kspSet[2][1])
print("3rd-shortest path:", g_obj_nd.kspSet[2][2])
