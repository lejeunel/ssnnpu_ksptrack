import networkx as nx

path = '/home/ubelix/runs/selective_search/Dataset00/frame_0000.p'

g = nx.read_gpickle(path)

for n in g.nodes:
    print(n)
