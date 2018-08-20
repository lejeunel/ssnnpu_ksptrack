import networkx as nx
g = nx.DiGraph()

class Tracklet:

    def __init__(self,sps=None):
        self.sps = sps

    def get_in(self):
        return self.sps[0]

    def get_out(self):
        return self.sps[-1]

def concat_tracklets(track0,track1):
    #Returns concatenated tracklet [track0,track1]
    sps = track0.sps + track1.sps
    return Tracklet(sps)

track0 = Tracklet([(0,0),(1,0),(2,0)])
track1 = Tracklet([(3,0),(4,0),(5,0)])

track_cat = concat_tracklets(track0,track1)

g.add_edge(track0,track1)
