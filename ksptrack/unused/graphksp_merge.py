import progressbar
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class GraphKSP:
    def __init__(self, tol, mode='edge'):
        self.g = None
        self.nodeTypes = ['virtual', 'input', 'output', 'none']
        self.source = 's'
        self.sink = 't'
        #self.orig_sink = 's'
        #self.orig_source = None
        self.kspSet = []
        self.costs = []
        self.sp = []  #Used for cost transformation
        self.nb_sets = 0
        self.mode = mode
        self.direction = None
        self.thr = 0.05 #For bernoulli costs
        self.tol = tol

    def set_aux_costs(self, sp_pm, thresh, mode='new'):
        #mode: {new} edge doesn't exist, will be added
        #mode: {update} edge exists, weight will be updated
        #thresh: Remove edges below this proba

        skipped_e = []

        with progressbar.ProgressBar(maxval=sp_pm.shape[0]) as bar:
            for i in range(sp_pm.shape[0]):
                bar.update(i)
                this_e = (((sp_pm['frame'][i], sp_pm['sp_label'][i]), 0),
                          ((sp_pm['frame'][i], sp_pm['sp_label'][i]), 1))
                proba = sp_pm['proba'][i]
                if (proba > 1 - self.thr): proba = 1 - self.thr
                if (proba < self.thr): proba = self.thr

                w = -np.log(proba / (1 - proba))
                if (self.direction is not 'forward'):
                    if((mode == 'new') & (proba > thresh)):
                        self.g.add_edge(this_e[1], this_e[0], weight=w)
                    elif((mode != 'new') & (proba > thresh)):
                        if(not self.g.has_edge(this_e[0],this_e[1])):
                            self.g.add_edge(this_e[1], this_e[0], weight=w)
                        else:
                            self.g[this_e[1]][this_e[0]]['weight'] = w
                    else: #Skipping edge
                        skipped_e.append(this_e)
                else:
                    if((mode == 'new') & (proba > thresh)):
                        self.g.add_edge(this_e[0], this_e[1], weight=w)
                    elif((mode != 'new') & (proba > thresh)):
                        if(not self.g.has_edge(this_e[0],this_e[1])):
                            self.g.add_edge(this_e[0], this_e[1], weight=w)
                        else:
                            self.g[this_e[0]][this_e[1]]['weight'] = w
                    else: #Skipping edge
                        skipped_e.append(this_e)
        return skipped_e

    def makeFullGraphSPM(self,
                         sp_entr,
                         sp_pom,
                         sp_inters,
                         loc,
                         gaze_points,
                         normNeighbor,
                         normNeighbor_in,
                         thresh_aux,
                         tau_u,
                         direction='forward',
                         labels=None):
        #Constructs graph from pre-computed costs

        print("Making " + direction + " graph")

        self.g = nx.DiGraph()
        self.direction = direction

        if (labels is not None):
            sp_costs_arr = np.zeros(labels.shape)

        #Exit edges (time lapse)
        self.g.add_node(self.sink)
        with progressbar.ProgressBar(maxval=loc.shape[0]) as bar:
            for i in range(loc.shape[0]):
                if (direction is not 'forward'):
                    this_e = (((loc['frame'][i], loc['sp_label'][i]), 0), self.sink)
                else:
                    this_e = (((loc['frame'][i], loc['sp_label'][i]), 1), self.sink)
                self.g.add_edge(*this_e, weight=0)

        #Entrance edges (location)
        with progressbar.ProgressBar(maxval=loc.shape[0]) as bar:
            for i in range(loc.shape[0]):
                bar.update(i)
                loc1 = gaze_points[loc['frame'][i],3:5]
                loc2 = np.asarray([loc['pos_norm_x'][i], loc['pos_norm_y'][i]])
                #if( isInNeighborhood(loc1,loc2,normNeighbor_in)):
                #print((loc['frame'][i],loc['sp_label'][i],np.linalg.norm(loc1-loc2)))
                if (np.linalg.norm(loc1 - loc2) < normNeighbor_in):
                    if (direction is not 'forward'):
                        this_e = ('s', ((loc['frame'][i], loc['sp_label'][i]),
                                        1))
                    else:
                        this_e = ('s', ((loc['frame'][i], loc['sp_label'][i]),
                                        0))

                    #print(this_e)
                    inters = sp_entr['inters'][i]
                    inter = np.max(inters)
                    #inter = hist_inter(sp_desc['desc'][i],seen_feats[1][loc['frame'][i]])
                    if (inter > 1 - self.thr): inter = 1 - self.thr
                    if (inter < self.thr): inter = self.thr
                    #expo = inter
                    #amp = 1.
                    #res1 = np.log(inter)
                    #res2 = logsumexp(np.array([0, expo]),b=np.array([1,-amp]))
                    #w = -(res1-res2)
                    w = -np.log(inter / (1 - inter))
                    self.g.add_edge(*this_e, weight=w)

        #tau_u = .55
        #Transition edges
        sp_inters = sp_inters.loc[sp_inters['loc_dist'] < normNeighbor]
        sp_inters = sp_inters.reset_index()
        with progressbar.ProgressBar(maxval=sp_inters.shape[0]) as bar:
            for i in range(sp_inters.shape[0]):
                bar.update(i)
                if (direction is not 'forward'):
                    this_e = (((sp_inters['output frame'][i],
                                sp_inters['output label'][i]), 0),
                              ((sp_inters['input frame'][i],
                                sp_inters['input label'][i]), 1))
                    hoof_inter = sp_inters['hoof_inter_b'][i]
                else:
                    this_e = (((sp_inters['input frame'][i],
                                sp_inters['input label'][i]), 1),
                              ((sp_inters['output frame'][i],
                                sp_inters['output label'][i]), 0))
                    #inters = 1-link_mat[i,4]
                    hoof_inter = sp_inters['hoof_inter_f'][i]
                if ((sp_inters['loc_dist'][i] < normNeighbor) &
                    (hoof_inter > tau_u)):
                #if ((sp_inters['loc_dist'][i] < normNeighbor)):
                    inters = sp_inters['hist_inter'][i]
                    if (np.isnan(inters)): inters = 0.5
                    else:
                        if (inters > 1 - self.thr): inters = 1 - self.thr
                        if (inters < self.thr): inters = self.thr

                    #expo = dissim
                    #amp = 1.
                    #res1 = np.log(amp*np.exp(expo))
                    #res2 = logsumexp(np.array([0, expo]),b=np.array([1,-amp]))
                    #w = -(res1-res2)
                    w = -np.log(inters / (1 - inters))
                    #if(this_e[0][0][1] == this_e[1][0][1]):
                    #    print(-w,inters)
                    self.g.add_edge(this_e[0], this_e[1], weight=w)

        #Auxiliary edges (appearance)
        skipped_e = self.set_aux_costs(sp_pom,thresh_aux)
        skipped_nodes_in = [e[0] for e in skipped_e]
        skipped_nodes_out = [e[1] for e in skipped_e]
        skipped_nodes = skipped_nodes_in + skipped_nodes_out
        self.g.remove_nodes_from(skipped_nodes)
        print('Num. of edges: ' + str(len(self.g.edges())))

        self.g.add_node(self.source)
        self.orig_weights = nx.get_edge_attributes(self.g, 'weight')
        self.setAllLabelToNull()
        self.unOccupy()
        self.g_orig = self.g.copy()

    def costOfPaths(self, p):
        cost = list()
        for i in range(len(p)):
            cost.append([])
            this_cost = 0
            for j in range(len(p[i])):
                this_cost += self.orig_weights[(p[i][j][0], p[i][j][1])]
            cost[i].append(this_cost)

        return cost

    def reverseEdgeAndWeightOnPaths(self, p, nb_paths, inv_arg=False):

        if (nb_paths == 1):
            self.reverseEdgeAndWeightOnPath(p, inv_arg=inv_arg)
        else:
            for i in range(len(p)):
                self.reverseEdgeAndWeightOnPath(p[i], inv_arg=inv_arg)

        return True

    def reverseEdgeAndWeightOnPath(self, p, inv_arg=False):

        #h = self.g.copy()
        for i in range(len(p)):
            if inv_arg:
                startNode = p[i][1]
                endNode = p[i][0]
            else:
                startNode = p[i][0]
                endNode = p[i][1]

            w = self.g[startNode][endNode]['weight']
            l = self.g[startNode][endNode]['label']
            o = self.g[startNode][endNode]['occupied']
            self.g.remove_edge(startNode, endNode)
            self.g.add_edge(endNode, startNode, weight=-w, label=l, occupied=o)
            #h.remove_edge(startNode,endNode)
            #h.add_edge(endNode,startNode,weight = -w,label=l,occupied=o)

        #self.g = h

        return True

    def augment(self, p, nb_paths, pstar, pstar_label, sink):

        #print('augment...')
        eSet = list(
        )  #This is where we store the edges of previous paths and pstar
        outp = list()  #Augmented family of paths

        #Build edge set
        if (nb_paths == 1):
            eSet = p
            p = [p]
        else:
            eSet = [e for sublist in p for e in sublist]

        eSet += pstar

        #Augment previous paths: Three cases:
        # 1. Path can be augmented with itself (found its own label)
        # 2. Own label not found (is occupied by pstar)
        # 2.1 Branch out to pstar
        # 2.2 Branch out to another path
        for i in reversed(range(nb_paths)):
            end_node = ''
            outp.append([])
            start_e = p[i][0]  #First edge is always correct
            orig_label = self.g[start_e[0]][start_e[1]][
                'label']  #We are tracking this label until sink
            next_label = orig_label  #This is changed if pstar takes over
            outp[-1].append(start_e)
            while (end_node != sink):
                this_out_e_list = [
                    e for e in self.g.out_edges(outp[-1][-1][1])
                    if (self.g[e[0]][e[1]]['occupied'] == 0)
                ]  #possible edges to go to
                this_out_e_list = [e for e in this_out_e_list
                                   if (e in eSet)]  #possible edges to go to
                labels_out = [
                    self.g[e[0]][e[1]]['label'] for e in this_out_e_list
                ]
                if (orig_label not in labels_out):
                    if (pstar_label in labels_out):  #Case 2.1
                        this_out_e = this_out_e_list[labels_out.index(
                            pstar_label)]
                    else:  #Case 2.2
                        matched_e = [(i, e) for i, e in enumerate(eSet)
                                     if ((e[0] == end_node))]
                        this_out_e = matched_e[0][1]
                else:  #Case 1
                    next_label = orig_label  #OK, continue following orig_label
                    this_out_e = [
                        e for e in this_out_e_list
                        if self.g[e[0]][e[1]]['label'] == next_label
                    ]
                    this_out_e = this_out_e[0]

                outp[-1].append(this_out_e)
                self.g[this_out_e[0]][this_out_e[1]][
                    'label'] = orig_label  #In case edge belongs to pstar, we update it
                end_node = outp[-1][-1][1]
                eSet = [e for e in eSet if (e != this_out_e)]

        #Augment pstar
        # 1. Path can be augmented with itself (found its own label)
        # 2. Own label not found (is occupied by a path of above section)
        # 2.1 Find matching edge in eSet
        outp.append([])
        start_e = pstar[0]
        end_node = start_e[1]
        orig_label = pstar_label
        next_label = orig_label
        outp[-1].append(start_e)
        while (end_node != sink):
            this_out_e_list = self.g.out_edges(outp[-1][-1][1])
            this_out_e_list = [
                e for e in this_out_e_list
                if (self.g[e[0]][e[1]]['occupied'] == 0)
            ]
            labels_out = [self.g[e[0]][e[1]]['label'] for e in this_out_e_list]
            if (orig_label not in labels_out):
                matched_e = [(i, e) for i, e in enumerate(eSet)
                             if ((e[0] == end_node))]
                this_out_e = matched_e[0][1]
            else:
                this_out_e = this_out_e_list[labels_out.index(orig_label)]

            eSet = [e for e in eSet if (e != this_out_e)]
            outp[-1].append(this_out_e)
            self.g[this_out_e[0]][this_out_e[1]]['label'] = orig_label
            #self.g[this_out_e[0]][this_out_e[1]]['occupied'] = 0
            end_node = outp[-1][-1][1]

        #Remove labels on edges not taken by paths in augmented set
        used_edges = [e for sublist in outp for e in sublist]
        unused_edges = [e for e in pstar if (e not in used_edges)]
        for i in range(len(unused_edges)):
            this_e = unused_edges[i]
            self.g[this_e[1]][this_e[0]]['label'] = 0

        #print('done...')
        return outp

    def nodes2splitted(self, p):

        #Make path with splitted nodes from sink to source
        p_out = list()
        for i in range(len(p)):
            if ((p[i] != self.orig_source) and (p[i] != self.orig_sink)):
                p_out.append((p[i], self.nodeTypes.index('input')))
                p_out.append((p[i], self.nodeTypes.index('output')))
            else:
                p_out.append((p[i], self.nodeTypes.index('virtual')))

        return p_out

    def addTypesToNodes(self):
        #Relabel nodes n to (n,t) where t is from nodeTypes. Source and sink are 'virtual', others are 'none'
        n = self.g.nodes()
        map_dict = dict()
        for i in range(len(n)):
            if ((n[i] != self.orig_source) and (n[i] != self.orig_sink)):
                map_dict[n[i]] = (n[i], self.nodeTypes.index('none'))
            else:
                map_dict[n[i]] = (n[i], self.nodeTypes.index('virtual'))
        self.g = nx.relabel_nodes(self.g, map_dict)

    def removeTypesFromNodes(self):

        n = self.g.nodes()
        map_dict = dict()
        for i in range(len(n)):
            map_dict[n[i]] = (n[i][0])
        self.g = nx.relabel_nodes(self.g, map_dict)

    def splitNodes(self, p, label=0):

        #Extract sets: virt_nodes (source and sink), p_nodes (nodes on path not virtual), other_nodes (the rest)
        virt_nodes = [
            n for n in self.g.nodes() if (self.nodeTypes[n[1]] == 'virtual')
        ]
        p_nodes = [
            n for n in self.g.nodes()
            if ((self.nodeTypes[n[1]] == 'none') and (n[0] in p))
        ]
        other_nodes = [
            n for n in self.g.nodes()
            if ((n not in virt_nodes) and (n not in p_nodes))
        ]

        #Nodes in p_nodes give 2 nodes each (input and output)
        for i in range(len(p_nodes)):
            in_node = (p_nodes[i][0], self.nodeTypes.index('input'))
            out_node = (p_nodes[i][0], self.nodeTypes.index('output'))
            self.g.add_node(in_node)
            self.g.add_node(out_node)

        #Build list of edges
        new_edges = list()
        for n in p_nodes:
            in_e = self.g.in_edges(n)
            for i in range(len(in_e)):
                n_start = in_e[i][0]
                n_end = in_e[i][1]
                if (n_start in p_nodes):
                    this_n_start = (n_start[0], self.nodeTypes.index('output'))
                    this_n_end = (n_end[0], self.nodeTypes.index('input'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))
                elif (n_start in virt_nodes):
                    this_n_start = (n_start[0],
                                    self.nodeTypes.index('virtual'))
                    this_n_end = (n_end[0], self.nodeTypes.index('input'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))
                elif (n_start in other_nodes):
                    this_n_start = (n_start[0], self.nodeTypes.index('none'))
                    this_n_end = (n_end[0], self.nodeTypes.index('input'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))
            out_e = self.g.out_edges(n)
            for i in range(len(out_e)):
                n_start = out_e[i][0]
                n_end = out_e[i][1]
                if (n_end in p_nodes):
                    this_n_start = (n_start[0], self.nodeTypes.index('output'))
                    this_n_end = (n_end[0], self.nodeTypes.index('none'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))
                elif (n_end in virt_nodes):
                    this_n_start = (n_start[0], self.nodeTypes.index('output'))
                    this_n_end = (n_end[0], self.nodeTypes.index('virtual'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))
                elif (n_end in other_nodes):
                    this_n_start = (n_start[0], self.nodeTypes.index('output'))
                    this_n_end = (n_end[0], self.nodeTypes.index('none'))
                    new_edges.append((this_n_start, this_n_end,
                                      self.g[n_start][n_end]['label'],
                                      self.g[n_start][n_end]['weight'],
                                      self.g[n_start][n_end]['occupied']))

        #Add new edges
        for i in range(len(new_edges)):
            n_start = new_edges[i][0]
            n_end = new_edges[i][1]
            lab = new_edges[i][2]
            w = new_edges[i][3]
            occ = new_edges[i][4]
            self.g.add_edge(n_start, n_end, label=lab, weight=w, occupied=occ)

        #Link auxiliary nodes
        for i in range(len(p_nodes)):
            in_node = (p_nodes[i][0], self.nodeTypes.index('input'))
            out_node = (p_nodes[i][0], self.nodeTypes.index('output'))
            self.g.add_edge(in_node, out_node, label=-1, weight=0, occupied=0)

            #in_ind = [i for i,v in enumerate(p) if(v[1] == self.nodeTypes.index('input'))]

        #Remove old (non-auxiliary) nodes
        self.g.remove_nodes_from(p_nodes)

        #Relabel source and sink
        self.source = (self.orig_source, self.nodeTypes.index('virtual'))
        self.sink = (self.orig_sink, self.nodeTypes.index('virtual'))

        return True

    def nodes2Paths(self, n, nb_NodeList):

        p = list()
        if (nb_NodeList > 1):
            for i in range(len(n)):
                p.append(self.nodes2Path(n[i]))
        else:
            p = self.nodes2Path(n)

        return p

    def nodes2Path(self, n):

        p = list()
        for i in np.arange(1, len(n)):
            p.append((n[i - 1], n[i]))

        return p

    def paths2Nodes(self, p, npaths):

        nodeLists = list()
        if (npaths > 1):
            for i in range(len(p)):
                nodeLists.append(self.path2Nodes(p[i]))
        else:
            nodeLists = self.path2Nodes(p)

        return nodeLists

    def path2Nodes(self, p):

        n = list()
        n.append(p[0][0])
        for i in np.arange(1, len(p)):
            n.append(p[i][0])

        n.append(p[-1][1])
        return n

    def costTransform(self, set_ind):

        #print('costTransform...')
        def do_it(dist,e):

            for i in range(len(e)):
                self.g[e[i][0]][e[i][1]]['weight'] = np.around(self.g[e[i][0]][e[i][1]]['weight'] + dist[e[i][0]] - dist[e[i][1]], decimals=7)

        if ((self.mode == 'node') and (set_ind == 1)):
            dist = self.sp[0]
            this_dist = {
                n: dist[n[0]]
                for n in self.g.nodes() if (n[0] in dist.keys())
            }
            do_it(this_dist)
        elif ((self.mode == 'node') and (set_ind > 1)):
            dist = self.sp[set_ind - 1]
            this_dist = {k[0]: dist[k] for k in dist.keys()}
            do_it(this_dist)
            self.costTransform(set_ind - 1)
        elif ((self.mode == 'edge') and (set_ind == 1)):
            dist = self.sp[0]
            es = [e for e in self.g.edges() if((e[0] in dist) and (e[1] in dist))]
            do_it(dist,es)
        elif ((self.mode == 'edge') and (set_ind > 1)):
            dist = self.sp[set_ind - 1]
            es = [e for e in self.g.edges() if((e[0] in dist) and (e[1] in dist))]
            do_it(dist,es)

        #print('done')
        return True

    def predList2Nodes(self, pred):
        p = list()
        n = self.sink
        while (n != self.source):
            p.append(n)
            n = pred[n]

        p.append(self.source)
        return p[::-1]

    def interlacing(self, pstar, pstar_label):

        #print('interlacing...')
        #Identify already occupied edges
        taken_edges = [
            e for e in self.g.edges() if self.g[e[0]][e[1]]['label'] != 0
        ]

        #Set pstar label
        for i in range(len(pstar)):
            this_e = pstar[i]
            if (this_e in taken_edges):
                self.g[this_e[0]][this_e[1]]['occupied'] = 1

            self.g[this_e[0]][this_e[1]]['label'] = pstar_label

        #print('done')
        return True

    def unlabelEdgesNotInPaths(self, p):

        occupied_edge_list = [e for sublist in p for e in sublist]
        edge_list = self.g.edges()
        edges_to_unlabel = [
            e for e in edge_list if (e not in occupied_edge_list)
        ]

        for i in range(len(edges_to_unlabel)):
            this_e = edges_to_unlabel[i]
            self.g[this_e[0]][this_e[1]]['label'] = 0

        return True

    def unOccupy(self):

        e = self.g.edges()
        for i in range(len(e)):
            self.g[e[i][0]][e[i][1]]['occupied'] = 0

        return True

    def setAllLabelToNull(self):

        e = self.g.edges()
        for i in range(len(e)):
            this_e = e[i]
            self.g[this_e[0]][this_e[1]]['label'] = 0

        return True

    def setWeightOnPaths(self, p, nb_paths, weight):

        if (nb_paths == 1): p = [p]

        for i in range(len(p)):
            for j in range(len(p[i])):
                this_e = p[i][j]
                self.g[this_e[0]][this_e[1]]['weight'] = weight
        return True

    def setLabelOnPaths(self, p, nb_paths, labels):

        if (nb_paths == 1): p = [p]

        for i in range(len(p)):
            for j in range(len(p[i])):
                this_e = p[i][j]
                self.g[this_e[0]][this_e[1]]['label'] = labels[i]

        return True

    def addOneNodeDisjointSP(self):

        if (self.nb_sets == 0):
            self.addOneEdgeDisjointSP()
            self.source = self.orig_source
            self.sink = self.orig_sink
        else:
            self.addTypesToNodes()
            if (self.nb_sets == 1):
                self.splitNodes(self.kspSet[-1], label=1)
                prevSet = self.nodes2splitted(self.kspSet[-1])
                self.setLabelOnPaths(
                    self.nodes2Path(prevSet), 1, np.array([1]))
            else:
                nodes_to_split = [
                    item for sublist in self.kspSet[-1] for item in sublist
                ]
                self.splitNodes(nodes_to_split)
                prevSet = list()
                for i in range(self.nb_sets):
                    prevSet.append(self.nodes2splitted(self.kspSet[-1][i]))

                for i in range(len(prevSet)):
                    self.setLabelOnPaths(
                        self.nodes2Path(prevSet[i]), 1, np.array([i + 1]))

            self.addOneEdgeDisjointSP(prevSet)

            self.g = self.g_orig

            coalesced_p = list()
            for i in range(len(self.kspSet[-1])):
                coalesced_p.append(
                    self.coalesceNodesInPath(self.kspSet[-1][i]))

            self.kspSet[-1][:] = coalesced_p

        return True

    def addOneEdgeDisjointSP(self, prevSet=None):

        if (prevSet == None):
            if (len(self.kspSet) == 0):
                prevSet = []
            else:
                prevSet = self.kspSet[-1]

        if (len(prevSet) == 0):
            p_pred, p_dist = nx.bellman_ford(self.g, self.source)
            this_nodes = self.predList2Nodes(p_pred)
            this_path = self.nodes2Path(this_nodes)
            self.kspSet.append(this_nodes)
            self.setLabelOnPaths(this_path, 1, np.array([1]))
            self.sp.append(p_dist)
            self.costs.append(self.costOfPaths([this_path]))
        else:
            #self.g = self.g_orig.copy() #will fix that later
            #self.setLabelOnPaths(self.nodes2Paths(prevSet,self.nb_sets),self.nb_sets,np.arange(1,self.nb_sets+1))
            self.reverseEdgeAndWeightOnPaths(
                self.nodes2Paths(prevSet, self.nb_sets),
                self.nb_sets,
                inv_arg=False)
            #cost transform!
            self.costTransform(self.nb_sets)
            #self.setWeightOnPaths(self.nodes2Paths(prevSet,self.nb_sets),self.nb_sets,0)
            p_dist, p_pred = nx.single_source_dijkstra(self.g, self.source,
                                                       self.sink)
            self.sp.append(p_dist)
            pstar = self.nodes2Path(p_pred[self.sink])

            self.interlacing(pstar, self.nb_sets + 1)
            self.reverseEdgeAndWeightOnPaths(
                self.nodes2Paths(prevSet, self.nb_sets),
                self.nb_sets,
                inv_arg=True)
            newSet = self.augment(
                self.nodes2Paths(prevSet, self.nb_sets), self.nb_sets, pstar,
                self.nb_sets + 1, self.sink)

            self.kspSet.append(self.paths2Nodes(newSet, self.nb_sets + 1))
            #self.unlabelEdgesNotInPaths(newSet)
            self.unOccupy()
            self.costs.append(self.costOfPaths(newSet))

        self.nb_sets += 1
        return True

    def disjointKSP(self, K=None, verbose=False):

        if (K is not None):
            if (self.nb_sets >= K): return self.kspSet[K - 1]
        else:
            orig_nb_sets = self.nb_sets
            #nb_extraPathsToCompute = K - orig_nb_sets
            n_paths = orig_nb_sets
            done = False
            while (done == False):
                try:
                    if (self.mode == 'edge'):
                        self.addOneEdgeDisjointSP()
                    elif (self.mode == 'node'):
                        self.addOneNodeDisjointSP()
                    if (verbose):
                        n_nodes = 0
                        for i in range(len(self.kspSet[-1])):
                            n_nodes += len(self.kspSet[-1][i])
                        print("K = %d, cost = %f, n_nodes = %d" % (n_paths + 1,
                                                                   np.sum(self.costs[-1]),
                                                                   n_nodes))
                    n_paths += 1
                    if (K is not None):
                        if (n_paths == K): done = True
                    elif (len(self.costs) >
                          1):  #stop condition = find min cost
                        if (np.sum(self.costs[-1]) > np.sum(self.costs[-2])):
                            done = True
                            K = n_paths
                        elif (np.abs(np.sum(self.costs[-1]) - np.sum(self.costs[-2]))/np.abs(np.sum(self.costs[-1]))) < self.tol:
                            print("hit tolerance threshold")
                            done = True
                            K = n_paths

                except (KeyError, IndexError):
                    print("Couldnt add additional path. Stopped at K=",
                          self.nb_sets)
                    return self.kspSet[-1]
            sys.stdout.write("\n")
            sys.stdout.flush()
            return self.kspSet[K - 1]
        return True

    def reset(self):
        #Makes graph good as new. Called after reevaluation of models (bagging).
        self.g = self.g_orig.copy()
        self.kspSet = []
        self.costs = []
        self.sp = []
        self.nb_sets = 0
        #self.setAllLabelToNull()
        #self.unOccupy()

    def pairEdgeDisjointKSP(self, g, source, sink, coalesceAndReverse=True):

        return g, p

    def coalesceNodesInGraph(self, nodeList, label=0):

        n_i = [n for n in nodeList if n[1] == self.nodeTypes.index('input')]
        n_o = [n for n in nodeList if n[1] == self.nodeTypes.index('output')]

        #Remove auxiliary edges
        e_aux = [e for e in self.g.edges() if (e[0][0] == e[1][0])]
        self.g.remove_edges_from(e_aux)

        #Add "normal" nodes
        for i in range(len(n_i)):
            self.g.add_node((n_i[i][0], self.nodeTypes.index('none')))

        new_edges = list()
        for i in range(len(n_i)):
            in_e = self.g.in_edges(n_i[i])
            for j in range(len(in_e)):
                this_n_start = ((in_e[j][0][0]), self.nodeTypes.index('none'))
                this_n_end = ((in_e[j][1][0]), self.nodeTypes.index('none'))
                this_label = self.g[in_e[j][0]][in_e[j][1]]['label']
                this_weight = self.g[in_e[j][0]][in_e[j][1]]['weight']
                this_occ = self.g[in_e[j][0]][in_e[j][1]]['occupied']
                new_edges.append((this_n_start, this_n_end, this_label,
                                  this_weight, this_occ))
            out_e = self.g.out_edges(n_i[i])
            for j in range(len(out_e)):
                this_n_start = ((out_e[j][0][0]), self.nodeTypes.index('none'))
                this_n_end = ((out_e[j][1][0]), self.nodeTypes.index('none'))
                this_label = self.g[out_e[j][0]][out_e[j][1]]['label']
                this_weight = self.g[out_e[j][0]][out_e[j][1]]['weight']
                this_occ = self.g[out_e[j][0]][out_e[j][1]]['occupied']
                new_edges.append((this_n_start, this_n_end, this_label,
                                  this_weight, this_occ))

        for i in range(len(n_o)):
            in_e = self.g.in_edges(n_o[i])
            for j in range(len(in_e)):
                this_n_start = ((in_e[j][0][0]), self.nodeTypes.index('none'))
                this_n_end = ((in_e[j][1][0]), self.nodeTypes.index('none'))
                this_label = self.g[in_e[j][0]][in_e[j][1]]['label']
                this_weight = self.g[in_e[j][0]][in_e[j][1]]['weight']
                this_occ = self.g[in_e[j][0]][in_e[j][1]]['occupied']
                new_edges.append((this_n_start, this_n_end, this_label,
                                  this_weight, this_occ))
            out_e = self.g.out_edges(n_o[i])
            for j in range(len(out_e)):
                this_n_start = ((out_e[j][0][0]), self.nodeTypes.index('none'))
                this_n_end = ((out_e[j][1][0]), self.nodeTypes.index('none'))
                this_label = self.g[out_e[j][0]][out_e[j][1]]['label']
                this_weight = self.g[out_e[j][0]][out_e[j][1]]['weight']
                this_occ = self.g[out_e[j][0]][out_e[j][1]]['occupied']
                new_edges.append((this_n_start, this_n_end, this_label,
                                  this_weight, this_occ))

        for i in range(len(new_edges)):
            this_e = (new_edges[i][0], new_edges[i][1])
            this_label = new_edges[i][2]
            this_weight = new_edges[i][3]
            this_occ = new_edges[i][4]
            self.g.add_edge(
                this_e[0], this_e[1], weight=this_weight, label=this_label)

        self.g.remove_nodes_from(n_i)
        self.g.remove_nodes_from(n_o)

        return True

    def coalesceNodesInPath(self, p):

        newp = list()
        in_ind = [
            i for i, v in enumerate(p)
            if (v[1] == self.nodeTypes.index('input'))
        ]
        out_ind = [
            i for i, v in enumerate(p)
            if (v[1] == self.nodeTypes.index('output'))
        ]
        virt_ind = [
            i for i, v in enumerate(p)
            if (v[1] == self.nodeTypes.index('virtual'))
        ]
        none_ind = [
            i for i, v in enumerate(p)
            if (v[1] == self.nodeTypes.index('none'))
        ]

        for i in range(len(in_ind)):
            p[in_ind[i]] = p[in_ind[i]][0]

        for i in range(len(virt_ind)):
            p[virt_ind[i]] = p[virt_ind[i]][0]

        for i in range(len(none_ind)):
            p[none_ind[i]] = p[none_ind[i]][0]

        for i in sorted(out_ind, reverse=True):
            del p[i]

        return p
