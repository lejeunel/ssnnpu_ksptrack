import datetime
import itertools
import copy
import joblib
import numpy
import scipy.sparse
import collections
import skimage.io
from . import features
from . import color_space
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import progressbar


def relabel(labels, map_dict):
    shape = labels.shape
    labels = labels.ravel()
    new_labels = np.copy(labels)
    for k, v in map_dict.items():
        new_labels[labels == k] = v

    return new_labels.reshape(shape)


def intersect(lists):
    return list(set.intersection(*map(set, lists)))


def set_ratios(g, marked):

    #Get leaves
    n_elems = nx.get_node_attributes(g, 'n_elems')
    leaves = [n for n in g.nodes() if (n_elems[n] == 1)]
    leaves_marked = [l for l in leaves if (l in marked)]

    #Mark leaves
    nx.set_node_attributes(g, 'marked', False)
    for i in range(len(leaves_marked)):
        g.node[leaves_marked[i]]['marked'] = True

    #Init ratios
    nx.set_node_attributes(g, 'ratio', 0.)

    #Set ratios (n_pos_leaves/n_leaves)
    non_leaves = [n for n in g.nodes() if (n_elems[n] > 1)]
    for i in range(len(non_leaves)):
        #this_leaves = g.successors(non_leaves[i])
        this_leaves = [n for n in nx.dfs_preorder_nodes(g, non_leaves[i])]
        this_leaves = [l for l in this_leaves if (g.node[l]['n_elems'] == 1)]
        this_pos_leaves = [
            l for l in this_leaves if (g.node[l]['marked'] == True)
        ]
        g.node[
            non_leaves[i]]['ratio'] = len(this_pos_leaves) / len(this_leaves)

    return g


def thr_all_graphs(g, marked, thr):

    new_marked = []
    #Mark all and set ratios
    with progressbar.ProgressBar(maxval=len(g)) as bar:
        for f in range(len(g)):
            bar.update(f)
            m = marked[marked[:, 0] == f, :][:, 1]
            g_marked = set_ratios(g[f], m)
            new_m = thr_graph(g_marked, thr)
            new_m += m.tolist()
            new_m = np.asarray(list(set(new_m)))
            new_m = np.concatenate(
                (np.tile(f, new_m.shape[0]).reshape(-1, 1), new_m.reshape(
                    -1, 1)),
                axis=1).astype(int)
            new_marked.append(new_m)

    return np.concatenate(new_marked)


def thr_graph(g, thr):

    ratios = nx.get_node_attributes(g, 'ratios')
    #n = [n for n in g.nodes() if((g.node[n]['ratio']>thr) & (g.node[n]['n_elems']>1))]
    n = [n for n in g.nodes() if ((g.node[n]['ratio'] > thr))]

    leaves = []
    for i in range(len(n)):
        this_leaves = [m for m in nx.dfs_preorder_nodes(g, n[i])]
        #this_leaves = g.successors(n[i])
        this_leaves = [l for l in this_leaves if (g.node[l]['n_elems'] == 1)]
        leaves.append(this_leaves)

    #Add "lost nodes"

    #Remove duplicates
    merged = [item for sublist in leaves for item in sublist]
    merged = list(set(merged))

    #Print added nodes
    #marked = [n for n in g.nodes() if(g.node[n]['marked'] == True)]
    #new = [n for n in merged if(n not in marked)]
    #print(new)

    return merged


def get_merge_candidates(g, candidates, thr):
    """
    Returns:
        merged: List of (label,stack_idx)

    """

    n_elems = nx.get_node_attributes(g, 'n_elems')
    stack = nx.get_node_attributes(g, 'stack')
    merged_parents = []
    merged_children = []

    redraw = False

    K = len(candidates)
    while ((len(candidates) > 0) and (K > 1)):
        #for i in np.arange(2, len(children) + 1):
        this_combs = list(combinations(candidates, K))
        redraw = False
        print('n_candidates, K, len(combs): ' + str(len(candidates)) + ',' +
              str(K) + ',' + str(len(this_combs)))
        for j in range(len(this_combs)):
            #print(K)
            ancestors = []
            for k in range(len(this_combs[j])):
                ancestors.append(list(nx.ancestors(g, this_combs[j][k])))
            #common_parents  = list(set(ancestors[0]).intersection(*ancestors[:1]))
            common_parents = intersect(ancestors)
            sum_elems_candidates = np.sum(
                [n_elems[this_combs[j][c]] for c in range(len(this_combs[j]))])
            #Compute ratio of combination with all common parents
            ratios = np.asarray([
                sum_elems_candidates / (n_elems[common_parents[l]])
                for l in range(len(common_parents))
            ])
            #ratios = ratios.sort()[::-1]

            idx_can_merge = np.where((ratios > thr) & (ratios <= 1))[0]
            ratios_above = ratios[idx_can_merge]
            if (idx_can_merge.size > 0):
                #Find lower ratio and add to merged set
                min_ratio_idx = idx_can_merge[np.argmin(ratios_above)]
                #Store label and scale
                to_append = common_parents[min_ratio_idx]
                merged_parents.append((to_append, stack[to_append]))
                children = get_children(g, to_append)
                merged_children.append(children)
                candidates = [
                    c for c in candidates if (c not in this_combs[j])
                ]
                redraw = True
                break
        if (redraw == False):
            K -= 1

    #candidates = [(candidates[i],stack[candidates[i]]) for i in range(len(candidates))]
    #print('done')

    #if(len(merged_children) > 0):
    #    merged_children = np.concatenate(merged_children)

    return merged_parents, merged_children, candidates


def get_children(g, n):

    children = list(nx.dfs_preorder_nodes(g, n))
    stack = nx.get_node_attributes(g, 'stack')
    children = np.asarray([c for c in children
                           if (stack[c] == 0)]).reshape(-1, 1)

    return children


def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors


def _calc_adjacency_matrix(label_img, n_region):
    r = numpy.vstack([label_img[:, :-1].ravel(), label_img[:, 1:].ravel()])
    b = numpy.vstack([label_img[:-1, :].ravel(), label_img[1:, :].ravel()])
    t = numpy.hstack([r, b])
    A = scipy.sparse.coo_matrix((numpy.ones(t.shape[1]), (t[0], t[1])),
                                shape=(n_region, n_region),
                                dtype=bool).todense().getA()
    A = A | A.transpose()

    for i in range(n_region):
        A[i, i] = True

    dic = {i: {i} ^ set(numpy.flatnonzero(A[i])) for i in range(n_region)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix=A, dictionary=dic)


def _new_adjacency_dict(A, i, j, t):
    Ak = copy.deepcopy(A)
    Ak[t] = (Ak[i] | Ak[j]) - {i, j}
    del Ak[i], Ak[j]
    for (p, Q) in Ak.items():
        if i in Q or j in Q:
            Q -= {i, j}
            Q.add(t)

    return Ak


def _new_label_image(F, i, j, t):
    Fk = numpy.copy(F)
    Fk[Fk == i] = Fk[Fk == j] = t
    return Fk


def _build_initial_similarity_set(A0, feature_extractor):
    S = list()

    for (i, J) in A0.items():
        S += [(feature_extractor.similarity(i, j), (i, j)) for j in J if i < j]

    return sorted(S)


def _merge_similarity_set(feature_extractor, Ak, S, i, j, t):
    # remove entries which have i or j
    S = list(filter(lambda x: not (i in x[1] or j in x[1]), S))

    # calculate similarity between region t and its adjacencies
    St = [(feature_extractor.similarity(t, x), (t, x)) for x in Ak[t] if t < x] +\
         [(feature_extractor.similarity(x, t), (x, t)) for x in Ak[t] if x < t]

    return sorted(S + St)


def hierarchical_segmentation(I,
                              feature_mask=features.SimilarityMask(1, 1, 1, 1),
                              F0=None,
                              k=100,
                              to_maxpool=None):
    """
    I: Image

    Returns:
    R: Merge dictionary (key: label of parent, value: (label of child1, label of child2))
    scales: scale dictionary (key: label, value: scale)
    """
    relabeled = False

    #pass
    #Remap if labels are not contiguous
    sorted_labels = np.asarray(sorted(np.unique(F0).ravel()))
    if (np.any((sorted_labels[1:] - sorted_labels[0:-1]) > 1)):
        relabeled = True
        map_dict = {sorted_labels[i]: i for i in range(sorted_labels.shape[0])}
        F0 = relabel(F0, map_dict)

    n_region = np.unique(F0.ravel()).shape[0]
    adj_mat, A0 = _calc_adjacency_matrix(F0, n_region)
    feature_extractor = features.Features(I, F0, n_region, feature_mask)

    # stores list of regions sorted by their similarity
    S = _build_initial_similarity_set(A0, feature_extractor)

    #Initialize scale dictionary
    unique_labels = np.unique(F0.ravel()).tolist()

    g = nx.DiGraph()
    if (to_maxpool is None):
        g.add_nodes_from(np.unique(F0), n_elems=1, stack=0)
    else:
        g.add_nodes_from([(l, {
            'n_elems': 1,
            'stack': 0,
            'pooled': to_maxpool[F0 == l].max()
        }) for l in np.unique(F0)])

    # stores region label and its parent (empty if initial).
    R = {i: () for i in range(n_region)}

    A = [A0]  # stores adjacency relation for each step
    F = [F0]  # stores label image for each step

    if (to_maxpool is not None):
        h0 = np.zeros(F0.shape)
        for l in np.unique(F0):
            h0 += np.max(to_maxpool[F0 == l]) * (F0 == l)
        H = [h0]

    # greedy hierarchical grouping loop
    stack = 0
    while len(S):
        stack += 1
        (s, (i, j)) = S.pop()
        t = feature_extractor.merge(i, j)
        n_elems_left = nx.get_node_attributes(g, 'n_elems')[i]
        n_elems_right = nx.get_node_attributes(g, 'n_elems')[j]
        new_n_elems = n_elems_left + n_elems_right
        g.add_node(t, n_elems=new_n_elems, stack=stack)
        g.add_edge(t, i, diff_n_elems=new_n_elems - n_elems_left)
        g.add_edge(t, j, diff_n_elems=new_n_elems - n_elems_right)

        # record merged region (larger region should come first)
        R[t] = (
            i,
            j) if feature_extractor.size[j] < feature_extractor.size[i] else (
                j, i)

        Ak = _new_adjacency_dict(A[-1], i, j, t)
        A.append(Ak)

        S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)

        new_label_image = _new_label_image(F[-1], i, j, t)
        F.append(new_label_image)

        if (to_maxpool is not None):
            new_pooled_image = H[-1].copy()
            new_pooled_image[new_label_image == t] = to_maxpool[new_label_image
                                                                == t].max()
            H.append(new_pooled_image)
            g.node[t]['pooled'] = to_maxpool[new_label_image == t].max()

    # bounding boxes for each hierarchy
    L = feature_extractor.bbox

    if (relabeled):
        inv_map = {v: k for k, v in map_dict.items()}
        g = nx.relabel_nodes(g, inv_map, copy=True)
        #g = nx.relabel_nodes(g,map_dict)

    if (to_maxpool is not None):
        return (R, F, g, H)
    else:
        return (R, F, g)


def _generate_regions(R, L):
    n_ini = sum(not parent for parent in R.values())
    n_all = len(R)

    regions = list()
    for label in R.keys():
        i = min(n_all - n_ini + 1, n_all - label)
        vi = numpy.random.rand() * i
        regions.append((vi, L[i]))

    return sorted(regions)


def _selective_search_one(I, color, k, mask):
    I_color = color_space.convert_color(I, color)
    (R, F, L) = hierarchical_segmentation(I_color, k, mask)
    return _generate_regions(R, L)


def selective_search(I,
                     color_spaces=['rgb'],
                     ks=[100],
                     feature_masks=[features.SimilarityMask(1, 1, 1, 1)],
                     n_jobs=-1):
    parameters = itertools.product(color_spaces, ks, feature_masks)
    region_set = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_selective_search_one)(I, color, k, mask)
        for (color, k, mask) in parameters)

    #flatten list of list of tuple to list of tuple
    regions = sum(region_set, [])
    return sorted(regions)
