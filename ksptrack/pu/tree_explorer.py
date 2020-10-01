#!/usr/bin/env python3
import numpy as np
from skimage.measure import regionprops
import higra as hg
from ksptrack.utils.loc_prior_dataset import relabel
import networkx as nx


def candidate_subtrees(tree, label_map, leaves_weights, clicked_coords=[]):
    # clicked_coords is a list of ij coordinates
    # returns list [(accumulated_weight, [l0, ...])]
    # where [l0, ...] are reachable leaves
    # leaves_weights will be propagated.
    # 1d array: It is taken as an area-normalized (average pooling) metric
    # 2d array: It will be average pooled according to label_map
    #
    # returns:
    #  - weights of each node
    #  - indices of potential parent nodes.
    #          A parent node means that _all_ its reachable leaves can form a cut

    leaves = np.unique(label_map)
    regions = regionprops(label_map + 1)
    area_leaves = np.array([p['area'] for p in regions])

    if (leaves_weights.ndim == 2):
        regions = regionprops(label_map + 1, intensity_image=leaves_weights)
        mean_intensities = np.array([p['mean_intensity'] for p in regions])
        leaves_weights = np.array([
            mean_inten * area
            # mean_inten
            for mean_inten, area in zip(mean_intensities, area_leaves)
        ])
    elif (leaves_weights.ndim == 1):
        leaves_weights = leaves_weights * area_leaves
    else:
        raise TypeError('leaves_weights: Only 1-D and 2-D arrays supported.')

    areas = hg.accumulate_sequential(tree, area_leaves, hg.Accumulators.sum)

    #each node is area-weighted sum of weights
    accum_weights = hg.accumulate_sequential(tree, leaves_weights,
                                             hg.Accumulators.sum) / areas
    # accum_weights = hg.accumulate_sequential(tree, leaves_weights,
    #                                          hg.Accumulators.sum)

    # each node is area-weighted sum of weights
    # accum_weights = hg.accumulate_sequential(tree, sums, hg.Accumulators.sum)

    # get list [(a, [l0, ...])] where a is a candidate ancestor and [l0, ...] are its leaves
    ok_parents = []
    for r in clicked_coords:
        clicked_label = label_map[tuple(r)]
        ok_parents.extend(tree.ancestors(clicked_label)[1:])

    return accum_weights, np.array(ok_parents)


def sum_accumulate(G, leaves, attr=['area', 'weight']):
    for p in G.nodes:
        if p not in leaves:
            for att in attr:
                att_ = 0
                for c in G.successors(p):
                    att_ += G.nodes[c][att]
                G.nodes[p][att] = att_
    return G


def candidate_subtrees_ss(tree, label_map, leaves_weights, clicked_coords=[]):
    # clicked_coords is a list of ij coordinates
    # returns list [(accumulated_weight, [l0, ...])]
    # where [l0, ...] are reachable leaves
    # leaves_weights will be propagated.
    # 1d array: It is taken as an area-normalized (average pooling) metric
    # 2d array: It will be average pooled according to label_map
    #
    # returns:
    #  - weights of each node
    #  - indices of potential parent nodes.
    #          A parent node means that _all_ its reachable leaves can form a cut

    leaves = np.unique(label_map)
    regions = regionprops(label_map + 1)
    area_leaves = np.array([p['area'] for p in regions])

    if (leaves_weights.ndim == 2):
        regions = regionprops(label_map + 1, intensity_image=leaves_weights)
        mean_intensities = np.array([p['mean_intensity'] for p in regions])
        leaves_weights = np.array([
            mean_inten * area
            for mean_inten, area in zip(mean_intensities, area_leaves)
        ])
    elif (leaves_weights.ndim == 1):
        leaves_weights = leaves_weights * area_leaves
    else:
        raise TypeError('leaves_weights: Only 1-D and 2-D arrays supported.')

    leaves_attr = {
        n: {
            'area': a,
            'weight': w
        }
        for n, a, w in zip(leaves, area_leaves, leaves_weights)
    }

    nx.set_node_attributes(tree, leaves_attr)

    tree = sum_accumulate(tree, leaves)

    for n in tree.nodes:
        tree.nodes[n][
            'norm_weight'] = tree.nodes[n]['weight'] / tree.nodes[n]['area']

    # get list [(a, [l0, ...])] where a is a candidate ancestor and [l0, ...] are its leaves
    ok_parents = []
    for r in clicked_coords:
        clicked_label = label_map[tuple(r)]
        ok_parents.extend(nx.ancestors(tree, clicked_label))

    return tree, np.array(ok_parents)


class TreeExplorer:
    """
    """
    def __init__(self,
                 tree,
                 label_map,
                 leaves_weights,
                 clicked_coords=[],
                 ascending=False):

        label_map = relabel(label_map)

        self.leaves = np.unique(label_map)
        self.tree = tree
        self.label_map = label_map
        self.leaves_weights = leaves_weights
        self.clicked_coords = clicked_coords
        self.clicked_labels = [
            self.label_map[n[0], n[1]] for n in self.clicked_coords
        ]
        self.clicked_label_map = np.array(
            [self.label_map == l for l in self.clicked_labels]).sum(axis=0)

        self.tree, self.parents = candidate_subtrees_ss(
            self.tree, self.label_map, self.leaves_weights,
            self.clicked_coords)

        # sort parents by weights
        self.parents_weights = [{
            'nodes': p,
            'weight': self.tree.nodes[p]['norm_weight']
        } for p in self.parents]
        self.parents_weights.sort(key=lambda x: x['weight'])
        if ascending == False:
            self.parents_weights = self.parents_weights[::-1]

    def __len__(self):
        return len(self.parents_weights)

    def __getitem__(self, idx):
        # this will return all leaf-nodes
        nodes = self.inorder_traversal(self.parents_weights[idx]['nodes'])
        nodes = [n for n in nodes if n in self.leaves]
        return {
            'nodes': nodes,
            'weight': self.parents_weights[idx]['weight'],
            'n_nodes': len(nodes)
        }

    def inorder_traversal(self, root):
        res = []
        # if not self.tree.is_leaf(root):
        children = list(self.tree.successors(root))
        if len(children) > 0:
            res = self.inorder_traversal(children[0])
            res.append(root)
            res = res + self.inorder_traversal(children[1])
        else:
            res.append(root)
        return res
