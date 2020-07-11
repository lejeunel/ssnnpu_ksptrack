#!/usr/bin/env python3
import numpy as np
from skimage.measure import regionprops
import higra as hg


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
            for mean_inten, area in zip(mean_intensities, area_leaves)
        ])
    elif (leaves_weights.ndim == 1):
        pass
    else:
        raise TypeError('leaves_weights: Only 1-D and 2-D arrays supported.')

    # tree, altitudes = hg.hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(
    #     tree, leaves_weights, approximation_piecewise_linear_function=10)

    areas = hg.accumulate_sequential(tree, area_leaves, hg.Accumulators.sum)

    # each node is area-weighted sum of weights
    accum_weights = hg.accumulate_sequential(tree, leaves_weights,
                                             hg.Accumulators.sum) / areas

    # each node is area-weighted sum of weights
    # accum_weights = hg.accumulate_sequential(tree, sums, hg.Accumulators.sum)

    # get list [(a, [l0, ...])] where a is a candidate ancestor and [l0, ...] are its leaves
    ok_parents = []
    for r in clicked_coords:
        clicked_label = label_map[tuple(r)]
        ok_parents.extend(tree.ancestors(clicked_label)[1:])

    return accum_weights, np.array(ok_parents)


class TreeExplorer:
    """
    """
    def __init__(self,
                 tree,
                 label_map,
                 leaves_weights,
                 clicked_coords=[],
                 sort_order='ascending'):

        sort_options = ['ascending', 'descending']
        assert (sort_order in sort_options
                ), 'sort_order must be in {}'.format(sort_options)
        self.leaves = np.unique(label_map)
        self.tree = tree
        self.label_map = label_map
        self.leaves_weights = leaves_weights
        self.clicked_coords = clicked_coords

        self.accum_weights, self.parents = candidate_subtrees(
            self.tree, self.label_map, self.leaves_weights,
            self.clicked_coords)

        # sort parents by weights
        self.parents_weights = [(p, self.accum_weights[p])
                                for p in self.parents]
        self.parents_weights.sort(key=lambda x: x[1])
        if sort_order == 'descending':
            self.parents_weights = self.parents_weights[::-1]

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, idx):
        # this will return all leaf-nodes
        nodes = self.inorder_traversal(self.parents_weights[idx][0])
        nodes = [n for n in nodes if n in self.leaves]
        return {'nodes': nodes, 'weight': self.parents_weights[idx][1]}

    def inorder_traversal(self, root):
        res = []
        if not self.tree.is_leaf(root):
            res = self.inorder_traversal(self.tree.children(root)[0])
            res.append(root)
            res = res + self.inorder_traversal(self.tree.children(root)[1])
        else:
            res.append(root)
        return res
