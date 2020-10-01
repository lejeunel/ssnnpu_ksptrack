#!/usr/bin/env python3

from ksptrack.utils.base_dataset import BaseDataset
from skimage import exposure
from os.path import join as pjoin
import matplotlib.pyplot as plt
from skimage import data, segmentation, filters, color
from skimage.future import graph
import numpy as np


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


dl = BaseDataset(pjoin('/home/ubelix/lejeune/data/medical-labeling',
                       'Dataset30'),
                 normalization='rescale',
                 resize_shape=512)
f = 50

img = (dl[f]['image'] * 255).astype(np.uint8)
labels = dl[f]['labels'].squeeze()
edges = filters.sobel(color.rgb2gray(img))

g = graph.rag_boundary(labels, edges)

graph.show_rag(labels, g, img)
plt.title('Initial RAG')

labels2 = graph.merge_hierarchical(labels,
                                   g,
                                   thresh=0.01,
                                   rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)
print('init num labels: {}'.format(np.unique(labels).size))
print('after num labels: {}'.format(np.unique(labels2).size))

graph.show_rag(labels, g, img)
plt.title('RAG after hierarchical merging')

plt.figure()
out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
plt.imshow(out)
plt.title('Final segmentation')
plt.show()

plt.imshow(edges)
plt.title('edges')
plt.show()
