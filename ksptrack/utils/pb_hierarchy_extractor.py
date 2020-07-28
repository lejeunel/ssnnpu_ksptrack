#!/usr/bin/env python3

import os
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from os.path import join as pjoin
import glob
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
import cv2 as cv
from tqdm import tqdm
from skimage import io
import urllib.request
import higra as hg
from skimage.measure import regionprops


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def get_cut_thr(tree, altitudes, thr=0.5):

    labels = hg.labelisation_horizontal_cut_from_threshold(
        tree, 1 - altitudes, 1 - thr)

    return labels


def labelisation_optimal_cut(tree, label_map, weight_map):
    weights_leaves = avg_pool(label_map, weight_map)
    mapping = hg.algo.hierarchy_to_optimal_MumfordShah_energy_cut_hierarchy(
        tree, weights_leaves)
    return mapping


def relabel(orig_labels, merge, increasing_labels=True):
    # orig_labels: initial label map
    # merge: array of shape (|unique(orig_labels)|) whose elements are new labels
    # increasing_labels will change the labels to {0,..., max(|unique(merge)|)-1}

    if (increasing_labels):
        mapping = np.concatenate(
            (np.unique(merge)[..., None], np.arange(
                np.unique(merge).size)[..., None]),
            axis=1)
        _, ind = np.unique(merge, return_inverse=True)
        merge = mapping[ind, 1:]

    shape = orig_labels.shape
    mapping = np.concatenate((np.unique(orig_labels)[..., None], merge),
                             axis=1)
    _, ind = np.unique(orig_labels, return_inverse=True)

    new_label = mapping[ind, 1:].reshape((shape[0], shape[1]))

    return new_label


def avg_pool(label_map, weight_map):
    regions = regionprops(label_map + 1, intensity_image=weight_map)
    means = np.array([p['mean_intensity'] for p in regions])

    return means


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
    #  - indices of parent nodes that are eligible (from which one can reach clicked)

    leaves = np.unique(label_map)
    regions = regionprops(label_map + 1)

    if (leaves_weights.ndim == 2):
        regions = regionprops(label_map + 1, intensity_image=leaves_weights)
        leaves_weights = np.array([p['mean_intensity'] for p in regions])
    elif (leaves_weights.ndim == 1):
        pass
    else:
        raise TypeError('leaves_weights: Only 1-D and 2-D arrays supported.')

    area_leaves = np.array([p['area'] for p in regions])
    areas = hg.accumulate_sequential(tree, area_leaves, hg.Accumulators.sum)
    sums = leaves_weights * area_leaves

    # each node is area-weighted sum of weights
    accum_weights = hg.accumulate_sequential(
        tree, sums, hg.Accumulators.sum) / areas.reshape([-1] + [1] *
                                                         (sums.ndim - 1))

    # get list [(a, [l0, ...])] where a is a candidate ancestor and [l0, ...] are its leaves

    parents = []
    for r in clicked_coords:
        clicked_label = label_map[tuple(r)]
        parents.extend(tree.ancestors(clicked_label)[1:])

    return accum_weights, np.array(parents)


def propagate_weights(tree, label_map, weight_map):
    # tree: input tree where leaves are superpixels
    # label_map: 2d array of integers of shape (M,N)
    # weight_map: 2d array of shape (M,N)
    #
    # return: propagated probabilities on each node of the tree

    regions = regionprops(label_map + 1, intensity_image=weight_map)
    # get area of all nodes
    area_leaves = np.array([p['area'] for p in regions])
    areas = hg.accumulate_sequential(tree, area_leaves, hg.Accumulators.sum)

    # get sum of proba on each leaf
    means = np.array([p['mean_intensity'] for p in regions])
    sums = means * area_leaves

    attribute = hg.accumulate_sequential(
        tree, sums, hg.Accumulators.sum) / areas.reshape([-1] + [1] *
                                                         (sums.ndim - 1))
    return attribute


class PbHierarchyExtractor:
    """
    Loads and augments images and ground truths
    """
    def __init__(self, root_path, *args, **kwargs):

        self.root_path = root_path

        self.model_root_path = os.path.expanduser(pjoin('~', '.models'))

        self.dl = LocPriorDataset(self.root_path, *args, **kwargs)
        self.do_pb()
        self.do_hierarchies()

    def get_model(self):
        if not os.path.exists(self.model_root_path):
            os.makedirs(self.model_root_path)

        model_weights_url = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
        self.model_weights_path = os.path.expanduser(
            pjoin(self.model_root_path, 'hed_pretrained_bsds.caffemodel'))
        self.model_arch_path = os.path.expanduser(
            pjoin(self.model_root_path, 'hed_pretrained_bsds.prototxt'))
        model_arch_url = 'https://raw.githubusercontent.com/legolas123/cv-tricks.com/master/OpenCV/Edge_detection/deploy.prototxt'

        if (not os.path.exists(self.model_weights_path)):
            print('Downloading HED model weights to {}'.format(
                self.model_weights_path))
            urllib.request.urlretrieve(model_weights_url,
                                       self.model_weights_path)

        if (not os.path.exists(self.model_arch_path)):
            print('Downloading HED model prototype to {}'.format(
                self.model_arch_path))
            urllib.request.urlretrieve(model_arch_url, self.model_arch_path)

    def do_pb(self):

        path = pjoin(self.root_path, 'precomp_desc', 'pb')
        # print('getting probability boundaries...')
        if (os.path.exists(path)):
            # print('found directory {}. Delete to re-run'.format(path))
            return
        else:
            os.makedirs(path)

        self.get_model()
        self.model = cv.dnn.readNet(self.model_arch_path,
                                    self.model_weights_path)
        cv.dnn_registerLayer('Crop', CropLayer)

        print('will save frames to {}'.format(path))
        pbar = tqdm(total=len(self.dl))
        for s in self.dl:
            im = (s['image'] * 255).astype(np.uint8)
            inp = cv.dnn.blobFromImage(im,
                                       scalefactor=1.0,
                                       size=(512, 512),
                                       mean=(104.00698793, 116.66876762,
                                             122.67891434),
                                       swapRB=False,
                                       crop=False)
            self.model.setInput(inp)
            out = self.model.forward()
            out = out[0, 0]
            out = cv.resize(out, (im.shape[1], im.shape[0]))
            out = 255 * out
            out = out.astype(np.uint8)

            io.imsave(pjoin(path, s['frame_name']), out)
            pbar.update(1)

    def do_hierarchies(self):
        path_trees = pjoin(self.root_path, 'precomp_desc', 'pb_trees')
        path_leaf_graphs = pjoin(self.root_path, 'precomp_desc',
                                 'pb_leaf_graphs')
        path_maps = pjoin(self.root_path, 'precomp_desc', 'pb_maps')
        # print('doing probability boundaries hierarchies...')
        if (os.path.exists(path_trees)):
            # print('found directory {}. Delete to re-run'.format(path_trees))
            return
        else:
            os.makedirs(path_trees)

        if (os.path.exists(path_leaf_graphs)):
            # print('found directory {}. Delete to re-run'.format(
            #     path_leaf_graphs))
            return
        else:
            os.makedirs(path_leaf_graphs)

        if (os.path.exists(path_maps)):
            # print('found directory {}. Delete to re-run'.format(path_maps))
            return
        else:
            os.makedirs(path_maps)

        print('will save trees to {}'.format(path_trees))
        print('will save leaf graphs to {}'.format(path_leaf_graphs))
        print('will save vertex/edge maps to {}'.format(path_maps))
        pbar = tqdm(total=len(self.dl))
        for s in self.dl:
            pb = io.imread(
                pjoin(self.root_path, 'precomp_desc', 'pb', s['frame_name']))

            graph = hg.get_4_adjacency_graph(pb.shape)
            edge_weights = hg.weight_graph(graph, pb, hg.WeightFunction.mean)
            rag, vertex_map, edge_map, tree, altitudes = hg.cpp._mean_pb_hierarchy(
                graph, pb.shape, edge_weights)

            hg.save_tree(
                pjoin(path_trees,
                      os.path.splitext(s['frame_name'])[0] + '.p'), tree,
                {'altitudes': altitudes})

            hg.save_graph_pink(
                pjoin(path_leaf_graphs,
                      os.path.splitext(s['frame_name'])[0] + '.p'), rag)
            np.savez(pjoin(path_maps,
                           os.path.splitext(s['frame_name'])[0]), **{
                               'vertex_map': vertex_map,
                               'edge_map': edge_map
                           })

            pbar.update(1)

    def rebuild_mean_pb_hierarchy(self, rag, vertex_map, edge_map, tree,
                                  altitudes, shape):

        shape = hg.normalize_shape(shape)
        graph = hg.get_4_adjacency_graph(shape)
        hg.CptRegionAdjacencyGraph.link(rag, graph, vertex_map, edge_map)
        hg.CptHierarchy.link(tree, rag)

        return tree, altitudes

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        fname = os.path.splitext(self.dl[idx]['frame_name'])[0]
        pb = io.imread(
            pjoin(self.root_path, 'precomp_desc', 'pb', fname + '.png'))
        tree, dict_ = hg.read_tree(
            pjoin(self.root_path, 'precomp_desc', 'pb_trees', fname + '.p'))
        altitudes = dict_['altitudes']

        rag = hg.read_graph_pink(
            pjoin(self.root_path, 'precomp_desc', 'pb_leaf_graphs',
                  fname + '.p'))[0]
        dict_ = np.load(
            pjoin(self.root_path, 'precomp_desc', 'pb_maps', fname + '.npz'))
        tree, altitudes = self.rebuild_mean_pb_hierarchy(
            rag, dict_['vertex_map'], dict_['edge_map'], tree, altitudes,
            pb.shape)
        # labels = get_cut(tree, altitudes)

        out = {
            'labels': dict_['vertex_map'].reshape(pb.shape),
            'tree': tree,
            'altitudes': altitudes,
            'rag': rag,
            'pb': pb
        }

        out_dset = self.dl[idx]
        out_dset.update(out)
        return out_dset


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ext = PbHierarchyExtractor(
        pjoin('/home/ubelix/lejeune/data/medical-labeling', 'Dataset00'), )
    ext.do_pb()
    ext.do_hierarchies()

    labels = ext[0]
    plt.imshow(labels)
    plt.show()
