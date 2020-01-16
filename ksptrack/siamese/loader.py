import os
from os.path import join as pjoin
from skimage import io, segmentation, measure, future
import glob
import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import torch
import networkx as nx
from scipy import sparse
import itertools
from tqdm import tqdm
import pandas as pd

def make_1d_gauss(length, std, x0):

    x = np.arange(length)
    y = np.exp(-0.5 * ((x - x0) / std)**2)

    return y / np.sum(y)


def make_2d_gauss(shape, std, center):
    """
    Make object prior (gaussians) on center
    """

    g = np.zeros(shape)
    g_x = make_1d_gauss(shape[1], std, center[1])
    g_x = np.tile(g_x, (shape[0], 1))
    g_y = make_1d_gauss(shape[0], std, center[0])
    g_y = np.tile(g_y.reshape(-1, 1), (1, shape[1]))

    g = g_x * g_y

    return g / np.sum(g)



def coord2Pixel(x, y, width, height):
    """
    Returns i and j (line/column) coordinate of point given image dimensions
    """

    j = int(np.round(x * (width - 1), 0))
    i = int(np.round(y * (height - 1), 0))

    return i, j



def imread(path, scale=True):
    im = io.imread(path)

    if (im.dtype == 'uint16'):
        im = (im / 255).astype(np.uint8)

    if (scale):
        im = im / 255

    if (len(im.shape) < 3):
        im = np.repeat(im[..., None], 3, -1)

    if (im.shape[-1] > 3):
        im = im[..., 0:3]

    return im

def readCsv(csvName, seqStart=None, seqEnd=None):

    out = np.loadtxt(
        open(csvName, "rb"), delimiter=";", skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)

    return pd.DataFrame(data=out, columns=['frame', 'time', 'visible', 'x', 'y'])



class Loader:
    def __init__(self,
                 root_path,
                 augmentation=None,
                 normalization=None,
                 sig_prior=0.1,
                 nn_radius=0.1):
        """

        """
        locs2d_path = pjoin(root_path, 'gaze-measurements',
                                    'video1.csv')
        if(os.path.exists(locs2d_path)):
            print('found 2d locs file {}'.format(locs2d_path))
            self.locs2d = readCsv(locs2d_path)
        else:
            self.locs2d = None

        self.root_path = root_path

        self.sig_prior = sig_prior

        self.nn_radius = nn_radius

        self.augmentation = augmentation
        self.normalization = normalization

        self.do_siam_data = True

        self.ignore_collate = [
            'frame_idx', 'frame_name', 'rag', 'nn_graph', 'centroids',
            'labels_clicked'
        ]

        exts = ['*.png', '*.jpg', '*.jpeg']
        img_paths = []
        for e in exts:
            img_paths.extend(
                sorted(glob.glob(pjoin(root_path, 'input-frames', e))))
        truth_paths = []
        for e in exts:
            truth_paths.extend(
                sorted(
                    glob.glob(pjoin(root_path, 'ground_truth-frames', e))))
        self.truth_paths = truth_paths
        self.img_paths = img_paths

        self.truths = [
            io.imread(f).astype('bool') for f in self.truth_paths
        ]
        self.truths = [
            t if (len(t.shape) < 3) else t[..., 0] for t in self.truths
        ]
        self.imgs = [imread(f, scale=False) for f in self.img_paths]

        self.labels = np.load(pjoin(root_path, 'precomp_desc', 'sp_labels.npz'))['sp_labels']

        graphs_path = pjoin(root_path, 'precomp_desc', 'siam_graphs.npz')
        if(not os.path.exists(graphs_path)):
            self.prepare_graphs()
            np.savez(graphs_path, **{'rags': self.rags, 'nn_graphs': self.nn_graphs})
        else:
            print('loading graphs at {}'.format(graphs_path))
            np_file = np.load(graphs_path, allow_pickle=True)
            self.rags = np_file['rags']
            self.nn_graphs = np_file['nn_graphs']

    def get_fnames(self):
        return [os.path.split(p)[-1] for p in self.img_paths]

    def __len__(self):
        return len(self.imgs)

    def prepare_graphs(self):
        
        self.rags = []
        self.nn_graphs = []

        print('preparing graphs...')

        pbar = tqdm(total=len(self.imgs))
        for idx, (im, truth) in enumerate(zip(self.imgs, self.truths)):
            labels = self.labels[..., idx]
            regions = measure.regionprops(labels + 1, intensity_image=truth)

            bboxes = [p['bbox'] for p in regions]
            bboxes = [(b[1], b[0], b[3], b[2]) for b in bboxes]
            centroids = [(p['centroid'][1] / labels.shape[1],
                         p['centroid'][0] / labels.shape[0]) for p in regions]
            truth_sp = [p['mean_intensity'] > 0.5 for p in regions]

            node_list = [[
                label,
                dict(truth=truth_,
                     labels=[label],
                     centroid=centroid,
                     bbox=bbox_)
            ] for label, truth_, centroid, bbox_ in zip(np.unique(labels),
                                                        truth_sp,
                                                        centroids,
                                                        bboxes)]

            # region adjancency graph
            rag = future.graph.RAG(labels)
            rag.add_nodes_from(node_list)
            edges = [(n0, n1,
                      dict(truth_sim=rag.nodes[n0]['truth'] == rag.nodes[n1]['truth']))
                     for n0, n1 in rag.edges()]
            rag.add_edges_from(edges)

            # nearest neighbor graph
            nn_graph = nx.Graph()
            nn_graph.add_nodes_from(node_list)
            node_label_list = [n[0] for n in node_list]
            nodes_ = np.array(np.meshgrid(node_label_list,
                                          node_label_list)).T.reshape(-1, 2)
            centroids_x = [n[1]['centroid'][0] for n in node_list]
            centroids_x = np.array(np.meshgrid(centroids_x,
                                               centroids_x)).T.reshape(-1, 2)
            centroids_y = [n[1]['centroid'][1] for n in node_list]
            centroids_y = np.array(np.meshgrid(centroids_y,
                                               centroids_y)).T.reshape(-1, 2)
            centroids_ = np.concatenate((centroids_x, centroids_y), axis=1)

            dists = np.sqrt((centroids_[:, 0] - centroids_[:, 1])**2 +
                            (centroids_[:, 2] - centroids_[:, 3])**2)
            inds = np.argwhere(dists < self.nn_radius).ravel()

            edges = [(nodes_[i, 0], nodes_[i, 1],
                      dict(weight=nn_graph.nodes[nodes_[i, 0]]['truth'] ==
                           nn_graph.nodes[nodes_[i, 1]]['truth'])) for i in inds]
            nn_graph.add_edges_from(edges)

            self.rags.append(rag)
            self.nn_graphs.append(nn_graph)
            pbar.update(1)
        pbar.close()

    def __getitem__(self, idx):

        truth = self.truths[idx]
        im = self.imgs[idx]

        shape = im.shape

        if (self.augmentation is not None):
            truth = ia.SegmentationMapsOnImage(truth, shape=truth.shape)
            seq_det = self.augmentation.to_deterministic()
            im = seq_det.augment_image(im)
            truth = seq_det.augment_segmentation_maps([truth])[0].get_arr()

        im_unnormal = im.copy()

        if (self.normalization is not None):
            im = self.normalization.augment_image(im)

        truth = truth[..., None]

        labels_clicked = []

        if(self.locs2d is not None):
            locs = self.locs2d[self.locs2d['frame'] == idx]
            locs = [coord2Pixel(l['x'], l['y'], shape[1], shape[0])
                    for _, l in locs.iterrows()]

            keypoints = ia.KeypointsOnImage(
                [ia.Keypoint(x=l[1], y=l[0]) for l in locs],
                shape=(shape[0], shape[1]))
            if(self.augmentation is not None):
                keypoints = seq_det.augment_keypoints([keypoints])[0]

            labels_clicked += [self.labels[kp.y, kp.x, idx]
                               for kp in keypoints.keypoints]

            if (len(locs) > 0):
                obj_prior = [
                    make_2d_gauss((shape[0], shape[1]),
                                       self.sig_prior * max(shape),
                                       (kp.y, kp.x)) for kp in keypoints.keypoints
                ]
                obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
                obj_prior += obj_prior.min()
                obj_prior /= obj_prior.max()
            else:
                obj_prior = (
                    np.ones((shape[0], shape[1])))[..., None]
        else:
            obj_prior = np.zeros((self.in_shape[0], self.in_shape[1]))[..., None]


        # make tensor of bboxes
        rag = self.rags[idx]
        bboxes = np.array([rag.nodes[n]['bbox'] for n in rag.nodes()])

        out = {
            'image': im,
            'image_unnormal': im_unnormal,
            'frame_idx': idx,
            'labels_clicked': labels_clicked,
            'frame_name': os.path.split(self.img_paths[idx])[-1],
            'rag': self.rags[idx],
            'nn_graph': self.nn_graphs[idx],
            'labels': self.labels[..., idx][..., None],
            'bboxes': bboxes,
            'label/segmentation': truth
        }
        return out

    def collate_fn(self, samples):
        out = {k: [dic[k] for dic in samples] for k in samples[0]}

        for k in out.keys():
            if (k not in self.ignore_collate):
                if(k == 'bboxes'):
                    bboxes = [np.concatenate((i*np.ones((bboxes_.shape[0], 1)), bboxes_), axis=1)
                                              for i, bboxes_ in enumerate(out[k])]
                    bboxes = np.concatenate(bboxes, axis=0)
                    out[k] = torch.from_numpy(bboxes)
                out[k] = np.array(out[k])
                out[k] = np.rollaxis(out[k], -1, 1)
                out[k] = torch.from_numpy(out[k]).float().squeeze(-1)

        return out
