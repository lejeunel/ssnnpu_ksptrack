import datetime
import os
import networkx as nx
from os.path import join as pjoin
import yaml
import tqdm
from skimage import io, draw, segmentation
import numpy as np
import configargparse
import selective_search
import glob
import pandas as pd
from unet_region import utils as utls
from unet_region.loader import Loader
import matplotlib.pyplot as plt
import selective_search


def apply_prior(label, g, prior, field='stack'):
    levels = np.unique([g.node[n][field] for n in g.nodes])

    h0 = np.zeros(label.shape).astype(np.float32)
    for l in np.unique(label):
        h0 += np.max(prior[label == l]) * (label == l)
    H = [h0]

    relabeled = False

    F0 = label

    #Remap if labels are not contiguous
    sorted_labels = np.asarray(sorted(np.unique(F0).ravel()))
    if (np.any((sorted_labels[1:] - sorted_labels[0:-1]) > 1)):
        relabeled = True
        map_dict = {sorted_labels[i]: i for i in range(sorted_labels.shape[0])}
        F0 = selective_search.relabel(F0, map_dict)

    F = [F0]  # stores label image for each step

    for i, s in enumerate(levels[1:]):

        # get label of parents
        l_parents = [n for n in g.nodes if (g.node[n][field] == s)]

        new_pooled_image = H[-1].copy()
        new_label_image = F[-1].copy()
        for l_p in l_parents:

            # get label of children
            l_c = [n for n in g.successors(l_p)]

            new_label_image = selective_search._new_label_image(
                new_label_image, l_c[0], l_c[1], l_p)
            new_pooled_image[new_label_image == l_p] = prior[new_label_image ==
                                                             l_p].max()
            F.append(new_label_image)
        H.append(new_pooled_image)

    return H


def main(cfg):
    locs = utls.readCsv(cfg.csv_path, as_pandas=True)

    print('loading labels: {}'.format(cfg.label_path))
    labels = np.load(cfg.label_path)['sp_labels']

    print('loading merge graphs')
    graphs_paths = sorted(glob.glob(pjoin(cfg.run_dir, 'frame_*.p')))
    graphs = [nx.read_gpickle(p) for p in graphs_paths]

    out_dir = pjoin(cfg.run_dir, os.path.split(cfg.csv_path)[-1])
    print('will write frames to {}'.format(out_dir))
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    loader = Loader(cfg.data_dir, 'hand')
    bar = tqdm.tqdm(total=labels.shape[-1])
    for i, sample in enumerate(loader):

        locs_ = locs[locs['frame'] == sample['frame_idx']]
        path = pjoin(out_dir, sample['frame_name'])
        if(not os.path.exists(path)):
            if (locs_.size > 0):
                label = labels[..., locs_['frame'].values[0]]
                locs_ = [
                    utls.coord2Pixel(l['x'], l['y'], label.shape[1],
                                    label.shape[0]) for _, l in locs_.iterrows()
                ]

                prior = [
                    utls.make_2d_gauss(label.shape, cfg.sigma * max(label.shape),
                                    (l[0], l[1])) for l in locs_
                ]
                prior = np.array(prior).sum(0)
                prior = prior / prior.max()

                g = graphs[i]

                H = np.array(apply_prior(label, g, prior, cfg.level))

                H = np.mean(H, axis=0)

                io.imsave(path,
                        (H * 255).astype(np.uint8))
        else:
            print('{} already exists'.format(path))

        bar.update(1)
    bar.close()

    return cfg


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('-v', help='verbose', action='store_true')

    #Paths, dirs, names ...
    p.add('--run-dir', type=str, required=True)
    p.add('--label-path', type=str, required=True)
    p.add('--csv-path', type=str, required=True)
    p.add('--data-dir', type=str, required=True)
    p.add('--sigma', type=float, default=0.05)
    p.add('--level', type=str, default='stack')

    cfg = p.parse_args()

    main(cfg)
