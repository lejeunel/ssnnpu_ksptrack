import datetime
import os
import networkx as nx
from os.path import join as pjoin
from ksptrack.utils.base_dataset import BaseDataset
import yaml
import tqdm
from skimage import io, draw, segmentation
import numpy as np
import configargparse
from . import selective_search
from . import features as F
import matplotlib.pyplot as plt


def main(cfg):

    d = datetime.datetime.now()

    # ds_dir = os.path.split(cfg.data_dir)[-1]

    loader = BaseDataset(cfg.data_dir)

    if (not os.path.exists(cfg.out_dir)):
        os.makedirs(cfg.out_dir)

    # Save cfg
    with open(pjoin(cfg.out_dir, 'cfg.yml'), 'w') as outfile:
        yaml.dump(cfg.__dict__, stream=outfile, default_flow_style=False)

    for i, sample in enumerate(loader):

        print('{}/{}'.format(i + 1, len(loader)))
        path = pjoin(cfg.out_dir,
                     '{}.p'.format(os.path.splitext(sample['frame_name'])[0]))
        if (not os.path.exists(path)):
            label = np.squeeze(sample['labels'])
            R, _, g = selective_search.hierarchical_segmentation(
                sample['image'], F0=label)

            nx.write_gpickle(g, path)
        else:
            print('{} already exists'.format(path))

    return cfg


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('-v', help='verbose', action='store_true')

    #Paths, dirs, names ...
    p.add('--data-dir', type=str, required=True)
    p.add('--out-dir', type=str, required=True)
    p.add('--k', type=int, default=50)
    p.add('--features',
          nargs='+',
          default=['size', 'color', 'texture', 'fill'])
    p.add('--alpha', type=float, default=1.)

    cfg = p.parse_args()

    main(cfg)
