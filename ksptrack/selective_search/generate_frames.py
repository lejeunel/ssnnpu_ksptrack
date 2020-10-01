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



def main(cfg):

    print('loading prior maps')
    prior_paths = sorted(glob.glob(pjoin(cfg.run_dir,
                                         os.path.split(cfg.csv_path)[-1],
                                         'frame_*.png')))
    priors = {os.path.split(p)[-1]: io.imread(p) for p in prior_paths}

    out_dir = pjoin(cfg.run_dir, 'previews')
    print('will write frames to {}'.format(out_dir))
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    loader = Loader(cfg.data_dir, 'hand')

    locs = utls.readCsv(cfg.csv_path, as_pandas=True)
    for i, sample in enumerate(loader):
        if(sample['frame_name'] in priors.keys()):
            path = pjoin(out_dir, sample['frame_name'])
            if(not os.path.exists(path)):
                im = sample['image']

                locs_ = locs[locs['frame'] == sample['frame_idx']]
                locs_ = [utls.coord2Pixel(l['x'], l['y'], im.shape[1], im.shape[0])
                        for _, l in locs_.iterrows()][0]

                prior = priors[sample['frame_name']] / 255

                mask = prior > cfg.thr
                mask_bnd = segmentation.find_boundaries(mask, mode='thick')

                im[mask_bnd, :] = (1, 0, 0)
                rr, cc = draw.circle(locs_[0], locs_[1], 5, im.shape)
                im[rr, cc, :] = (0, 1, 0)

                all_ = np.concatenate((im,
                                    np.repeat(prior[..., None], 3, -1)), axis=1)

                print('writing {}'.format(path))
                io.imsave(path, all_)
            else:
                print('{} already exists'.format(path))


    return cfg


if __name__ == "__main__":

    p = configargparse.ArgParser()

    p.add('-v', help='verbose', action='store_true')

    #Paths, dirs, names ...
    p.add('--run-dir', type=str, required=True)
    p.add('--csv-path', type=str, required=True)
    p.add('--data-dir', type=str, required=True)
    p.add('--thr', type=float, required=True)

    cfg = p.parse_args()

    main(cfg)
