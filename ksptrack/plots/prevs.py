import glob
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from ksptrack.cfgs import params
from skimage import color, io
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
from ksptrack.utils.loc_prior_dataset import LocPriorDataset, draw_2d_loc, draw_gt_contour
import configargparse

if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--root-in-path', required=True)
    p.add('--shape', default=512)
    p.add('--dsets',
          nargs='+',
          type=str,
          default=[
              '00', '01', '02', '03', '10', '11', '12', '13', '20', '21', '22',
              '23', '30', '31', '32', '33'
          ])
    p.add('--fin',
          nargs='+',
          type=int,
          default=[
              39, 10, 50, 0, 37, 46, 49, 48, 68, 67, 22, 64, 27, 26, 26, 36
          ])
    p.add('--save-path', default='prevs.png')
    cfg = p.parse_args()

    cfg.csv_fname = 'video1.csv'
    cfg.locs_dir = 'gaze-measurements'
    cfg.coordconv = False

    ims = []
    for fin, dset in zip(cfg.fin, cfg.dsets):

        in_path = pjoin(cfg.root_in_path, 'Dataset' + dset)
        dl = LocPriorDataset(in_path, resize_shape=cfg.shape)
        sample = dl[fin]
        im = draw_gt_contour(sample['image'], sample['label/segmentation'])
        i, j = sample['annotations'].iloc[0].y, sample['annotations'].iloc[0].x
        im = draw_2d_loc(im, i, j)
        ims.append(im)

    n_cols = 4
    n_rows = 4
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.02)

    pos = 0
    for i, im in enumerate(ims):
        grid[pos].imshow(im)
        grid[pos].axis('off')
        pos += 1

    print('saving fig to {}'.format(cfg.save_path))
    plt.savefig(cfg.save_path, dpi=400, bbox_inches='tight')
