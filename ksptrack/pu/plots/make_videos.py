#!/usr/bin/env python3
from ksptrack.cfgs import params
import configargparse
from ksptrack.utils.loc_prior_dataset import LocPriorDataset
from os.path import join as pjoin
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from skimage.draw import disk, circle_perimeter
from skimage import (io, segmentation, transform)
import os
from tqdm import tqdm
from glob import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2


def colorize(map_):
    cmap = plt.get_cmap('viridis')
    map_colorized = (cmap(map_)[..., :3] * 255).astype(np.uint8)

    return map_colorized


def make_video(in_path, out_path):
    # ffmpeg -r 1/5 -i img%03d.png -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
    os.system(
        "ffmpeg -r 10 -i {}/im_%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p {}"
        .format(in_path, out_path))

    print('wrote video ', out_path)


def make_frames(cfg, dset):

    in_path = pjoin(cfg.root_in_path, 'Dataset' + dset)
    loader = LocPriorDataset(in_path)

    tmp_dir = pjoin(cfg.save_path, 'tmp_' + dset)

    if not os.path.exists(tmp_dir):
        print('creating temporary directory ', tmp_dir)
        os.makedirs(tmp_dir)

    res_path = pjoin(cfg.root_run_path, 'Dataset' + dset, cfg.exp_name,
                     'results')
    print('loading results from ', res_path)

    res = sorted(glob(pjoin(res_path, '*')))
    res = [os.path.split(r)[-1] for r in res]
    res = [re.match('^im_\d{4}.png$', f) for f in res]
    res = [pjoin(res_path, r[0]) for r in res if r]
    print('got {} frames'.format(len(res)))
    segs = [io.imread(f) for f in res]

    pbar = tqdm(total=len(loader))
    for i, (sample, seg) in enumerate(zip(loader, segs)):

        im = sample['image']
        cntr = segmentation.find_boundaries(sample['label/segmentation'],
                                            mode='thick')
        im[cntr, ...] = (255, 0, 0)

        if not sample['annotations'].empty:
            r = sample['annotations'].iloc[0]
            x, y = r['x'], r['y']
            h, w = r['h'], r['w']

            rr, cc = disk((y, x), radius=10, shape=(h, w))
            im[rr, cc, :] = (0, 255, 0)

            im = np.concatenate((im, colorize(seg)), axis=1)

            if im.shape[0] % 2:
                im = np.concatenate(
                    (im,
                     np.zeros((1, im.shape[1], im.shape[-1]), dtype=np.uint8)),
                    axis=0)

            if im.shape[1] % 2:
                im = np.concatenate(
                    (im,
                     np.zeros((im.shape[0], 1, im.shape[-1]), dtype=np.uint8)),
                    axis=1)

            io.imsave(pjoin(tmp_dir, 'im_{:04d}.png'.format(i)), im)
        pbar.update(1)
    pbar.close()

    print('saved frames to ', tmp_dir)


def main(cfg):
    for dset in cfg.dsets:
        make_frames(cfg, dset)
        make_video(pjoin(cfg.save_path, 'tmp_' + dset),
                   pjoin(cfg.save_path, 'video_' + dset + '.mp4'))


if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--root-in-path', required=True)
    p.add('--root-run-path', required=True)
    p.add('--exp-name', default='pu_piovrs_1.4_ph2')
    p.add('--dsets', nargs='+', type=str, default=['02', '10', '23', '31'])
    p.add('--disk-radius', default=0.05)
    p.add('--save-path', default='')
    cfg = p.parse_args()
    main(cfg)
