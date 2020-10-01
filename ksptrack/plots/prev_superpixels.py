import os
from ksptrack.utils import csv_utils as csv
from ksptrack.utils import my_utils as utls
from ksptrack.cfgs import params
import numpy as np
from skimage import (io, segmentation, transform)
import matplotlib.pyplot as plt
import tqdm
from os.path import join as pjoin
from ksptrack.utils.loc_prior_dataset import LocPriorDataset


def colorize(map_):
    cmap = plt.get_cmap('viridis')
    map_colorized = (cmap(map_)[..., :3] * 255).astype(np.uint8)

    return map_colorized


def do_one(cfg):
    dl = LocPriorDataset(cfg.in_path)

    # ---------- Descriptors/superpixel costs

    sample = dl[cfg.fin]

    truth = sample['label/segmentation']
    truth_ct = segmentation.find_boundaries(truth, mode='thick')
    im = sample['image']
    im = sample['loc_keypoints'].draw_on_image(im, size=10)

    labels = segmentation.slic(im, n_segments=200)
    labels_ct = segmentation.find_boundaries(labels)
    im[labels_ct, ...] = (255, 255, 255)
    im[truth_ct, ...] = (255, 0, 0)

    return im


if __name__ == "__main__":
    p = params.get_params()
    p.add('--root-path', required=True)
    p.add('--out-path', required=True)
    p.add('--data-dirs', nargs='+', type=str, required=True)
    p.add('--resize-h', type=int, default=512)
    p.add('--frames', nargs='+', type=int, default=[0])
    cfg = p.parse_args()

    assert len(cfg.data_dirs) == len(
        cfg.frames), print('give only one frame per sequence')

    paths = [pjoin(cfg.root_path, 'Dataset' + d) for d in cfg.data_dirs]

    ims = []
    for p, frame in zip(paths, cfg.frames):
        cfg.in_path = p
        cfg.fin = frame
        im_ = do_one(cfg)
        im_ = transform.resize(im_, (cfg.resize_h, im_.shape[1]))
        im_ = np.pad(im_, ((5, 5), (5, 5), (0, 0)),
                     constant_values=((1., 1.), (1., 1.), (1., 1.)))
        ims.append(im_)

    out = np.concatenate(ims, axis=1)
    io.imsave(cfg.out_path, out)
