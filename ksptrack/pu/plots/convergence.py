#!/usr/bin/env python3

import numpy as np
from os.path import join as pjoin
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import configargparse
import pandas as pd
from ksptrack.pu.plots.tf2pd import tflog2pandas
from scipy.signal import find_peaks
from ksptrack.utils.base_dataset import BaseDataset
from torch.utils.data import DataLoader
import yaml
import os
from ksptrack.pu.modeling.unet import UNet

sns.set_style('darkgrid')

model = UNet(out_channels=1)


def main(cfg):

    assert len(cfg.eps) == len(
        cfg.frames), 'eps and frames should be same length'

    for e, f in zip(cfg.ep1, cfg.ep2, cfg.frames):
        dl = BaseDataset(pjoin(cfg.root_path, cfg.train_dir))
        cp = glob(
            pjoin(cfg.run_path, cfg.train_dir, cfg.exp_name, 'cps',
                  '*.pth.tar'))
        cps = sorted(
            glob(
                pjoin(cfg.run_path, cfg.train_dir, cfg.exp_name, 'cps',
                      '*.pth.tar')))
        cp_eps = np.array([
            int(os.path.split(f)[-1].split('_')[-1].split('.')[0]) for f in cps
        ])
        cp_fname = cps[np.argmin(np.abs(cp_eps - e))]
        path_ = pjoin(cfg.run_path, cfg.train_dir, cfg.exp_name, cp_fname)
        print('loading checkpoint {}'.format(path_))
        state_dict = torch.load(path_,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

        samp = dl.collate_fn([dl[f]])

        res = model(samp)


if __name__ == "__main__":
    p = configargparse.ArgParser()

    p.add('--root-path', required=True)
    p.add('--run-path', required=True)
    p.add('--train-dir', required=True)
    p.add('--exp-name', required=True)
    p.add('--curves-dir', default='curves_data')
    p.add('--eps', nargs='+', required=True)
    p.add('--frames', nargs='+', required=True)
    p.add('--save', default='')
    cfg = p.parse_args()

    main(cfg)
