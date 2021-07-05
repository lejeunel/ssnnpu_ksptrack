#!/usr/bin/env python3
from ksptrack.baselines.kato18 import run as run_one
from os.path import join as pjoin
from ksptrack import params
"""
Method for alternate class-prior estimation from paper

Kato, Masahiro, et al. "Alternate estimation of a classifier and the class-prior from positive and unlabeled data."
arXiv preprint arXiv:1809.05710 (2018).

We apply prior updates independently on each frame.
"""

if __name__ == "__main__":

    p = params.get_params('..')
    p.add('--out-path', required=True)
    p.add('--in-path', required=True)
    p.add('--dsets',
          default=[
              'Dataset00',
              'Dataset10',
              'Dataset20',
              'Dataset30',
              'Dataset01',
              'Dataset11',
              'Dataset21',
              'Dataset31',
              'Dataset02',
              'Dataset12',
              'Dataset22',
              'Dataset32',
              'Dataset03',
              'Dataset13',
              'Dataset23',
              'Dataset33',
          ])
    cfg = p.parse_args()

    cfg.epochs_pred = 50

    root_in_path = cfg.in_path
    root_out_path = cfg.out_path

    for ds in cfg.dsets:
        cfg.in_path = pjoin(root_in_path, ds)
        cfg.out_path = pjoin(root_out_path, ds)
        run_one(cfg)
