import numpy as np
from ksptrack import iterative_ksp
from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils import my_utils as utls
from itertools import product


if __name__ == "__main__":

    p = params.get_params('../cfgs')

    p.add('--out-path', required=True)
    p.add('--root-path', required=True)
    p.add('--sets', nargs='+', required=True)

    cfg = p.parse_args()

    cfg.sets = ['Dataset{}'.format(set_) for set_ in cfg.sets]

    sigmas = [0.6, 0.8, 1., 1.2, 1.4]
    r_trans = [0.05, 0.08, 0.1, 0.12]
    combs = list(product(sigmas, r_trans))

    for i, dset in enumerate(cfg.sets):
        cfg.in_path = pjoin(cfg.root_path, 'data/medical-labeling', dset)
        cfg.out_path = pjoin(cfg.root_path, 'runs/ksptrack', dset)

        # constant radius ---------------------------------------------------
        cfg.feats_mode = 'autoenc'
        cfg.entrance_masks_path = None
        cfg.model_path = None

        for sig_, r in combs:
            cfg.exp_name = 'transexp_sig_{}_r_{}'.format(sig_, r)
            cfg.hoof_tau_u = 0
            cfg.ml_sigma = sig_
            cfg.norm_neighbor = r
            
            iterative_ksp.main(cfg)

