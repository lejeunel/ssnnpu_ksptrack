import numpy as np
from ksptrack import iterative_ksp
from ksptrack.cfgs import params
from os.path import join as pjoin
from ksptrack.utils import my_utils as utls
from itertools import product
from sklearn.model_selection import ParameterGrid


if __name__ == "__main__":

    p = params.get_params('../cfgs')

    p.add('--out-path', required=True)
    p.add('--root-path', required=True)
    p.add('--siam-run-path', default='')
    p.add('--sets', nargs='+', required=True)

    cfg = p.parse_args()

    cfg.sets = ['Dataset{}'.format(set_) for set_ in cfg.sets]
    params_ = {'hoof_tau_u': [0., 0.5],
            'norm_neighbor': [0., 0.08]}
    param_grid = ParameterGrid(params_)

    for i, dset in enumerate(cfg.sets):
        cfg.in_path = pjoin(cfg.root_path, 'data/medical-labeling', dset)
        cfg.out_path = pjoin(cfg.root_path, 'runs/ksptrack', dset)

        # constant radius ---------------------------------------------------
        cfg.feats_mode = 'autoenc'

        for params__ in param_grid:
            cfg.ml_up_thr = 0.7
            cfg.ml_down_thr = 0.3
            cfg.hoof_tau_u = params__['hoof_tau_u']
            cfg.norm_neighbor = params__['norm_neighbor']
            cfg.exp_name = 'transexp_up_{:.2f}_down_{:.2f}_neigh_{:.2f}_hoof_{:.2f}'.format(
                cfg.ml_up_thr,
                cfg.ml_down_thr,
                cfg.norm_neighbor,
                cfg.hoof_tau_u)
            # print(dset)
            # print(cfg.exp_name)
            # print(params__)
            iterative_ksp.main(cfg)
