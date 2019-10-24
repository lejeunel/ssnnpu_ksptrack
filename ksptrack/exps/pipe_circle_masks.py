from ksptrack import iterative_ksp
from ksptrack.cfgs import params
from os.path import join as pjoin

masks_paths_unet = {
    ('Dataset00', 'Dataset01', 'Dataset02', 'Dataset03', 'Dataset04', 'Dataset05'):
    'Dataset00_2019-08-06_13-48',
    ('Dataset10', 'Dataset11', 'Dataset12', 'Dataset13'):
    'Dataset10_2019-08-06_16-10',
    ('Dataset20', 'Dataset21', 'Dataset22', 'Dataset23', 'Dataset24', 'Dataset25'):
    'Dataset20_2019-08-06_11-46',
    ('Dataset30', 'Dataset31', 'Dataset32', 'Dataset33', 'Dataset34', 'Dataset35'):
    'Dataset30_2019-08-06_17-27'
}

if __name__ == "__main__":

    p = params.get_params()

    p.add('--in-path', required=True)
    p.add('--out-path', required=True)
    p.add('--root-path', required=True)
    p.add('--sets', nargs='+', required=True)

    cfg = p.parse_args()

    cfg.sets = ['Dataset{}'.format(set_) for set_ in cfg.sets]

    for dset in cfg.sets:
        cfg.in_path = pjoin(cfg.root_path, 'data/medical-labeling', dset)
        cfg.out_path = pjoin(cfg.root_path, 'runs/ksptrack', dset)

        # with entrance mask (u_net)
        mask_path = [v for k, v in masks_paths_unet.items() if (dset in k)][0]
        cfg.entrance_masks_path = pjoin(cfg.root_path, 'runs/unet_region',
                                        mask_path, dset, 'entrance_masks',
                                        'proba')
        iterative_ksp.main(cfg)

        # with entrance mask (selective search)
        cfg.entrance_masks_path = pjoin(cfg.root_path,
                                        'runs',
                                        'selective_search',
                                        dset,
                                        cfg.csv_fname)
        iterative_ksp.main(cfg)

        # with constant radius
        # cfg.entrance_masks_path = None
        # iterative_ksp.main(cfg)


