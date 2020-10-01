import configargparse
from unet_region.baselines.selective_search import make_loc_maps, main
from os.path import join as pjoin

if __name__ == "__main__":

    p = configargparse.ArgParser()

    #Paths, dirs, names ...
    p.add('--root-dir', type=str, required=True)
    p.add('--data-dirs', nargs='+', required=True)
    p.add('--out-dir', type=str, required=True)
    p.add('--label-dir', type=str, required=True)
    p.add('--csv-fname', type=str, required=True)
    p.add('--level', type=str, default='stack')
    p.add('--sigma', type=float, default=0.05)
    p.add('--k', type=int, default=50)
    p.add('--features',
          nargs='+',
          default=['size', 'color', 'texture', 'fill'])
    p.add('--alpha', type=float, default=1.)


    cfg = p.parse_args()

    for dset in cfg.data_dirs:
        cfg.data_dir = pjoin(cfg.root_dir, 'Dataset{}'.format(dset))
        cfg.label_path = pjoin(cfg.data_dir, 'precomp_desc', 'sp_labels.npz')
        cfg.csv_path = pjoin(cfg.data_dir, 'gaze-measurements', cfg.csv_fname)

        main.main(cfg)

        make_loc_maps.main(cfg)
